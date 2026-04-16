/*
    KUPL 动态并发度伸缩 parallel for实现 Demo
    编译命令: clang++ -o dynamic_parallelism_scaling dynamic_parallelism_scaling.cpp -lkupl
    运行命令: KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-15 ./dynamic_parallelism_scaling
*/

#include "kupl.h"
#include <unistd.h>

/*
场景示例行为如下：
    thead 0     thead 1 
        |          |
        V          V
    +-------+   +-------+
    | taskB |   |       |
    +-------+   |       |
        |       |       |
        V       |       |
    +-----------+       | <- 此处发生动态的并发度扩展，taskA获取0号线程资源参与任务执行
    |      taskA        |
    +-------------------+
        |           |
        V           V        
    +-------------------+  
    |      taskC        |
    +-------------------+
*/

static kupl_egroup_h g_egroup;

static inline void loop_func(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    auto &range = nd_range->nd_range[0];
    for (int i = range.lower; i < range.upper; i += range.step) {
        usleep(1000);
    }
}

void node_a_func(void *args)
{
    auto egroup = (kupl_egroup_h)args; 
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, 100);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = egroup,
        .concurrency = KUPL_CONCURRENCY_DEFAULT,
        .policy = KUPL_LOOP_POLICY_STATIC,
    };

    // 任务a需要花费相对更多的时间去执行完毕
    for (int i = 0; i < 100; i++) {
        // 当任务b执行完后会释放资源到g_egroup，从而任务a可以获取到所有线程资源进行加速
        kupl_egroup_borrow(egroup, g_egroup);
        kupl_parallel_for(&desc, loop_func, nullptr);
    }

    kupl_egroup_return(g_egroup, egroup);
    kupl_egroup_reset(egroup);
}

void node_b_func(void *args)
{
    auto egroup = (kupl_egroup_h)args; 
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, 100);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = egroup,
        .concurrency = KUPL_CONCURRENCY_DEFAULT,
        .policy = KUPL_LOOP_POLICY_STATIC,
    };

    // 任务b需要花费相对更少的时间去执行完毕
    for (int i = 0; i < 10; i++) {
        kupl_egroup_borrow(egroup, g_egroup);
        kupl_parallel_for(&desc, loop_func, nullptr);
    }

    // 任务b执行完毕后释放线程资源至g_egroup，从而可以供任务a获取
    kupl_egroup_return(g_egroup, egroup);
    kupl_egroup_reset(egroup);
}

void node_c_func(void *args)
{
    auto egroup = (kupl_egroup_h)args; 
    kupl_egroup_reset(g_egroup);
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, 100);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = egroup,
        .concurrency = KUPL_CONCURRENCY_DEFAULT,
        .policy = KUPL_LOOP_POLICY_STATIC,
    };

    for (int i = 0; i < 10; i++) {
        kupl_parallel_for(&desc, loop_func, nullptr);
    }
}

kupl_egroup_h egroup_create(int start_eid, int executors_num)
{
    int exe[executors_num];
    for (int i = 0; i < executors_num; i++) {
        exe[i] = start_eid + i;
    }
    return kupl_egroup_create(exe, executors_num);
}

kupl_sgraph_node_h add_node(kupl_sgraph_h sgraph, kupl_sgraph_node_func_t node_func, kupl_egroup_h egroup)
{
    kupl_sgraph_node_desc_t desc = {
        .field_mask = KUPL_SGRAPH_NODE_DESC_FIELD_EGROUP,
        .func = node_func,
        .args = egroup,
        .egroup = egroup,
    };
    return kupl_sgraph_add_node(sgraph, &desc);
}

int main()
{
    auto graph = kupl_graph_create(nullptr);
    auto sgraph = kupl_sgraph_create();
    // g_egroup表示全局可用线程资源，用于资源的释放和借用
    g_egroup = egroup_create(0, 0);
    // egroup_a/b/c表示a/b/c计算任务的初始线程资源
    auto egroup_a = egroup_create(0, 8);
    auto egroup_b = egroup_create(8, 8);
    auto egroup_c = egroup_create(0, 16);

    // 通过静态图定义a/b/c任务的计算依赖关系
    auto node_a = add_node(sgraph, node_a_func, egroup_a);
    auto node_b = add_node(sgraph, node_b_func, egroup_b);
    auto node_c = add_node(sgraph, node_c_func, egroup_c);
    kupl_sgraph_add_dep(node_a, node_c);
    kupl_sgraph_add_dep(node_b, node_c);
    // 提交静态图任务至动态图中从而执行
    kupl_sgraph_task_desc_t desc = {
        .field_mask = 0,
        .sgraph = sgraph
    };
    kupl_task_info_t info = {
        .type = KUPL_TASK_TYPE_SGRAPH,
        .desc = &desc,
    };
    kupl_graph_submit(graph, &info);
    kupl_graph_wait(graph);

    kupl_egroup_destroy(g_egroup);
    kupl_egroup_destroy(egroup_a);
    kupl_egroup_destroy(egroup_b);
    kupl_egroup_destroy(egroup_c);
    kupl_sgraph_destroy(sgraph);
    kupl_graph_destroy(graph);

    return 0;
}