/*
    KUPL 多队列 parallel for 实现多流算子并行 Demo
    编译命令: clang++ -o kernel_parallelism_queue kernel_parallelism_queue.cpp -lkupl
    运行命令: KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-15 ./kernel_parallelism_queue
*/

#include "kupl.h"
#include <unistd.h>

static inline void loop_func(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    printf("do loop_func, eid: %d\n", kupl_get_executor_num());
    sleep(1);
}

static void do_parallel_for(void *args)
{
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, 100);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = nullptr,
        .concurrency = KUPL_CONCURRENCY_DEFAULT,
        .policy = KUPL_LOOP_POLICY_STATIC,
    };
    kupl_parallel_for(&desc, loop_func, nullptr);
}

void submit_parallel_for(kupl_queue_h queue, kupl_egroup_h egroup)
{
    // 定义queue上任务的描述，其中指定egroup资源从而确认任务执行的线程集
    kupl_queue_item_desc_t item_desc = {
        .field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_EGROUP,
        .func = do_parallel_for,
        .args = nullptr,
        .egroup = egroup
    };
    kupl_queue_submit(queue, &item_desc);
}

kupl_egroup_h egroup_create(int start_eid, int executors_num)
{
    int exe[executors_num];
    for (int i = 0; i < executors_num; i++) {
        exe[i] = start_eid + i;
    }
    return kupl_egroup_create(exe, executors_num);
}

int main()
{
    int num_executors = kupl_get_num_executors();
    int executors_per_queue = num_executors / 4;
    // 创建KUPL queue队列资源
    auto q0 = kupl_queue_create();
    auto q1 = kupl_queue_create();
    auto q2 = kupl_queue_create();
    auto q3 = kupl_queue_create();
    // 将多线程任务分别绑定到各个线程集从而提交到各个queue队列上执行
    auto egroup0 = egroup_create(executors_per_queue * 0, executors_per_queue);
    auto egroup1 = egroup_create(executors_per_queue * 1, executors_per_queue);
    auto egroup2 = egroup_create(executors_per_queue * 2, executors_per_queue);
    auto egroup3 = egroup_create(executors_per_queue * 3, executors_per_queue);
    submit_parallel_for(q0, egroup0);
    submit_parallel_for(q1, egroup1);
    submit_parallel_for(q2, egroup2);
    submit_parallel_for(q3, egroup3);
    // 等待提交到队列上的任务执行完毕
    kupl_queue_wait(q0);
    kupl_queue_wait(q1);
    kupl_queue_wait(q2);
    kupl_queue_wait(q3);
    kupl_egroup_destroy(egroup0);
    kupl_egroup_destroy(egroup1);
    kupl_egroup_destroy(egroup2);
    kupl_egroup_destroy(egroup3);
    kupl_queue_destroy(q0);
    kupl_queue_destroy(q1);
    kupl_queue_destroy(q2);
    kupl_queue_destroy(q3);
    return 0;
}