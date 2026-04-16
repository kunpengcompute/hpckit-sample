/*
    KUPL 计算机图编程 lambda匿名函数接口版本实现 Demo
    编译命令: clang++ graph_lambda.cpp -o graph_lambda -O3 -lkupl
    运行命令: KUPL_SCHED_POLICY=mq KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-1 ./graph_lambda
*/

#include <stdio.h>
#include <unistd.h>
#include "kupl.h"

int main()
{
    int A = 1, B = 2, C = 3, D = 4, E = 0, F = 0, G = 0;

    auto sgraph = kupl_sgraph_create();

    // Step1: 向静态图中添加计算节点
    kupl_sgraph_node_desc_t node1_desc = {
        .field_mask = KUPL_SGRAPH_NODE_DESC_FIELD_NAME,
        .name = "axb",
    };
    auto node1 = kupl::sgraph_add_node(sgraph, &node1_desc, [&]() { F = A * B; });

    kupl_sgraph_node_desc_t node2_desc = {
        .field_mask = KUPL_SGRAPH_NODE_DESC_FIELD_NAME,
        .name = "cxd",
    };
    auto node2 = kupl::sgraph_add_node(sgraph, &node2_desc, [&]() { G = C * D; });

    kupl_sgraph_node_desc_t node3_desc = {
        .field_mask = KUPL_SGRAPH_NODE_DESC_FIELD_NAME,
        .name = "fpg",
    };
    auto node3 = kupl::sgraph_add_node(sgraph, &node3_desc, [&]() { E = F + G; });

    // Step2: 向静态图中添加节点依赖
    kupl_sgraph_add_dep(node1, node3);
    kupl_sgraph_add_dep(node2, node3);

    // Step3: 提交任务并执行
    auto graph = kupl_graph_create(KUPL_ALL_EXECUTORS);
    kupl_sgraph_task_desc_t desc = {
        .field_mask = 0,
        .sgraph = sgraph
    };
    kupl_task_info_t info = {
        .type = KUPL_TASK_TYPE_SGRAPH,
        .desc = &desc
    };

    // 提交静态图任务到动态图中
    kupl_graph_submit(graph, &info);
    // 等待提交的静态图任务执行完毕
    kupl_graph_wait(graph);
    printf("E = A * B + C * D = %d\n", E);
    kupl_graph_destroy(graph);

    kupl_sgraph_destroy(sgraph);
}