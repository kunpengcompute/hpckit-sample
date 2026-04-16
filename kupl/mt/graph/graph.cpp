/*
    KUPL 计算机图编程 Demo
    编译命令: clang++ graph.cpp -o graph -O3 -lkupl
    运行命令: KUPL_SCHED_POLICY=mq KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-1 ./graph
*/

#include <stdio.h>
#include <unistd.h>
#include "kupl.h"

struct data_axb_t {
    int *A_data;
    int *B_data;
    int *F_data;
};

struct data_cxd_t {
    int *C_data;
    int *D_data;
    int *G_data;
};

struct data_fpg_t {
    int *E_data;
    int *F_data;
    int *G_data;
};

static void fun_axb(void *args) {
    data_axb_t *data = (data_axb_t *)args;
    int *A = data->A_data;
    int *B = data->B_data;
    int *F = data->F_data;
    (*F) = (*A) * (*B);
}

static void fun_cxd(void *args) {
    data_cxd_t *data = (data_cxd_t *)args;
    int *C = data->C_data;
    int *D = data->D_data;
    int *G = data->G_data;
    (*G) = (*C) * (*D);
}

static void fun_fpg(void *args) {
    data_fpg_t *data = (data_fpg_t *)args;
    int *E = data->E_data;
    int *F = data->F_data;
    int *G = data->G_data;
    (*E) = (*F) + (*G);
}

int main()
{
    int A = 1, B = 2, C = 3, D = 4, E = 0, F = 0, G = 0;

    auto sgraph = kupl_sgraph_create();

    // Step1: 向静态图中添加计算节点
    data_axb_t data1 = {
        .A_data = &A,
        .B_data = &B,
        .F_data = &F
    };
    kupl_sgraph_node_desc_t node1_desc = {
        .field_mask = 0,
        .func = fun_axb,
        .args = &data1
    };
    auto node1 = kupl_sgraph_add_node(sgraph, &node1_desc);

    data_cxd_t data2 = {
        .C_data = &C,
        .D_data = &D,
        .G_data = &G
    };
    kupl_sgraph_node_desc_t node2_desc = {
        .field_mask = 0,
        .func = fun_cxd,
        .args = &data2
    };
    auto node2 = kupl_sgraph_add_node(sgraph, &node2_desc);

    data_fpg_t data3 = {
        .E_data = &E,
        .F_data = &F,
        .G_data = &G
    };
    kupl_sgraph_node_desc_t node3_desc = {
        .field_mask = 0,
        .func = fun_fpg,
        .args = &data3
    };
    auto node3 = kupl_sgraph_add_node(sgraph, &node3_desc);

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