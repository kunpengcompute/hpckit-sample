/*
    计算图版本实现
    编译指令：clang++ mt_kupl_graph_version.cpp -o mt_kupl_graph_version -O3 -lkupl
    运行指令：
        4核运行指令：KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-3 ./mt_kupl_graph_version
        32核运行指令：KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-31 ./mt_kupl_graph_version
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "kupl.h"

uint64_t getclock()
{
    struct timespec nowtime;
    clock_gettime(CLOCK_MONOTONIC, &nowtime);
    return 1000000 * nowtime.tv_sec + nowtime.tv_nsec / 1000;   // us
}

const int LOOP = 100;
const int N = 10000;
double *d, *a, *b, *c;
int num_executors;

void axb_func(void *args)
{
    int *id = (int*)args;
    for (int i = (int)(1ll * (*id) * N / num_executors); i < (int)(1ll * ((*id) + 1) * N / num_executors); i++) {
        for (int j = 0; j < i; j++) {
            d[i] += a[i] * b[i];
        }
    }
}

void d_func(void *args)
{
    int *id = (int*)args;
    for (int i = (int)(1ll * (*id) * N / num_executors); i < (int)(1ll * ((*id) + 1) * N / num_executors); i++) {
        for (int j = 0; j < (N - i); j++) {
            d[i] += c[i];
        }
    }
}

void test()
{
    // 静态图创建
    auto sgraph = kupl_sgraph_create();
    num_executors = kupl_get_num_executors();
    int nums[num_executors];
    for (int i = 0; i < num_executors; i++) {
        nums[i] = i;
    }
    // 建立静态图中的点边关系，将整个数组划分成多段，对每一段中的 d = a x b 和 d += c 计算行为建立依赖关系
    for (int i = 0; i < num_executors; i++) {
        kupl_sgraph_node_desc_t node_axb_desc = {
            .field_mask = 0,
            .func = axb_func,
            .args = &nums[i]
        };
        auto sgraph_node_axb = kupl_sgraph_add_node(sgraph, &node_axb_desc);
        kupl_sgraph_node_desc_t node_d_desc = {
            .field_mask = 0,
            .func = d_func,
            .args = &nums[i]
        };
        auto sgraph_node_d = kupl_sgraph_add_node(sgraph, &node_d_desc);
        kupl_sgraph_add_dep(sgraph_node_axb, sgraph_node_d);
    }
    // 将静态图包装成任务提交给动态图并阻塞执行
    auto graph = kupl_graph_create(KUPL_ALL_EXECUTORS);
    kupl_sgraph_task_desc_t sgraph_desc = {
        .field_mask = 0,
        .sgraph = sgraph
    };
    kupl_task_info_t task_info = {
        .type = KUPL_TASK_TYPE_SGRAPH,
        .desc = &sgraph_desc
    };
    kupl_graph_submit(graph, &task_info);
    kupl_graph_wait(graph);
    kupl_graph_destroy(graph);
    kupl_sgraph_destroy(sgraph);
}

int main()
{
    d = (double *)malloc(sizeof(double) * N);
    a = (double *)malloc(sizeof(double) * N);
    b = (double *)malloc(sizeof(double) * N);
    c = (double *)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        d[i] = 0;
        a[i] = i;
        b[i] = 1;
        c[i] = N - i;
    }
    // 执行LOOP次，取平均耗时从而体现性能差异
    uint64_t start_clock = getclock();
    for (int loop = 0; loop < LOOP; loop++) {
        test();
    }
    uint64_t end_clock = getclock();
    uint64_t timecost = end_clock - start_clock;
    printf("cost time = %lu us\n", timecost / LOOP);
    bool res = true;
    for (int i = 0; i < N; i++) {
        if ((d[i] / LOOP) !=  ((N - i) * (N - i)) + (double)(i * i)) {
            res = false;
            break;
        }
    }
    if (res) {
        printf("compute correctly\n");
    } else {
        printf("compute failed\n");
    }
    free(c);
    free(b);
    free(a);
    free(d);
    return 0;
}