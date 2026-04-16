/*
    KUPL parallel for 1d 并行 Demo
    编译命令: clang++ parallel_for.cpp -o parallel_for -lkupl
    运行命令: KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-7 ./parallel_for
*/

#include <stdio.h>
#include "kupl.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

static const int A[] = {1, 2, 3, 4, 6, 8, 9, 17, 18, 20, 32, 67, 83, 128};
static const int B[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 16, 17, 23, 24};
int C[14];

static inline void task_int_loop(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    // 打印每个线程每次任务执行的for循环LOOP范围
    printf("tid: %d\n--range: lower %zu upper %zu\n", tid, nd_range->nd_range[0].lower, nd_range->nd_range[0].upper);
    for (int i = nd_range->nd_range[0].lower; i < nd_range->nd_range[0].upper; i += nd_range->nd_range[0].step) {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const int NUM_THREADS = kupl_get_num_executors();
    printf("num_thread: %d\n", NUM_THREADS);
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, 14);

    int executor[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) {
        executor[i] = i;
    }
    kupl_egroup_h eg = kupl_egroup_create(executor, NUM_THREADS);

    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = eg,
        .concurrency = NUM_THREADS,
        .policy = KUPL_LOOP_POLICY_STATIC       // 此处可切换不同的策略，观察执行效果，对比static与dynamic策略的执行区别
    };
    kupl_parallel_for(&desc, task_int_loop, nullptr);
    for (int i = range.nd_range[0].lower; i < range.nd_range[0].upper; i += range.nd_range[0].step) {
        printf("C[%d] result: %d\n", i, C[i]);
    }
    kupl_egroup_destroy(eg);
    return 0;
}