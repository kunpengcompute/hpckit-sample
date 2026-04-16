/*
    parallel for版本实现
    编译指令：clang++ mt_kupl_pf_version.cpp -o mt_kupl_pf_version -O3 -lkupl
    运行指令：
        4核运行指令：KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-3 ./mt_kupl_pf_version
        32核运行指令：KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-31 ./mt_kupl_pf_version
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

void axb_func(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    for (int i = nd_range->nd_range[0].lower; i < nd_range->nd_range[0].upper; i += nd_range->nd_range[0].step) {
        for (int j = 0; j < i; j++) {
            d[i] += a[i] * b[i];
        }
    }
}

void d_func(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    for (int i = nd_range->nd_range[0].lower; i < nd_range->nd_range[0].upper; i += nd_range->nd_range[0].step) {
        for (int j = 0; j < (N - i); j++) {
            d[i] += c[i];
        }
    }
}

void test()
{
    // parallel for操作所需基础信息的创建
    int num_executors = kupl_get_num_executors();
    int eids[num_executors];
    for (int i = 0; i < num_executors; i++) {
        eids[i] = i;
    }
    kupl_egroup_h eg = kupl_egroup_create(eids, num_executors);
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, N);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = eg,
        .concurrency = num_executors,
        .policy = KUPL_LOOP_POLICY_STATIC
    };

    // 串行提交 d = a x b 和 d += c 这两个parallel for并行任务
    kupl_parallel_for(&desc, axb_func, nullptr);
    kupl_parallel_for(&desc, d_func, nullptr);

    kupl_egroup_destroy(eg);
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