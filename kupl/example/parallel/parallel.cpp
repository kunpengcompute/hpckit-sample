/*
    KUPL parallel 并行域中 egroup barrier 同步Demo
    编译命令: clang++ parallel.cpp -o parallel -lkupl
    运行命令: KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-7 ./parallel
*/

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <unistd.h>
#include <cstdlib>
#include "kupl.h"

kupl_egroup_h egroup;

// parallel kernel算子，其中nd_range入参可不再使用，通过tid/tnum操作即可
static inline void task_egroup_barrier(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    printf("in parallel: tid %d before egroup barrier\n", tid);
    kupl_egroup_barrier(egroup);
    printf("in parallel: tid %d after egroup barrier\n", tid);
}

int main()
{
    const int num_threads = kupl_get_num_executors();
    // 设置range的下界为0，上界为num_threads线程数即可实现parallel for模拟parallel效果
    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, num_threads);
    int exe[num_threads];
    for (int i = 0; i < num_threads; i++) {
        exe[i] = i;
    }
    egroup = kupl_egroup_create(exe, num_threads);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = egroup,
        .concurrency = num_threads,
        .policy = KUPL_LOOP_POLICY_STATIC
    };

    kupl_parallel_for(&desc, task_egroup_barrier, &egroup);

    kupl_egroup_destroy(egroup);

    return 0;
}
