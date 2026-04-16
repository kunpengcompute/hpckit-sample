/*
    KUPL 多队列并行编程 Demo
    编译命令: clang++ queue_submit.cpp -o queue_submit -lkupl
    运行命令: KUPL_SCHED_POLICY=static_mq KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-19 ./queue_submit
*/

#include <stdio.h>
#include <unistd.h>
#include <atomic>
#include <random>
#include <time.h>
#include "kupl.h"

int main()
{
    auto q1 = kupl_queue_create();
    auto q2 = kupl_queue_create();
    const int group_size = 10;
    const int loop_size = group_size;
    int executor1[group_size];
    int executor2[group_size];
    for (int i = 0; i < group_size; i++) {
        executor1[i] = i;
        executor2[i] = i + 10;
    }
    kupl_egroup_h egroup1 = kupl_egroup_create(executor1, group_size);
    kupl_egroup_h egroup2 = kupl_egroup_create(executor2, group_size);

    srand((unsigned)time(NULL));
    const size_t arr_size = 100000;
    int arr1[arr_size] = {0};
    int arr2[arr_size] = {0};
    for (size_t i = 0; i < arr_size; i++) {
        arr1[i] = rand() & 0xFF;
        arr2[i] = rand() & 0xFF;
    }
    std::atomic<int> sum1(0);
    std::atomic<int> sum2(0);

    kupl_nd_range_t range;
    KUPL_1D_RANGE_INIT(range, 0, loop_size);
    kupl_queue_kernel_desc_t desc1 = {
        .field_mask = 0,
        .range = &range,
        .egroup = egroup1
    };

    // 并行计算 arr1 中所有元素的和，提交到队列 q1
    kupl::queue_submit(q1, &desc1, [&](const kupl_nd_range_t *nd_range) {
        size_t start_index = nd_range->nd_range[0].lower * (arr_size / loop_size);
        size_t local_sum = 0;
        for (size_t i = 0; i < arr_size / loop_size; i++) {
            local_sum += arr1[start_index + i];
        }
        sum1 += local_sum;
    });

    kupl_queue_kernel_desc_t desc2 = {
        .field_mask = 0,
        .range = &range,
        .egroup = egroup2
    };
    // 并行计算 arr2 中所有元素的和，提交到队列 q2
    kupl::queue_submit(q2, &desc2, [&](const kupl_nd_range_t *nd_range) {
        size_t start_index = nd_range->nd_range[0].lower * (arr_size / loop_size);
        size_t local_sum = 0;
        for (size_t i = 0; i < arr_size / loop_size; i++) {
            local_sum += arr2[start_index + i];
        }
        sum2 += local_sum;
    });
    kupl_queue_wait(q1);
    kupl_queue_wait(q2);

    printf("sum1: %d, sum2: %d\n", sum1.load(), sum2.load());

    kupl_egroup_destroy(egroup1);
    kupl_egroup_destroy(egroup2);
    kupl_queue_destroy(q1);
    kupl_queue_destroy(q2);

    return 0;
}