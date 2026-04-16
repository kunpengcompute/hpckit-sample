/*
    KUPL parallel for 3d 并行 Demo
    编译命令: clang++ parallel_for_3d.cpp -o parallel_for_3d -lkupl
    运行命令: KUPL_EXECUTOR_BACKEND=pthread taskset -c 0-7 ./parallel_for_3d
*/

#include <stdio.h>
#include "kupl.h"
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

#define HEIGHT 10
#define LENGTH 7
#define WIDTH 5

static int A[HEIGHT][LENGTH][WIDTH];
static int B[HEIGHT][LENGTH][WIDTH];
int C[HEIGHT][LENGTH][WIDTH];

static void array_3d_init() {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < LENGTH; j++) {
            for (int k = 0; k < WIDTH; k++) {
                A[i][j][k] = i + j + k;
                B[i][j][k] = i + j + k;
                C[i][j][k] = 0;
            }
        }
    }
}

static bool array_3d_check() {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < LENGTH; j++) {
            for (int k = 0; k < WIDTH; k++) {
                if (C[i][j][k] != A[i][j][k] + B[i][j][k]) {
                    return false;
                }
            }
        }
    }
    return true;
}

static inline void task_int_loop3d(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    // 打印每个线程每次任务执行的for循环LOOP范围
    printf("tid: %d\n--range2_page: lower %zu upper %zu\n--range1_row: lower %zu upper %zu\n--range0_col: lower %zu upper %zu\n",
           tid, nd_range->nd_range[2].lower, nd_range->nd_range[2].upper, nd_range->nd_range[1].lower,
           nd_range->nd_range[1].upper, nd_range->nd_range[0].lower, nd_range->nd_range[0].upper);
    for (int i = nd_range->nd_range[2].lower;
         i < nd_range->nd_range[2].upper; i += nd_range->nd_range[2].step) {
        for (int j = nd_range->nd_range[1].lower;
             j < nd_range->nd_range[1].upper; j += nd_range->nd_range[1].step) {
            for (int k = nd_range->nd_range[0].lower;
                 k < nd_range->nd_range[0].upper; k += nd_range->nd_range[0].step) {
                C[i][j][k] = A[i][j][k] + B[i][j][k];
            }
        }
        
    }
}

int main()
{
    array_3d_init();
    const int NUM_THREADS = kupl_get_num_executors();
    printf("num_thread: %d\n", NUM_THREADS);
    kupl_nd_range_t range;
    KUPL_3D_RANGE_INIT(range, 0, HEIGHT, 0, LENGTH, 0, WIDTH);

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
        .policy = KUPL_LOOP_POLICY_STATIC      // 此处可切换不同的策略，观察执行效果，对比static与dynamic策略的执行区别
    };
    
    kupl_parallel_for(&desc, task_int_loop3d, nullptr);
    bool res = array_3d_check();
    printf("parallel for 3d result: %d\n", res);

    kupl_egroup_destroy(eg);
    return 0;
}