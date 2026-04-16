/*
    KUPL 多队列事件编程 Demo
    编译命令: clang++ queue_event.cpp -o queue_event -lkupl
    运行命令: KUPL_SCHED_POLICY=mq KUPL_EXECUTOR_BACKEND=pthread taskset -c 0 ./queue_event
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

    auto q1 = kupl_queue_create();
    auto q2 = kupl_queue_create();
    auto event = kupl_event_create();
    data_axb_t data1 = {
        .A_data = &A,
        .B_data = &B,
        .F_data = &F
    };
    kupl_queue_item_desc_t desc1 = {
        .field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_NAME,
        .func = fun_axb,
        .args = &data1,
        .name = "axb"
    };
    // 提交到队列 q1
    kupl_queue_submit(q1, &desc1);
    // 提交到队列 q2, 和 q2 同步事件
    data_cxd_t data2 = {
        .C_data = &C,
        .D_data = &D,
        .G_data = &G
    };
    kupl_queue_item_desc_t desc2 = {
        .field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_NAME,
        .func = fun_cxd,
        .args = &data2,
        .name = "cxd"
    };
    // 提交到队列 q2，并为 q1 的同步进行事件记录
    kupl_queue_submit(q2, &desc2);
    kupl_event_record(event, q2);

    // 同步队列 q1 的事件
    kupl_queue_wait_event(q1, event);

    data_fpg_t data3 = {
        .E_data = &E,
        .F_data = &F,
        .G_data = &G
    };
    kupl_queue_item_desc_t desc3 = {
        .field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_NAME,
        .func = fun_fpg,
        .args = &data3,
        .name = "fpg"
    };
    kupl_queue_submit(q1, &desc3);

    // 等待队列 q1
    kupl_queue_wait(q1);
    printf("E = A * B + C * D, E = %d\n", E);

    kupl_event_destroy(event);
    kupl_queue_destroy(q2);
    kupl_queue_destroy(q1);

    return 0;
}