/*
    KUPL 异步数据拷贝 1D Demo
    编译命令: clang++ memcpy_async.cpp -o memcpy_async -O3 -lkupl
    运行命令: KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-1 ./memcpy_async
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kupl.h"

static const size_t len = 67108864;
int c[len] = {0};

static void task_compute(void *args)
{
    for (int i = 0; i < len; i++) {
        c[i] = i + 1;
    }
    printf("task_compute\n");
}

bool array_1d_check(char *array_src, char *array_dst)
{
    int diffnum = 0;
    for (int i = 0; i < len / sizeof(char); i++) {
        if (array_src[i] != array_dst[i]) {
            diffnum++;
        }
    }
    return diffnum == 0;
}

bool array_check()
{
    int diffnum = 0;
    for (int i = 0; i < len; i++) {
        if (c[i] != i + 1) {
            diffnum++;
        }
    }
    return diffnum == 0;
}

void test_kupl_memcpy_async() {
    char *src = (char *)kupl_malloc(len);
    char *dst = (char *)kupl_malloc(len);
    for (size_t i = 0; i < len; i++) {
        src[i] = (char)(i + 1);
        dst[i] = 0;
    }
    kupl_event_h event = kupl_event_create();
    kupl_queue_h q1 = kupl_queue_create();
    kupl_queue_h q2 = kupl_queue_create();

    kupl_memcpy_async(dst, src, len, q1, event);

    kupl_queue_item_desc_t desc = {
        .field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_NAME,
        .func = task_compute,
        .args = nullptr,
        .name = "task_compute"
    };
    int ret = kupl_queue_submit(q2, &desc);
    kupl_queue_wait(q2);
    kupl_event_wait(event);

    kupl_queue_destroy(q2);
    kupl_queue_destroy(q1);
    kupl_event_destroy(event);

    bool res1 = array_1d_check(src, dst);
    bool res2 = array_check();
    printf("memcpy result: %d\n", res1);
    printf("compute result: %d\n", res2);

    kupl_free(src);
    kupl_free(dst);
    return;
}

int main(int argc, char **argv)
{
    test_kupl_memcpy_async();

    return 0;
}