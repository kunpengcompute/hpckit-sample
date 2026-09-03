/*
    KUPL 计算与数据拷贝相互隐藏 Demo
    编译命令: clang++ compute_memcpy_overlap.cpp -o compute_memcpy_overlap -O3 -lkupl
    运行命令: taskset -c 0 ./compute_memcpy_overlap
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "kupl.h"

static const size_t len = 67108864;
static const size_t loop = 16;
int src[len];
int dst[len];
kupl_event_h event;

// 计算任务
void task_compute(size_t now_loop)
{
    for (size_t i =  len / loop * now_loop; i < len / loop * (now_loop + 1); i++) {
        dst[i] = dst[i] * 2;
    }
}

bool array_check()
{
    int diffnum = 0;
    for (size_t i = 0; i < len; i++) {
        if (dst[i] != (int)(i + 1) * 2) {
            diffnum++;
        }
    }
    return diffnum == 0;
}

int main(int argc, char **argv)
{
    for (size_t i = 0; i < len; i++) {
        src[i] = (int)(i + 1);
        dst[i] = 0;
    }
    event = kupl_event_create();

    kupl_memcpy_async(dst, src, len / loop * sizeof(int), nullptr, event);
    // 第i轮的数据计算与第i+1的数据拷贝做隐藏
    // 数据计算: [0] -> [1] -> [2] -> ... -> [loop - 1] -> [loop]
    //                ^     ^      ^                    ^
    //               /     /      /                  /
    //              /     /      /                /
    // 数据拷贝: [1] -> [2] -> [3] -> ... -> [loop]
    for (size_t i = 0; i < loop; i++) {
        // 第i轮数据计算任务需要等待第i轮的异步数据拷贝任务完成后才能够执行
        kupl_event_wait(event);
        // 提交第i+1轮的异步数据拷贝任务
        if (i < loop - 1) {
            kupl_memcpy_async(&dst[len / loop * (i + 1)], &src[len / loop * (i + 1)], len / loop * sizeof(int), nullptr, event);
        }
        // 执行第i轮的数据计算任务
        task_compute(i);
    }
    bool res = array_check();
    printf("result: %d\n", res);

    kupl_event_destroy(event);

    return 0;
}
