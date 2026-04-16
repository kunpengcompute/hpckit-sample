#include <stdio.h>
#include <unistd.h>
#include "kupl.h"
#include <stdlib.h>
#include <math.h>
#include <string>
#include <cstring>
#include <time.h>
#include <numa.h>
#include <sys/mman.h>
#include <numaif.h>

int innerreps = 10000;

double getclock() {
    struct timespec nowtime;
    clock_gettime(CLOCK_MONOTONIC, &nowtime);
    return nowtime.tv_sec + 1.0e-9 * nowtime.tv_nsec;
}

static size_t len;

void init_memcpy() 
{
    printf("The memcpy len is:");
    scanf("%zu", &len);
    printf("\n");
}

void bind_memory_to_numa(void * ptr, int node) {
    nodemask_t nodemask;
    // 假定目标机器有32个numa
    struct bitmask bitmask = {32, nodemask.n};

    numa_bitmask_clearall(&bitmask);
    numa_bitmask_setbit(&bitmask, node);

    // 使用 mbind 绑定这块内存到指定的节点
    int ret = mbind(ptr, len, MPOL_BIND, nodemask.n, 32, 0);
    if (ret != 0) {
        printf("mbind failed\n");
    } else {
        printf("Allocated memory at %p on NUMA node %d\n", ptr, node);
    }
}

void test_memcpy1d()
{
    double total = 0;
    // 使用mmap申请内存，其中MAP_HUGETLB代表使用大页
    char *src = (char *)mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    // 将申请的内存绑定到0号numa节点上
    bind_memory_to_numa(src, 0);

    char *dst = (char *)mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    // 将申请的内存绑定到16号numa节点上
    bind_memory_to_numa(dst, 16);   

    kupl_mlock(src, len);
    kupl_mlock(dst, len);
    for (size_t i = 0; i < len / sizeof(char); i++) {
        src[i] = (char)(i + 1);
        dst[i] = 0;
    }
    for (int i = 0; i < 100; i++) {
        kupl_memcpy(dst, src, len);
    }
    for (int i = 0; i < innerreps; i++) {
        double start = getclock();
        kupl_memcpy(dst, src, len);
        double end = getclock();
        total += end - start;
    }

    kupl_munlock(src, len);
    kupl_munlock(dst, len);

    // 计算数据拷贝的时延与带宽
    printf("memcpy1d:%f\n", total * 1.0e6/innerreps);
    uint64_t bandwidth;
    bandwidth = ((uint64_t)len * innerreps) / total/ 1.0e6;
    bandwidth = bandwidth * 1000 / 1024;
    bandwidth = bandwidth * 1000 / 1024;
    printf("bandwidth:%ldMB/s\n", bandwidth);
    munmap(src,len);
    munmap(dst,len);
    return;
}

void finalize_memcpy()
{
    return;
}

int main() {
    init_memcpy(); 

    test_memcpy1d();

    finalize_memcpy();
    return 0;
}
