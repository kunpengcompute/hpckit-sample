/*
    KUPL 数据拷贝 1D Demo
    编译命令: clang++ memcpy.cpp -o memcpy -O3 -lkupl
    运行命令: ./memcpy
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "kupl.h"

bool array_1d_check(char *array_src, char *array_dst, int sz)
{
    int diffnum = 0;
    for (int i = 0; i < sz; i++) {
        if (array_src[i] != array_dst[i]) {
            diffnum++;
        }
    }
    return diffnum == 0;
}

int main()
{
    int len = 1024;
    char *src = (char *)kupl_malloc(len);
    char *dest = (char *)kupl_malloc(len);
    for (int i = 0; i < len / sizeof(char); i++) {
        src[i] = i;
        dest[i] = 0;
    }
    int ret = kupl_memcpy(dest, src, len);
    assert(ret == KUPL_OK);
    bool res = array_1d_check(src, dest, len / sizeof(char));
    printf("memcpy result: %d\n", res);
    kupl_free(src);
    kupl_free(dest);
    return 0;
}