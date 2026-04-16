/*
    KUPL 数据拷贝 2D Demo
    编译命令: clang++ memcpy2d.cpp -o memcpy2d -O3 -lkupl
    运行命令: ./memcpy2d
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "kupl.h"

bool array_2d_check(char *array_src, char *array_dst, int height, int width, int spitch, int dpitch)
{
    int diffnum = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (array_src[i * spitch + j] != array_dst[i * dpitch + j]) {
                diffnum++;
            }
        }
    }
    return diffnum == 0;
}

int main()
{
    const int len = 65536;
    char *src = (char *)kupl_malloc(len);
    char *dest = (char *)kupl_malloc(len);
    for (int i = 0; i < len / sizeof(char); i++) {
        src[i] = i;
        dest[i] = 0;
    }
    int height = 2, width = 200;
    int spitch = 300, dpitch = 400;
    int ret = kupl_memcpy2d(dest, dpitch, src, spitch, width, height);
    assert(ret == KUPL_OK);
    bool res = array_2d_check(src, dest, height, width / sizeof(char), spitch, dpitch);
    printf("memcpy2d result: %d\n", res);
    kupl_free(src);
    kupl_free(dest);
    return 0;
}