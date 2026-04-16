/*
    KUPL int8精度Copy转置 矩阵编程 Demo
    编译命令: clang++ copy_int8.cpp -o copy_int8 -O3 -lkupl
    运行命令: taskset -c 0 ./copy_int8
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "kupl_mma.h"
using namespace kupl::tensor;  // 引用 kupl::tensor 命名空间从而使能矩阵编程能力

#define MATRIX_M 16
#define MATRIX_N 576

void init_matrix_data(int8_t *data_src, int8_t *data_dst);
void print_matrix(int8_t *data, int m, int n);

int main()
{
    // 用户原始矩阵数据 Buffer 创建初始化
    int8_t *data_src = (int8_t*)malloc(sizeof(int8_t) * MATRIX_M * MATRIX_N);
    int8_t *data_dst = (int8_t*)malloc(sizeof(int8_t) * MATRIX_M * MATRIX_N);
    init_matrix_data(data_src, data_dst);

    // KUPL Shape 对象：描述计算矩阵的形状
    auto shape_d = make_shape(Int<MATRIX_M>{}, make_shape(Int<4>{}, Int<MATRIX_N / 4>{}));
    auto shape_s = make_shape(Int<MATRIX_M>{}, Int<MATRIX_N>{});

    // KUPL Stride 对象：描述计算矩阵的各个维度步长，此处的 Stride 描述要与 copy 转置的 src/dst 矩阵一致
    auto stride_d = make_stride(Int<4>{}, make_stride(Int<1>{}, Int<64>{}));
    auto stride_s = make_stride(Int<MATRIX_N>{}, Int<1>{});

    // KUPL Layout 对象：描述计算矩阵的排布，包含矩阵形状和各维度步长
    auto layout_d = make_layout(shape_d, stride_d);
    auto layout_s = make_layout(shape_s, stride_s);

    // 创建 KUPL copy Ops 方法用于后续 copy 行为：具体基于底层 Ops 按照 atom_shape 进行各维度计算尺寸拓展
    auto atom_copy_shape = make_shape(Int<1>{}, Int<MATRIX_N / 4>{});
    auto tiled_copy = make_tiled_copy(Ops<COPY_16x4_INT8_RM2ZZ>{}, atom_copy_shape);

    // 创建 KUPL Tensor 对象，用于 mma 和 store 行为
    auto tensor_d = make_tensor(data_dst, layout_d);
    auto tensor_s = make_tensor(data_src, layout_s);
    tensor_tiled_copy(tiled_copy, tensor_d, tensor_s);

    print_matrix(data_dst, MATRIX_M, MATRIX_N);
    
    free(data_dst);
    free(data_src);

    return 0;
}

void init_matrix_data(int8_t *data_src, int8_t *data_dst)
{
    for (int i = 0; i < MATRIX_M; i++) {        // row-major
        for (int j = 0; j < MATRIX_N; j++) {
            data_src[i * MATRIX_N + j] = (i * MATRIX_N + j) % 100;
            data_dst[i * MATRIX_N + j] = 0;
        }
    }
}

void print_matrix(int8_t *data, int m, int n) {
    printf("matrix_print begin\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%3d ", data[j / 4 * m * 4 + i * 4 + j % 4]);
        }
        printf("\n");
    }
    printf("matrix_print end\n");
}