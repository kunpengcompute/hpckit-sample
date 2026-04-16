/*
    KUPL fp64精度MMA 矩阵编程 Demo
    编译命令: clang++ mma_fp64.cpp -o mma_fp64 -O3 -lkupl
    运行命令: taskset -c 0 ./mma_fp64
*/
#include <stdio.h>
#include <stdlib.h>
#include "kupl_mma.h"
using namespace kupl::tensor;  // 引用 kupl::tensor 命名空间从而使能矩阵编程能力

#define MATRIX_M 32
#define MATRIX_N 16
#define MATRIX_K 512

void init_matrix_data(double *data_a, double *data_b, double *data_c);
void print_matrix(double *data, int m, int n);

int main()
{
    // 用户原始矩阵数据 Buffer 创建
    double *data_a = (double*)malloc(sizeof(double) * MATRIX_M * MATRIX_K); // col-major
    double *data_b = (double*)malloc(sizeof(double) * MATRIX_K * MATRIX_N); // row-major
    double *data_c = (double*)malloc(sizeof(double) * MATRIX_M * MATRIX_N); // rou-major
    init_matrix_data(data_a, data_b, data_c);

    // KUPL Shape 对象：描述计算矩阵的形状
    auto shape_a = make_shape(Int<32>{}, Int<512>{});
    auto shape_b = make_shape(Int<512>{}, Int<16>{});
    auto shape_c = make_shape(Int<32>{}, Int<16>{});

    // KUPL Stride 对象：描述计算矩阵的各个维度步长
    auto stride_a = make_stride(Int<1>{}, Int<32>{});
    auto stride_b = make_stride(Int<16>{}, Int<1>{});
    auto stride_c = make_stride(Int<16>{}, Int<1>{});

    // KUPL Layout 对象：描述计算矩阵的排布，包含矩阵形状和各维度步长
    auto layout_a = make_layout(shape_a, stride_a);
    auto layout_b = make_layout(shape_b, stride_b);
    auto layout_c = make_layout(shape_c, stride_c);

    // 创建 KUPL mma Ops 方法用于后续 mma 行为：具体基于底层 Ops 按照 atom_shape 进行各维度计算尺寸拓展
    auto atom_mma_shape = make_shape(Int<1>{}, Int<1>{}, Int<512>{});
    auto tiled_mma = make_tiled_mma(Ops<MMA_32x16x1_F64F64F64>{}, atom_mma_shape);
    // 创建 KUPL store Ops 方法用于后续 store 行为：具体基于底层 Ops 按照 atom_shape 进行各维度写回尺寸扩展
    auto atom_store_shape = make_shape(Int<1>{}, Int<1>{});
    auto tiled_store = make_tiled_store(Ops<STORE_32x16_F64>{}, atom_store_shape);

    // 创建 KUPL Tensor 对象，用于 mma 和 store 行为
    auto tensor_a = make_tensor(data_a, layout_a);
    auto tensor_b = make_tensor(data_b, layout_b);
    auto tensor_c = make_tensor(data_c, layout_c);
    tensor_tiled_mma(tiled_mma, tensor_c, tensor_a, tensor_b, tensor_c);
    tensor_tiled_store(tiled_store, tensor_c);

    print_matrix(data_c, MATRIX_M, MATRIX_N);
    
    free(data_c);
    free(data_b);
    free(data_a);

    return 0;
}

void init_matrix_data(double *data_a, double *data_b, double *data_c)
{
    for (int i = 0; i < MATRIX_K; i++) {            // col-major
        for (int j = 0; j < MATRIX_M; j++) {
            if (i == j) {
                data_a[i * MATRIX_M + j] = 1.0;
            } else {
                data_a[i * MATRIX_M + j] = 0.0;
            }
        }
    }

    for (int i = 0; i < MATRIX_K; i++) {           // row-major
        for (int j = 0; j < MATRIX_N; j++) {
            data_b[i * MATRIX_N + j] = 1.0 * (i * MATRIX_N + j);
        }
    }

    for (int i = 0; i < MATRIX_M; i++) {           // row-major
        for (int j = 0; j < MATRIX_N; j++) {
            data_c[i * MATRIX_N + j] = 0.0;
        }
    }
}

void print_matrix(double *data, int m, int n) {
    printf("matrix_print begin\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%5.0lf ", data[i * n + j]);
        }
        printf("\n");
    }
    printf("matrix_print end\n");
}