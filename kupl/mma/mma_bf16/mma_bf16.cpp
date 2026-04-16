/*
    KUPL bf16精度MMA 矩阵编程 Demo
    编译命令: clang++ mma_bf16.cpp -o mma_bf16 -O3 -lkupl
    运行命令: taskset -c 0 ./mma_bf16
*/
#include <stdio.h>
#include <stdlib.h>
#include <arm_bf16.h>
#include "kupl_mma.h"
using namespace kupl::tensor;  // 引用 kupl::tensor 命名空间从而使能矩阵编程能力

#define MATRIX_M 16
#define MATRIX_N 64
#define MATRIX_K 576

void init_matrix_data(bfloat16_t *data_a, bfloat16_t *data_b, float *data_c);
void prepack_matrix_a(bfloat16_t *data_a, bfloat16_t *pack_data_a, int m, int k);
void prepack_matrix_b(bfloat16_t *data_b, bfloat16_t *pack_data_b, int k, int n);
void print_matrix(float *data, int m, int n);

int main()
{
    // 用户原始矩阵数据 Buffer 创建
    bfloat16_t *data_a = (bfloat16_t*)malloc(sizeof(bfloat16_t) * MATRIX_M * MATRIX_K); // row-major
    bfloat16_t *data_b = (bfloat16_t*)malloc(sizeof(bfloat16_t) * MATRIX_K * MATRIX_N); // col-major
    float *data_c = (float*)malloc(sizeof(float) * MATRIX_M * MATRIX_N); // rou-major
    init_matrix_data(data_a, data_b, data_c);
    bfloat16_t *pack_data_a = (bfloat16_t*)malloc(sizeof(bfloat16_t) * MATRIX_M * MATRIX_K);
    bfloat16_t *pack_data_b = (bfloat16_t*)malloc(sizeof(bfloat16_t) * MATRIX_K * MATRIX_N);
    prepack_matrix_a(data_a, pack_data_a, MATRIX_M, MATRIX_K);
    prepack_matrix_b(data_b, pack_data_b, MATRIX_K, MATRIX_N);

    // KUPL Shape 对象：描述计算矩阵的形状
    auto shape_a = make_shape(Int<16>{}, make_shape(Int<2>{}, Int<288>{}));
    auto shape_b = make_shape(make_shape(Int<2>{}, Int<288>{}), Int<64>{});
    auto shape_c = make_shape(Int<16>{}, Int<64>{});

    // KUPL Stride 对象：描述计算矩阵的各个维度步长，此处的 Stride 描述要与 prepack 转置成的排布步长一致
    auto stride_a = make_stride(Int<2>{}, make_stride(Int<1>{}, Int<32>{}));
    auto stride_b = make_stride(make_stride(Int<1>{}, Int<128>{}), Int<2>{});
    auto stride_c = make_stride(Int<64>{}, Int<1>{});

    // KUPL Layout 对象：描述计算矩阵的排布，包含矩阵形状和各维度步长
    auto layout_a = make_layout(shape_a, stride_a);
    auto layout_b = make_layout(shape_b, stride_b);
    auto layout_c = make_layout(shape_c, stride_c);

    // 创建 KUPL mma Ops 方法用于后续 mma 行为：具体基于底层 Ops 按照 atom_shape 进行各维度计算尺寸拓展
    auto atom_mma_shape = make_shape(Int<1>{}, Int<1>{}, Int<288>{});
    auto tiled_mma = make_tiled_mma(Ops<MMA_16x64x2_BF16BF16F32>{}, atom_mma_shape);
    // 创建 KUPL store Ops 方法用于后续 store 行为：具体基于底层 Ops 按照 atom_shape 进行各维度写回尺寸扩展
    auto atom_store_shape = make_shape(Int<1>{}, Int<1>{});
    auto tiled_store = make_tiled_store(Ops<STORE_16x64_F32>{}, atom_store_shape);

    // 创建 KUPL Tensor 对象，用于 mma 和 store 行为
    auto tensor_a = make_tensor(pack_data_a, layout_a);
    auto tensor_b = make_tensor(pack_data_b, layout_b);
    auto tensor_c = make_tensor(data_c, layout_c);
    tensor_tiled_mma(tiled_mma, tensor_c, tensor_a, tensor_b, tensor_c);
    tensor_tiled_store(tiled_store, tensor_c);

    print_matrix(data_c, MATRIX_M, MATRIX_N);
    
    free(pack_data_b);
    free(pack_data_a);
    free(data_c);
    free(data_b);
    free(data_a);

    return 0;
}

void init_matrix_data(bfloat16_t *data_a, bfloat16_t *data_b, float *data_c)
{
    for (int i = 0; i < MATRIX_M; i++) {        // row-major
        for (int j = 0; j < MATRIX_K; j++) {
            if (i == j) {
                data_a[i * MATRIX_K + j] = (bfloat16_t)1;
            } else {
                data_a[i * MATRIX_K + j] = (bfloat16_t)0;
            }
        }
    }

    for (int i = 0; i < MATRIX_N; i++) {        // col-major
        for (int j = 0; j < MATRIX_K; j++) {
            data_b[i * MATRIX_K + j] = (bfloat16_t)((j * MATRIX_N + i) % 256);
        }
    }

    for (int i = 0; i < MATRIX_M; i++) {           // row-major
        for (int j = 0; j < MATRIX_N; j++) {
            data_c[i * MATRIX_N + j] = (float)0;
        }
    }
}

// 重拍A矩阵排布，从而满足MMA_16x64x2_BF16BF16F32原子方法的排布需要
void prepack_matrix_a(bfloat16_t *data_a, bfloat16_t *pack_data_a, int m, int k)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            pack_data_a[j / 2 * m * 2 + i * 2 + j % 2] = data_a[i * k + j];
        }
    }
}

// 重拍B矩阵排布，从而满足MMA_16x64x2_BF16BF16F32原子方法的排布需要
void prepack_matrix_b(bfloat16_t *data_b, bfloat16_t *pack_data_b, int k, int n)
{
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            pack_data_b[j / 2 * n * 2 + i * 2 + j % 2] = data_b[i * k + j];
        }
    }
}

void print_matrix(float *data, int m, int n) {
    printf("matrix_print begin\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%3.0f ", data[i * n + j]);
        }
        printf("\n");
    }
    printf("matrix_print end\n");
}