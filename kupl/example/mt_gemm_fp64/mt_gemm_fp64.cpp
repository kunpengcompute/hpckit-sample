/*
    依赖 KUPL 特性能力编写 Gemm sample 示例
    编译命令: clang++ mt_gemm_fp64.cpp -o mt_gemm_fp64 -O3 -lkupl
    运行命令: KUPL_EXECUTOR_BACKEND=pthread KUPL_SCHED_POLICY=mq taskset -c 0-3 ./mt_gemm_fp64
*/

#include <stdio.h>
#include <stdlib.h>
#include "kupl.h"
#include "kupl_mma.h"

using namespace kupl::tensor;

const int M = 64;
const int N = 32;
const int K = 1024;
const int MMA_M = 32;
const int MMA_N = 16;
const int MMA_K = 512;

void gemm_kupl_way(double *data_a, double *data_b, double *data_c);
void init_matrix_data(double *data_a, double *data_b, double *data_c);
void check_correct(double *data_c);

int main()
{
    double *data_a = (double *)kupl_malloc(sizeof(double) * M * K);
    double *data_b = (double *)kupl_malloc(sizeof(double) * K * N);
    double *data_c = (double *)kupl_malloc(sizeof(double) * M * N);
    init_matrix_data(data_a, data_b, data_c);

    gemm_kupl_way(data_a, data_b, data_c);

    check_correct(data_c);

    kupl_free(data_c);
    kupl_free(data_b);
    kupl_free(data_a);
    return 0;
}

struct pack_args_t {
    double *data_a;
    double *data_b;
    double *data_c;
};

void pack_matrix_data(void *args)
{
    pack_args_t *pack_args = (pack_args_t *)args;
    double *data_a = pack_args->data_a;
    double *data_b = pack_args->data_b;
    double *data_c = pack_args->data_c;
    double *data = (double *)kupl_malloc(sizeof(double) * M * K);
    for (int a = 0; a < K; a += MMA_K) {
        for (int b = 0; b < M; b += MMA_M) {
            double *data_tmp = data + a * M + b * MMA_K;
            for (int i = 0; i < MMA_K; i++) {
                for (int j = 0; j < MMA_M; j++) {
                    data_tmp[i * MMA_M + j] = data_a[(a + i) * M + (b + j)];
                }
            }
        }
    }
    kupl_memcpy(data_a, data, sizeof(double) * M * K);
    kupl_free(data);

    data = (double *)kupl_malloc(sizeof(double) * K * N);
    for (int a = 0; a < K; a += MMA_K) {
        for (int b = 0; b < N; b += MMA_N) {
            double *data_tmp = data + a * N + b * MMA_K;
            for (int i = 0; i < MMA_K; i++) {
                for (int j = 0; j < MMA_N; j++) {
                    data_tmp[i * MMA_N + j] = data_b[(a + i) * N + (b + j)];
                }
            }
        }
    }
    kupl_memcpy(data_b, data, sizeof(double) * K * N);
    kupl_free(data);

    data = (double *)kupl_malloc(sizeof(double) * M * N);
    for (int a = 0; a < M; a += MMA_M) {
        for (int b = 0; b < N; b += MMA_N) {
            double *data_tmp = data + a * N + b * MMA_M;
            for (int i = 0; i < MMA_M; i++) {
                for (int j = 0; j < MMA_N; j++) {
                    data_tmp[i * MMA_N + j] = data_c[(a + i) * N + (b + j)];
                }
            }
        }
    }
    kupl_memcpy(data_c, data, sizeof(double) * M * N);
    kupl_free(data);
}

struct unpack_args_t {
    double *data_c;
};

void unpack_matrix_data(void *args)
{
    unpack_args_t *unpack_args = (unpack_args_t *)args;
    double *data_c = unpack_args->data_c;
    double *data = (double *)kupl_malloc(sizeof(double) * M * N);
    for (int a = 0; a < M; a += MMA_M) {
        for (int b = 0; b < N; b += MMA_N) {
            double *data_c_tmp = data_c + a * N + b * MMA_M;
            for (int i = 0; i < MMA_M; i++) {
                for (int j = 0; j < MMA_N; j++) {
                    data[(a + i) * N + (b + j)] = data_c_tmp[i * MMA_N + j];
                }
            }
        }
    }
    kupl_memcpy(data_c, data, sizeof(double) * M * N);
    kupl_free(data);
}

struct mma_args_t {
    double *data_a;
    double *data_b;
    double *data_c;
};

void kupl_mma_func(void *args)
{
    mma_args_t *mma_args = (mma_args_t *)args;
    double *data_a = mma_args->data_a;
    double *data_b = mma_args->data_b;
    double *data_c = mma_args->data_c;

    // KUPL Shape 对象：描述计算矩阵的形状
    auto shape_a = make_shape(Int<32>{}, Int<512>{});
    auto shape_b = make_shape(Int<512>{}, Int<16>{});
    auto shape_c = make_shape(Int<32>{}, Int<16>{});

    // KUPL Stride 对象：描述计算矩阵的各维度步长
    auto stride_a = make_stride(Int<1>{}, Int<32>{});
    auto stride_b = make_stride(Int<16>{}, Int<1>{});
    auto stride_c = make_stride(Int<16>{}, Int<1>{});

    // KUPL Layout 对象：描述计算矩阵的排布，包含矩阵形状和各维度步长
    auto layout_a = make_layout(shape_a, stride_a);
    auto layout_b = make_layout(shape_b, stride_b);
    auto layout_c = make_layout(shape_c, stride_c);

    // 创建 KUPL mma Ops 方法，用于后续 mma 行为：具体基于底层 Ops 按照 atom_shape 进行各维度计算尺寸扩展
    auto atom_mma_shape = make_shape(Int<1>{}, Int<1>{}, Int<512>{});
    auto tiled_mma = make_tiled_mma(Ops<MMA_32x16x1_F64F64F64>{}, atom_mma_shape);
    // 创建 KUPL store Ops 方法，用于后续 store 行为：具体基于底层 Ops 按照 atom_shape 进行个维度写回尺寸扩展
    auto atom_store_shape = make_shape(Int<1>{}, Int<1>{});
    auto tiled_store = make_tiled_store(Ops<STORE_32x16_F64>{}, atom_store_shape);

    // 创建 KUPL Tensor 对象，用于 mma 和 store 行为
    auto tensor_a = make_tensor(data_a, layout_a);
    auto tensor_b = make_tensor(data_b, layout_b);
    auto tensor_c = make_tensor(data_c, layout_c);
    tensor_tiled_mma(tiled_mma, tensor_c, tensor_a, tensor_b, tensor_c);
    tensor_tiled_store(tiled_store, tensor_c);
}

void gemm_kupl_way(double *data_a, double *data_b, double *data_c)
{
    // 静态图创建
    auto sgraph = kupl_sgraph_create();

    // 依据gemm行为拆分依赖关系，从而构建静态图中的点边关系实现多线程并行加速
    kupl_sgraph_node_h mma_nodes[M / MMA_M * N / MMA_N][K / MMA_K];
    mma_args_t mma_args[M / MMA_M * N / MMA_N][K / MMA_K];

    for (int m = 0; m < M; m += MMA_M) {
        for (int n = 0; n < N; n += MMA_N) {
            int fir_pos = m / MMA_M * N / MMA_N + n / MMA_N;
            for (int k = 0; k < K; k += MMA_K) {
                int sec_pos = k / MMA_K;
                mma_args[fir_pos][sec_pos] = {
                    .data_a = data_a + k * M + m * MMA_K,
                    .data_b = data_b + k * N + n * MMA_K,
                    .data_c = data_c + m * N + n * MMA_M
                };
                kupl_sgraph_node_desc_t node_desc = {
                    .field_mask = 0,
                    .func = kupl_mma_func,
                    .args = &mma_args[fir_pos][sec_pos]
                };
                mma_nodes[fir_pos][sec_pos] = kupl_sgraph_add_node(sgraph, &node_desc);
            }
        }
    }

    pack_args_t pack_args = {
        .data_a = data_a,
        .data_b = data_b,
        .data_c = data_c
    };
    kupl_sgraph_node_desc_t pack_node_desc = {
        .field_mask = 0,
        .func = pack_matrix_data,
        .args = &pack_args
    };
    kupl_sgraph_node_h pack_node = kupl_sgraph_add_node(sgraph, &pack_node_desc);

    unpack_args_t unpack_args = {
        .data_c = data_c
    };
    kupl_sgraph_node_desc_t unpack_node_desc = {
        .field_mask = 0,
        .func = unpack_matrix_data,
        .args = &unpack_args
    };
    kupl_sgraph_node_h unpack_node = kupl_sgraph_add_node(sgraph, &unpack_node_desc);

    for (int m = 0; m < M; m += MMA_M) {
        for (int n = 0; n < N; n += MMA_N) {
            int fir_pos = m / MMA_M * N / MMA_N + n / MMA_N;
            kupl_sgraph_add_dep(pack_node, mma_nodes[fir_pos][0]);
            kupl_sgraph_add_dep(mma_nodes[fir_pos][K / MMA_K - 1], unpack_node);
            for (int k = 0; k < K; k += MMA_K) {
                int sec_pos = k / MMA_K;
                if (sec_pos != 0) {
                    kupl_sgraph_add_dep(mma_nodes[fir_pos][sec_pos - 1], mma_nodes[fir_pos][sec_pos]);
                }
            }
        }
    }

    // 动态图创建，并将上述构建的静态图包装成任务提到到动态图中执行
    auto graph = kupl_graph_create(KUPL_ALL_EXECUTORS);
    kupl_sgraph_task_desc_t sgraph_desc = {
        .field_mask = 0,
        .sgraph = sgraph
    };
    kupl_task_info_t task_info = {
        .type = KUPL_TASK_TYPE_SGRAPH,
        .desc = &sgraph_desc
    };
    kupl_graph_submit(graph, &task_info);
    kupl_graph_wait(graph);
    kupl_graph_destroy(graph);

    kupl_sgraph_destroy(sgraph);
}

void init_matrix_data(double *data_a, double *data_b, double *data_c)
{
    for (int i = 0; i < K; i++) {                    // 列主序
        for (int j = 0; j < M; j++) {
            if (i == j) {
                data_a[i * M + j] = 1.0;
            } else {
                data_a[i * M + j] = 0.0;
            }
        }
    }
    for (int i = 0; i < K; i++) {                   // 行主序
        for (int j = 0;  j < N; j++) {
            data_b[i * N + j] = 1.0 * (i * N + j);
        }
    }
    for (int i = 0; i < M; i++) {                   // 行主序
        for (int j = 0; j < N; j++) {
            data_c[i * N + j] = 0.0;
        }
    }
}

void check_correct(double *data_c)
{
    int diff = 0;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (data_c[i * N + j] != i * N + j) {
                printf("pos[%d---%d] is wrong --- data_c = %lf\n", i, j, data_c[i * N + j]);
                diff++;
            }
            if (diff >= 10) {
                break;
            }
        }
        if (diff >= 10) {
            break;
        }
    }
    if (diff == 0) {
        printf("all correct\n");
    } else {
        printf("failed\n");
    }
}