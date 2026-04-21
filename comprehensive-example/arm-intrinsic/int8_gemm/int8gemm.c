#include <stdlib.h>
#include <arm_sme.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

static svint8_t load_a_block_4x4(const int8_t *A, int lda, int i, int k, int M, int K) {
    svint8_t col0 = svld1_s8(svptrue_b8(), &A[i + (k+0) * lda]);
    svint8_t col1 = svld1_s8(svptrue_b8(), &A[i + (k+1) * lda]);
    svint8_t col2 = svld1_s8(svptrue_b8(), &A[i + (k+2) * lda]);
    svint8_t col3 = svld1_s8(svptrue_b8(), &A[i + (k+3) * lda]);
    int valid_rows = (M - i) < 4 ? (M - i) : 4;
    svbool_t pg_row = svwhilelt_b8(0, valid_rows);
    col0 = svld1_s8(pg_row, &A[i + (k+0) * lda]);
    col1 = svld1_s8(pg_row, &A[i + (k+1) * lda]);
    col2 = svld1_s8(pg_row, &A[i + (k+2) * lda]);
    col3 = svld1_s8(pg_row, &A[i + (k+3) * lda]);
    if (k+0 >= K) col0 = svdup_n_s8(0);
    if (k+1 >= K) col1 = svdup_n_s8(0);
    if (k+2 >= K) col2 = svdup_n_s8(0);
    if (k+3 >= K) col3 = svdup_n_s8(0);
    svint8_t t0 = svzip1_s8(col0, col2);
    svint8_t t1 = svzip1_s8(col1, col3);
    svint8_t result = svzip1_s8(t0, t1);
    return result;
}

static svint8_t load_b_block_4x4(const int8_t *B, int ldb, int k, int j, int K, int N) {
    svint8_t col0 = svdup_n_s8(0);
    svint8_t col1 = svdup_n_s8(0);
    svint8_t col2 = svdup_n_s8(0);
    svint8_t col3 = svdup_n_s8(0);
    int valid_rows = (K - k) < 4 ? (K - k) : 4;
    svbool_t pg_row = svwhilelt_b8(0, valid_rows);
    if (j+0 < N) col0 = svld1_s8(pg_row, &B[k + (j+0) * ldb]);
    if (j+1 < N) col1 = svld1_s8(pg_row, &B[k + (j+1) * ldb]);
    if (j+2 < N) col2 = svld1_s8(pg_row, &B[k + (j+2) * ldb]);
    if (j+3 < N) col3 = svld1_s8(pg_row, &B[k + (j+3) * ldb]);
    int8_t buf[64] __attribute__((aligned(64))) = {0};
    svst1_s8(pg_row, buf,      col0);
    svst1_s8(pg_row, buf + 4,  col1);
    svst1_s8(pg_row, buf + 8,  col2);
    svst1_s8(pg_row, buf + 12, col3);
    return svld1_s8(svptrue_b8(), buf);
}

__arm_new("za") void sme_gemm_int8(
    int M, int N, int K,
    const int8_t *A, int lda,
    const int8_t *B, int ldb,
    int32_t *C, int ldc
) __arm_streaming {
    const int BLOCK = 4;
    svbool_t pg = svwhilelt_b8(0, 16);

    for (int i = 0; i < M; i += BLOCK) {
        int i_len = (i + BLOCK <= M) ? BLOCK : (M - i);
        for (int j = 0; j < N; j += BLOCK) {
            int j_len = (j + BLOCK <= N) ? BLOCK : (N - j);
            svzero_za();
            for (int k = 0; k < K; k += BLOCK) {
                int k_len = (k + BLOCK <= K) ? BLOCK : (K - k);

                svint8_t va = load_a_block_4x4(A, lda, i, k, M, K);
                svint8_t vb = load_b_block_4x4(B, ldb, k, j, K, N);
                svmopa_za32_s8_m(0, pg, pg, va, vb);
            }

            // 将 ZA 的 4x4 结果累加到 C
            for (int col = 0; col < j_len; ++col) {
                svbool_t pg_i32 = svwhilelt_b32(0, i_len);
                svint32_t za_col = svread_ver_za32_s32_m(svundef_s32(), pg_i32, 0, col);
                int32_t *c_ptr = &C[i + (j + col) * ldc];
                svint32_t c_col = svld1_s32(pg_i32, c_ptr);
                svint32_t res = svadd_s32_m(pg_i32, c_col, za_col);
                svst1_s32(pg_i32, c_ptr, res);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <M> <N> <K> <Check_flag>\n", argv[0]);
        printf("Example: %s 128 128 128 1\n", argv[0]);
        return -1;
    }
    
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int check_flag = atoi(argv[4]);
    
    if (M <= 0 || N <= 0 || K <= 0) {
        printf("Error: M, N, K must be positive integers\n");
        return -1;
    }
    
    int8_t *A = (int8_t*)aligned_alloc(64, M * K * sizeof(int8_t));
    int8_t *B = (int8_t*)aligned_alloc(64, K * N * sizeof(int8_t));
    int32_t *C = (int32_t*)aligned_alloc(64, M * N * sizeof(int32_t));
    int32_t *C_correct = (int32_t*)aligned_alloc(64, M * N * sizeof(int32_t));
    
    if (!A || !B || !C || !C_correct) {
        printf("Memory allocation failed\n");
        free(A); free(B); free(C); free(C_correct);
        return -1;
    }
    
    printf("Initializing matrices...\n");
    for (int k = 0; k < K; k++) {
        for (int i = 0; i < M; i++) {
            A[i + k * M] = (int8_t)((i + k) % 127);
        }
    }

    for (int j = 0; j < N; j++) {
        for (int k = 0; k < K; k++) {
            B[k + j * K] = (int8_t)((j + k) % 127);
        }
    }
    
    for (int i = 0; i < M * N; i++) {
        C[i] = 0;
        C_correct[i] = 0;
    }
    
    sme_gemm_int8(M, N, K, A, M, B, K, C, M);
    
    printf("\nResult matrix C (first %dx%d, column-major):\n", 
           M < 8 ? M : 8, N < 8 ? N : 8);
    for (int i = 0; i < (M < 8 ? M : 8); i++) {
        for (int j = 0; j < (N < 8 ? N : 8); j++) {
            printf("%10d ", C[i + j * M]);
        }
        printf("\n");
    }
    
    if (check_flag) {
        printf("Verifying results...\n");
        bool correct = true;
        int errors = 0;
        int32_t max_error = 0;
        
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int32_t sum = 0;
                for (int k = 0; k < K; k++) {
                    sum += (int32_t)A[i + k * M] * (int32_t)B[k + j * K];
                }
                C_correct[i + j * M] = sum;
                int32_t diff = C[i + j * M] - sum;
                if (diff < 0) diff = -diff;
                if (diff != 0) {
                    errors++;
                    correct = false;
                }
            }
        }
        
        printf("\nCorrect matrix C (first %dx%d):\n", M < 8 ? M : 8, N < 8 ? N : 8);
        for (int i = 0; i < (M < 8 ? M : 8); i++) {
            for (int j = 0; j < (N < 8 ? N : 8); j++) {
                printf("%10d ", C_correct[i + j * M]);
            }
            printf("\n");
        }
        
        if (correct) {
            printf("\n✓ All results are correct!\n");
        } else {
            printf("\n✗ Found %d errors out of %d elements (max error: %d)\n", 
                   errors, M * N, max_error);
        }
    }
    
    free(A);
    free(B);
    free(C);
    free(C_correct);
    
    return 0;
}