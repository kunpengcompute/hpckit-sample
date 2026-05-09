#include <stdlib.h>
#include <arm_sme.h>
#include <stdio.h>
#include <stdbool.h>

__arm_new("za") void sme_gemm_fp32(
    const int M,
    const int N,
    const int K,
    const float *restrict A,
    const int lda,
    const float *restrict B,
    const int ldb,
    float *restrict C,
    const int ldc
) __arm_streaming {
    int svl = svcntw();
    svbool_t pg = svptrue_b32();
    
    for (int i = 0; i < M; i += svl) {
        int i_limit = (i + svl <= M) ? svl : (M - i);
        svbool_t pg_i = svwhilelt_b32(0, i_limit);
        
        for (int j = 0; j < N; j += svl) {
            int j_limit = (j + svl <= N) ? svl : (N - j);
            svbool_t pg_j = svwhilelt_b32(0, j_limit);
            svzero_za();
            
            int k = 0;
            float temp_b[svl];
            
            for (; k + 7 < K; k += 8) {
                svfloat32_t vec_a0 = svld1_f32(pg_i, &A[i + (k+0) * lda]);
                svfloat32_t vec_a1 = svld1_f32(pg_i, &A[i + (k+1) * lda]);
                svfloat32_t vec_a2 = svld1_f32(pg_i, &A[i + (k+2) * lda]);
                svfloat32_t vec_a3 = svld1_f32(pg_i, &A[i + (k+3) * lda]);
                svfloat32_t vec_a4 = svld1_f32(pg_i, &A[i + (k+4) * lda]);
                svfloat32_t vec_a5 = svld1_f32(pg_i, &A[i + (k+5) * lda]);
                svfloat32_t vec_a6 = svld1_f32(pg_i, &A[i + (k+6) * lda]);
                svfloat32_t vec_a7 = svld1_f32(pg_i, &A[i + (k+7) * lda]);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+0) + (j+idx) * ldb];
                }
                svfloat32_t vec_b0 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+1) + (j+idx) * ldb];
                }
                svfloat32_t vec_b1 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+2) + (j+idx) * ldb];
                }
                svfloat32_t vec_b2 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+3) + (j+idx) * ldb];
                }
                svfloat32_t vec_b3 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+4) + (j+idx) * ldb];
                }
                svfloat32_t vec_b4 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+5) + (j+idx) * ldb];
                }
                svfloat32_t vec_b5 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+6) + (j+idx) * ldb];
                }
                svfloat32_t vec_b6 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+7) + (j+idx) * ldb];
                }
                svfloat32_t vec_b7 = svld1_f32(pg_j, temp_b);
                
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a0, vec_b0);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a1, vec_b1);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a2, vec_b2);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a3, vec_b3);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a4, vec_b4);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a5, vec_b5);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a6, vec_b6);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a7, vec_b7);
            }
            
            for (; k + 3 < K; k += 4) {
                svfloat32_t vec_a0 = svld1_f32(pg_i, &A[i + (k+0) * lda]);
                svfloat32_t vec_a1 = svld1_f32(pg_i, &A[i + (k+1) * lda]);
                svfloat32_t vec_a2 = svld1_f32(pg_i, &A[i + (k+2) * lda]);
                svfloat32_t vec_a3 = svld1_f32(pg_i, &A[i + (k+3) * lda]);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+0) + (j+idx) * ldb];
                }
                svfloat32_t vec_b0 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+1) + (j+idx) * ldb];
                }
                svfloat32_t vec_b1 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+2) + (j+idx) * ldb];
                }
                svfloat32_t vec_b2 = svld1_f32(pg_j, temp_b);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[(k+3) + (j+idx) * ldb];
                }
                svfloat32_t vec_b3 = svld1_f32(pg_j, temp_b);
                
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a0, vec_b0);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a1, vec_b1);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a2, vec_b2);
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a3, vec_b3);
            }
            
            for (; k < K; k++) {
                svfloat32_t vec_a = svld1_f32(pg_i, &A[i + k * lda]);
                
                for (int idx = 0; idx < j_limit; idx++) {
                    temp_b[idx] = B[k + (j+idx) * ldb];
                }
                svfloat32_t vec_b = svld1_f32(pg_j, temp_b);
                
                svmopa_za32_f32_m(0, pg_i, pg_j, vec_a, vec_b);
            }
            
            for (int col = 0; col < j_limit; col++) {
                svfloat32_t col_data = svread_ver_za32_f32_m(svundef_f32(), pg_i, 0, col);
                svst1_f32(pg_i, &C[i + (j + col) * ldc], col_data);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        printf("Usage: %s <M> <N> <K> <Check_flag>\n", argv[0]);
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
    
    float *A = (float*)aligned_alloc(64, M * K * sizeof(float));
    float *B = (float*)aligned_alloc(64, K * N * sizeof(float));
    float *C = (float*)aligned_alloc(64, M * N * sizeof(float));
    float *C_correct = (float*)aligned_alloc(64, M * N * sizeof(float));
    
    if (!A || !B || !C || !C_correct) {
        printf("Memory allocation failed\n");
        return -1;
    }
    
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A[i + k * M] = (float)(i + 1);
        }
    }
    
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            B[k + j * K] = (float)(j + 1);
        }
    }
    
    for (int i = 0; i < M * N; i++) {
        C[i] = 0.0f;
        C_correct[i] = 0.0f;
    }
    
    sme_gemm_fp32(M, N, K, A, M, B, K, C, M);
    
    printf("\nResult matrix C (first 8x8, column-major):\n");
    for (int i = 0; i < (M < 8 ? M : 8); i++) {
        for (int j = 0; j < (N < 8 ? N : 8); j++) {
            printf("%8.2f ", C[i + j * M]);
        }
        printf("\n");
    }
    
    bool correct = true;
    int errors = 0;
    float max_error = 0.0f;
    
    if (check_flag) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += A[i + k * M] * B[k + j * K];
                }
                C_correct[i + j * M] = sum;
                float diff = C[i + j * M] - sum;
                if (diff < 0) diff = -diff;
                if (diff > 1e-5f) {
                    errors++;
                    correct = false;
                    if (diff > max_error) max_error = diff;
                }
            }
        }
        
        printf("\nResult matrix C_correct (first 8x8, column-major):\n");
        for (int i = 0; i < (M < 8 ? M : 8); i++) {
            for (int j = 0; j < (N < 8 ? N : 8); j++) {
                printf("%8.2f ", C_correct[i + j * M]);
            }
            printf("\n");
        }
        
        if (correct) {
            printf("\n✓ All results are correct! (M=%d, N=%d, K=%d)\n", M, N, K);
        } else {
            printf("\n✗ Found %d errors out of %d elements (max error: %e)\n", errors, M*N, max_error);
        }
    }
    
    free(A);
    free(B);
    free(C);
    free(C_correct);
    
    return 0;
}