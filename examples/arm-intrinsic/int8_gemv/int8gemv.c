#include <stdlib.h>
#include <stdio.h>
#include <arm_sve.h>
#include <stdbool.h>
#include <stdint.h>

void gemv(int M, int N, const int8_t *A, int lda, const int8_t *x, int32_t *y) {
    int svl = svcntw();    
    int N_aligned_8 = N & ~7;
    int N_remainder = N & 7;
    
    for (int i = 0; i < M; i += 2 * svl) {
        int rows_remaining1 = M - i;
        int rows1 = (rows_remaining1 < svl) ? rows_remaining1 : svl;
        svbool_t pg1 = svwhilelt_b32(0, rows1);
        
        int rows_remaining2 = M - (i + svl);
        int rows2 = (rows_remaining2 < svl) ? rows_remaining2 : svl;
        svbool_t pg2 = svwhilelt_b32(0, rows2);
        
        svint32_t y_acc1 = svdup_s32(0);
        svint32_t y_acc2 = svdup_s32(0);
        
        for (int j = 0; j < N_aligned_8; j += 8) {
            // 使用 svld1sb_s32 直接加载 int8 并符号扩展到 int32
            svint32_t x_vec0 = svdup_s32(x[j]);
            svint32_t x_vec1 = svdup_s32(x[j+1]);
            svint32_t x_vec2 = svdup_s32(x[j+2]);
            svint32_t x_vec3 = svdup_s32(x[j+3]);
            svint32_t x_vec4 = svdup_s32(x[j+4]);
            svint32_t x_vec5 = svdup_s32(x[j+5]);
            svint32_t x_vec6 = svdup_s32(x[j+6]);
            svint32_t x_vec7 = svdup_s32(x[j+7]);
            
            // 加载 int8 数据并扩展为 int32
            svint32_t a11 = svld1sb_s32(pg1, &A[j * lda + i]);
            svint32_t a12 = svld1sb_s32(pg1, &A[(j+1) * lda + i]);
            svint32_t a13 = svld1sb_s32(pg1, &A[(j+2) * lda + i]);
            svint32_t a14 = svld1sb_s32(pg1, &A[(j+3) * lda + i]);
            svint32_t a15 = svld1sb_s32(pg1, &A[(j+4) * lda + i]);
            svint32_t a16 = svld1sb_s32(pg1, &A[(j+5) * lda + i]);
            svint32_t a17 = svld1sb_s32(pg1, &A[(j+6) * lda + i]);
            svint32_t a18 = svld1sb_s32(pg1, &A[(j+7) * lda + i]);
            
            svint32_t a21 = svld1sb_s32(pg2, &A[j * lda + i + svl]);
            svint32_t a22 = svld1sb_s32(pg2, &A[(j+1) * lda + i + svl]);
            svint32_t a23 = svld1sb_s32(pg2, &A[(j+2) * lda + i + svl]);
            svint32_t a24 = svld1sb_s32(pg2, &A[(j+3) * lda + i + svl]);
            svint32_t a25 = svld1sb_s32(pg2, &A[(j+4) * lda + i + svl]);
            svint32_t a26 = svld1sb_s32(pg2, &A[(j+5) * lda + i + svl]);
            svint32_t a27 = svld1sb_s32(pg2, &A[(j+6) * lda + i + svl]);
            svint32_t a28 = svld1sb_s32(pg2, &A[(j+7) * lda + i + svl]);
            
            // 乘积累加
            y_acc1 = svmla_s32_m(pg1, y_acc1, a11, x_vec0);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a12, x_vec1);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a13, x_vec2);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a14, x_vec3);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a15, x_vec4);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a16, x_vec5);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a17, x_vec6);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a18, x_vec7);
            
            y_acc2 = svmla_s32_m(pg2, y_acc2, a21, x_vec0);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a22, x_vec1);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a23, x_vec2);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a24, x_vec3);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a25, x_vec4);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a26, x_vec5);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a27, x_vec6);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a28, x_vec7);
        }
        
        // 处理剩余列
        for (int r = 0; r < N_remainder; r++) {
            int j = N_aligned_8 + r;
            svint32_t x_vec = svdup_s32(x[j]);
            svint32_t a1 = svld1sb_s32(pg1, &A[j * lda + i]);
            svint32_t a2 = svld1sb_s32(pg2, &A[j * lda + i + svl]);
            y_acc1 = svmla_s32_m(pg1, y_acc1, a1, x_vec);
            y_acc2 = svmla_s32_m(pg2, y_acc2, a2, x_vec);
        }
        
        // 写回结果
        if (rows1 > 0) {
            svint32_t y1 = svld1_s32(pg1, &y[i]);
            y1 = svadd_s32_m(pg1, y1, y_acc1);
            svst1_s32(pg1, &y[i], y1);
        }
        
        if (rows2 > 0) {
            svint32_t y2 = svld1_s32(pg2, &y[i + svl]);
            y2 = svadd_s32_m(pg2, y2, y_acc2);
            svst1_s32(pg2, &y[i + svl], y2);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <M> <N> <check_flag>\n", argv[0]);
        fprintf(stderr, "Example: %s 1280 1280 1\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int check_flag = atoi(argv[3]);
    if (M <= 0 || N <= 0) {
        fprintf(stderr, "Error: M and N must be positive integers\n");
        return 1;
    }
    
    int8_t *A = aligned_alloc(64, M * N * sizeof(int8_t));
    int8_t *x = aligned_alloc(64, N * sizeof(int8_t));
    int32_t *y = aligned_alloc(64, M * sizeof(int32_t));
    int32_t *y_correct = aligned_alloc(64, M * sizeof(int32_t));
    
    if (!A || !x || !y || !y_correct) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(A); free(x); free(y); free(y_correct);
        return 1;
    }
    // 初始化 A 和 x
    for (int j = 0; j < N; j++) {
        x[j] = (int8_t)(j % 127);
        for (int i = 0; i < M; i++) {
            A[j * M + i] = (int8_t)(i % 127);
        }
    }
    for (int i = 0; i < M; i++) {
        y[i] = 0;
        y_correct[i] = 0;
    }
    
    gemv(M, N, A, M, x, y);
    
    printf("\nFirst 10 results:\n");
    int print_count = (M < 10 ? M : 10);
    for (int i = 0; i < print_count; i++) {
        printf("y[%d]: %12d \n", i, y[i]);
    }
    printf("\n");
    
    if(check_flag) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                y_correct[i] += (int32_t)A[i + j * M] * (int32_t)x[j];
            }
        }
        printf("\n y_correct First 10 results:\n");
        for (int i = 0; i < print_count; i++) {
            printf("y_correct[%d]: %12d \n", i, y_correct[i]);
        }
        
        bool isPass = true;
        int failCount = 0;
        for (int i = 0; i < M; i++) {
            if (y_correct[i] != y[i]) {
                isPass = false;
                failCount++;
            }
        }
        
        if (isPass) {
            printf("✓ All results are correct\n");
        } else {
            printf("✗ Check Fail, total errors: %d\n", failCount);
        }
    }
    
    free(A);
    free(x);
    free(y);
    free(y_correct);
    
    return 0;
}