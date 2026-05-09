#include <stdlib.h>
#include <stdio.h>
#include <arm_sve.h>
#include <stdbool.h>

void gemv(int M, int N, const float *A, int lda, const float *x, float *y) {
    int svl = svcntw();    
    int N_aligned_8 = N & ~7;  // 最大的8的倍数
    int N_remainder = N & 7;    // 剩余列数 (0-7)
    
    for (int i = 0; i < M; i += 2 * svl) {
        int rows_remaining1 = M - i;
        int rows1 = (rows_remaining1 < svl) ? rows_remaining1 : svl;
        svbool_t pg1 = svwhilelt_b32(0, rows1);
        
        int rows_remaining2 = M - (i + svl);
        int rows2 = (rows_remaining2 < svl) ? rows_remaining2 : svl;
        svbool_t pg2 = svwhilelt_b32(0, rows2);
        
        svfloat32_t y_acc1 = svdup_f32(0.0f);
        svfloat32_t y_acc2 = svdup_f32(0.0f);
        
        // 8列循环展开（无分支）
        for (int j = 0; j < N_aligned_8; j += 8) {
            // 加载8个x向量值
            svfloat32_t x_vec0 = svdup_f32(x[j]);
            svfloat32_t x_vec1 = svdup_f32(x[j+1]);
            svfloat32_t x_vec2 = svdup_f32(x[j+2]);
            svfloat32_t x_vec3 = svdup_f32(x[j+3]);
            svfloat32_t x_vec4 = svdup_f32(x[j+4]);
            svfloat32_t x_vec5 = svdup_f32(x[j+5]);
            svfloat32_t x_vec6 = svdup_f32(x[j+6]);
            svfloat32_t x_vec7 = svdup_f32(x[j+7]);
            
            // 加载第一组的8列数据
            svfloat32_t a11 = svld1_f32(pg1, &A[j * lda + i]);
            svfloat32_t a12 = svld1_f32(pg1, &A[(j+1) * lda + i]);
            svfloat32_t a13 = svld1_f32(pg1, &A[(j+2) * lda + i]);
            svfloat32_t a14 = svld1_f32(pg1, &A[(j+3) * lda + i]);
            svfloat32_t a15 = svld1_f32(pg1, &A[(j+4) * lda + i]);
            svfloat32_t a16 = svld1_f32(pg1, &A[(j+5) * lda + i]);
            svfloat32_t a17 = svld1_f32(pg1, &A[(j+6) * lda + i]);
            svfloat32_t a18 = svld1_f32(pg1, &A[(j+7) * lda + i]);
            
            // 加载第二组的8列数据
            svfloat32_t a21 = svld1_f32(pg2, &A[j * lda + i + svl]);
            svfloat32_t a22 = svld1_f32(pg2, &A[(j+1) * lda + i + svl]);
            svfloat32_t a23 = svld1_f32(pg2, &A[(j+2) * lda + i + svl]);
            svfloat32_t a24 = svld1_f32(pg2, &A[(j+3) * lda + i + svl]);
            svfloat32_t a25 = svld1_f32(pg2, &A[(j+4) * lda + i + svl]);
            svfloat32_t a26 = svld1_f32(pg2, &A[(j+5) * lda + i + svl]);
            svfloat32_t a27 = svld1_f32(pg2, &A[(j+6) * lda + i + svl]);
            svfloat32_t a28 = svld1_f32(pg2, &A[(j+7) * lda + i + svl]);
            
            // 乘积累加
            y_acc1 = svmla_f32_m(pg1, y_acc1, a11, x_vec0);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a12, x_vec1);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a13, x_vec2);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a14, x_vec3);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a15, x_vec4);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a16, x_vec5);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a17, x_vec6);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a18, x_vec7);
            
            y_acc2 = svmla_f32_m(pg2, y_acc2, a21, x_vec0);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a22, x_vec1);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a23, x_vec2);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a24, x_vec3);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a25, x_vec4);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a26, x_vec5);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a27, x_vec6);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a28, x_vec7);
        }
        
        // 处理剩余列（0-7列）
        for (int r = 0; r < N_remainder; r++) {
            int j = N_aligned_8 + r;
            svfloat32_t x_vec = svdup_f32(x[j]);
            svfloat32_t a1 = svld1_f32(pg1, &A[j * lda + i]);
            svfloat32_t a2 = svld1_f32(pg2, &A[j * lda + i + svl]);
            y_acc1 = svmla_f32_m(pg1, y_acc1, a1, x_vec);
            y_acc2 = svmla_f32_m(pg2, y_acc2, a2, x_vec);
        }
        
        // 写回结果
        if (rows1 > 0) {
            svfloat32_t y1 = svld1_f32(pg1, &y[i]);
            y1 = svadd_f32_m(pg1, y1, y_acc1);
            svst1_f32(pg1, &y[i], y1);
        }
        
        if (rows2 > 0) {
            svfloat32_t y2 = svld1_f32(pg2, &y[i + svl]);
            y2 = svadd_f32_m(pg2, y2, y_acc2);
            svst1_f32(pg2, &y[i + svl], y2);
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
    float *A = aligned_alloc(64, M * N * sizeof(float));
    float *x = aligned_alloc(64, N * sizeof(float));
    float *y = aligned_alloc(64, M * sizeof(float));
    float *y_correct = aligned_alloc(64, M * sizeof(float));
    
    if (!A || !x || !y || !y_correct) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(A); free(x); free(y); free(y_correct);
        return 1;
    }
    
    // 初始化数据
    printf("Initializing data...\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            A[i * M + j] = (float)j;
        }
        x[i] = (float)i;
    }
    for (int i = 0; i < M; i++) {
        y[i] = 0.0f;
        y_correct[i] = 0.0f;
    }
    
    gemv(M, N, A, M, x, y);

    printf("\nFirst 10 results:\n");
    int print_count = (M < 10 ? M : 10);
    for (int i = 0; i < print_count; i++) {
        printf("y[%d]: %12.6f \n", i, y[i]);
    }
    printf("\n");
    
    // 验证结果
    if(check_flag) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                y_correct[i] += A[i + j * M] * x[j];
            }
        }
        printf("\n y_correct First 10 results:\n");
        for (int i = 0; i < print_count; i++) {
            printf("y_correct[%d]: %12.6f \n", i, y_correct[i]);
        }
        bool isPass = true;
        int failCount = 0;
        
        for (int i = 0; i < M; i++) {
            float diff = y_correct[i] - y[i];
            if (diff > 1e-10f || diff < -1e-10f) {
                isPass = false;
                failCount++;
            }
        }
        
        if (isPass) {
            printf("✓ All results are correct within epsilon = 1e-10f\n");
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