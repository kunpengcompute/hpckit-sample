#include <stdlib.h>
#include <stdio.h>
#include <arm_sve.h>
#include <arm_sme.h>
#include <stdbool.h>
#include <stdint.h>

void gemv(int M, int N, const bfloat16_t *A, int lda, const bfloat16_t *x, float *y) {
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
        
        svfloat32_t y_acc1 = svdup_f32(0.0f);
        svfloat32_t y_acc2 = svdup_f32(0.0f);
        
        for (int j = 0; j < N_aligned_8; j += 8) {
            // 加载 x 向量值（BF16），然后转换为 float32
            svbfloat16_t x_bf16_0 = svdup_n_bf16(x[j]);
            svbfloat16_t x_bf16_1 = svdup_n_bf16(x[j+1]);
            svbfloat16_t x_bf16_2 = svdup_n_bf16(x[j+2]);
            svbfloat16_t x_bf16_3 = svdup_n_bf16(x[j+3]);
            svbfloat16_t x_bf16_4 = svdup_n_bf16(x[j+4]);
            svbfloat16_t x_bf16_5 = svdup_n_bf16(x[j+5]);
            svbfloat16_t x_bf16_6 = svdup_n_bf16(x[j+6]);
            svbfloat16_t x_bf16_7 = svdup_n_bf16(x[j+7]);
            
            // BF16 转换为 F32
            svfloat32_t x_vec0 = svcvt_f32_bf16(x_bf16_0);
            svfloat32_t x_vec1 = svcvt_f32_bf16(x_bf16_1);
            svfloat32_t x_vec2 = svcvt_f32_bf16(x_bf16_2);
            svfloat32_t x_vec3 = svcvt_f32_bf16(x_bf16_3);
            svfloat32_t x_vec4 = svcvt_f32_bf16(x_bf16_4);
            svfloat32_t x_vec5 = svcvt_f32_bf16(x_bf16_5);
            svfloat32_t x_vec6 = svcvt_f32_bf16(x_bf16_6);
            svfloat32_t x_vec7 = svcvt_f32_bf16(x_bf16_7);
            
            // 加载 A 的第一组数据（BF16）并转换为 float32
            svbfloat16_t a_bf16_11 = svld1_bf16(pg1, &A[j * lda + i]);
            svbfloat16_t a_bf16_12 = svld1_bf16(pg1, &A[(j+1) * lda + i]);
            svbfloat16_t a_bf16_13 = svld1_bf16(pg1, &A[(j+2) * lda + i]);
            svbfloat16_t a_bf16_14 = svld1_bf16(pg1, &A[(j+3) * lda + i]);
            svbfloat16_t a_bf16_15 = svld1_bf16(pg1, &A[(j+4) * lda + i]);
            svbfloat16_t a_bf16_16 = svld1_bf16(pg1, &A[(j+5) * lda + i]);
            svbfloat16_t a_bf16_17 = svld1_bf16(pg1, &A[(j+6) * lda + i]);
            svbfloat16_t a_bf16_18 = svld1_bf16(pg1, &A[(j+7) * lda + i]);
            
            svfloat32_t a11 = svcvt_f32_bf16(a_bf16_11);
            svfloat32_t a12 = svcvt_f32_bf16(a_bf16_12);
            svfloat32_t a13 = svcvt_f32_bf16(a_bf16_13);
            svfloat32_t a14 = svcvt_f32_bf16(a_bf16_14);
            svfloat32_t a15 = svcvt_f32_bf16(a_bf16_15);
            svfloat32_t a16 = svcvt_f32_bf16(a_bf16_16);
            svfloat32_t a17 = svcvt_f32_bf16(a_bf16_17);
            svfloat32_t a18 = svcvt_f32_bf16(a_bf16_18);
            
            // 加载 A 的第二组数据（BF16）并转换为 float32
            svbfloat16_t a_bf16_21 = svld1_bf16(pg2, &A[j * lda + i + svl]);
            svbfloat16_t a_bf16_22 = svld1_bf16(pg2, &A[(j+1) * lda + i + svl]);
            svbfloat16_t a_bf16_23 = svld1_bf16(pg2, &A[(j+2) * lda + i + svl]);
            svbfloat16_t a_bf16_24 = svld1_bf16(pg2, &A[(j+3) * lda + i + svl]);
            svbfloat16_t a_bf16_25 = svld1_bf16(pg2, &A[(j+4) * lda + i + svl]);
            svbfloat16_t a_bf16_26 = svld1_bf16(pg2, &A[(j+5) * lda + i + svl]);
            svbfloat16_t a_bf16_27 = svld1_bf16(pg2, &A[(j+6) * lda + i + svl]);
            svbfloat16_t a_bf16_28 = svld1_bf16(pg2, &A[(j+7) * lda + i + svl]);
            
            svfloat32_t a21 = svcvt_f32_bf16(a_bf16_21);
            svfloat32_t a22 = svcvt_f32_bf16(a_bf16_22);
            svfloat32_t a23 = svcvt_f32_bf16(a_bf16_23);
            svfloat32_t a24 = svcvt_f32_bf16(a_bf16_24);
            svfloat32_t a25 = svcvt_f32_bf16(a_bf16_25);
            svfloat32_t a26 = svcvt_f32_bf16(a_bf16_26);
            svfloat32_t a27 = svcvt_f32_bf16(a_bf16_27);
            svfloat32_t a28 = svcvt_f32_bf16(a_bf16_28);
            
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
        
        // 处理剩余列
        for (int r = 0; r < N_remainder; r++) {
            int j = N_aligned_8 + r;
            svbfloat16_t x_bf16 = svdup_n_bf16(x[j]);
            svfloat32_t x_vec = svcvt_f32_bf16(x_bf16);
            
            svbfloat16_t a_bf16_1 = svld1_bf16(pg1, &A[j * lda + i]);
            svbfloat16_t a_bf16_2 = svld1_bf16(pg2, &A[j * lda + i + svl]);
            svfloat32_t a1 = svcvt_f32_bf16(a_bf16_1);
            svfloat32_t a2 = svcvt_f32_bf16(a_bf16_2);
            
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

// 辅助函数用于初始化
static inline bfloat16_t float_to_bf16(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    uint16_t bf_bits = (bits >> 16) & 0xFFFF;
    // 四舍五入
    if ((bits & 0x8000) && (bits & 0x7FFF)) {
        bf_bits++;
    }
    bfloat16_t bf;
    memcpy(&bf, &bf_bits, sizeof(bfloat16_t));
    return bf;
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
    
    bfloat16_t *A = aligned_alloc(64, M * N * sizeof(bfloat16_t));
    bfloat16_t *x = aligned_alloc(64, N * sizeof(bfloat16_t));
    float *y = aligned_alloc(64, M * sizeof(float));
    float *y_correct = aligned_alloc(64, M * sizeof(float));
    
    if (!A || !x || !y || !y_correct) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        free(A); free(x); free(y); free(y_correct);
        return 1;
    }
    
    // 初始化 A 和 x
    printf("Initializing data...\n");
    for (int j = 0; j < N; j++) {
        float x_float = (float)(j % 127);
        x[j] = float_to_bf16(x_float);
        for (int i = 0; i < M; i++) {
            float a_float = (float)(i % 127);
            A[j * M + i] = float_to_bf16(a_float);
        }
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
    
    if (check_flag) {
        // 正确结果使用 float 计算
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float a_float, x_float;
                // BF16 转 float
                uint32_t a_bits = (uint32_t)((uint16_t*)&A[i + j * M])[0] << 16;
                uint32_t x_bits = (uint32_t)((uint16_t*)&x[j])[0] << 16;
                memcpy(&a_float, &a_bits, sizeof(float));
                memcpy(&x_float, &x_bits, sizeof(float));
                y_correct[i] += a_float * x_float;
            }
        }
        
        printf("\n y_correct First 10 results:\n");
        for (int i = 0; i < print_count; i++) {
            printf("y_correct[%d]: %12.6f \n", i, y_correct[i]);
        }
        
        bool isPass = true;
        int failCount = 0;
        float epsilon = 1e-2f;
        
        for (int i = 0; i < M; i++) {
            float diff = y_correct[i] - y[i];
            if (diff > epsilon || diff < -epsilon) {
                isPass = false;
                failCount++;
            }
        }
        
        if (isPass) {
            printf("✓ All results are correct within epsilon = %f\n", epsilon);
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