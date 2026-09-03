#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <math.h>
#include <string.h>
#if defined(USE_MEM_ON_CHIP) || defined(USE_MEM_ON_CHIP_HUGETLB)
#include <memkind.h>
#endif
#include "omp.h"
#include <kblas.h>
#define HUGETBL_ALIGNED_SIZE (1 << 21)
#define ALIGNED_SIZE 4096

#ifndef MIN
#define MIN(num1, num2) (((num1) > (num2)) ? (num2) : (num1))
#endif

#ifndef MAX
#define MAX(num1, num2) (((num1) < (num2)) ? (num2) : (num1))
#endif
#define GEMM_P 128
#define GEMM_Q 384
#define GEMM_R 4096

#ifndef MAX_CPU_CORES
#define MAX_CPU_CORES 608
#endif
#define GEMM_UNROLL_N 8

#define BUFFER_NUMBERS 38
#define BUFFER_SIZE (36 << 20)
#define GEMM_PREFERRED_SIZE 1
#define COMPSIZE 2
double *buffers[BUFFER_NUMBERS];
double *c_buffer_tmp[BUFFER_NUMBERS];

int krestrain_malloc(void **ptr, size_t size)
{
#if defined(USE_MEM_ON_CHIP)
    return memkind_posix_memalign(MEMKIND_HBW, ptr, ALIGNED_SIZE, size);
#elif defined(USE_MEM_ON_CHIP_HUGETLB)
    return memkind_posix_memalign(MEMKIND_HBW_HUGETLB, ptr, HUGETBL_ALIGNED_SIZE, size);
#else
    return posix_memalign(ptr, ALIGNED_SIZE, size);
#endif
}

void krestrain_free(void *ptr)
{
    if (ptr == NULL) {
        return;
    }
#if defined(USE_MEM_ON_CHIP)
    memkind_free(MEMKIND_HBW, ptr);
#elif defined(USE_MEM_ON_CHIP_HUGETLB)
    memkind_free(MEMKIND_HBW_HUGETLB, ptr);
#else
    free(ptr);
#endif
}
static int GetMThreadsRegions(int m_, int *rangeM, const int nthreadsM)
{
    int m = m_;
    int width;
    int i;
    int numParts = 0;
    while (m > 0) {
        width = (m + nthreadsM - numParts - 1) / (nthreadsM - numParts);
        width = (width + GEMM_PREFERRED_SIZE - 1) / GEMM_PREFERRED_SIZE;
        width = width * GEMM_PREFERRED_SIZE;
        m -= width;

        if (m < 0) {
            width = width + m;
        }

        rangeM[numParts + 1] = rangeM[numParts] + width;

        numParts++;
    }
    for (i = numParts; i < MAX_CPU_CORES; i++) {
        rangeM[i + 1] = rangeM[numParts];
    }
    return numParts;
}

void zgemm_restrain(const int M, const int N, const int K, const void *alpha, const void *A, const int lda,
    const void *B, const int ldb, const void *beta, void *R, int threadNum)
{
    int rangeM[MAX_CPU_CORES + 1] = {0};

    int newThreadNum = GetMThreadsRegions(M, rangeM, threadNum);
    double *a = (double *)A;
    double *b = (double *)B;
    double *r = (double *)R;

#pragma omp parallel num_threads(newThreadNum)
    {
        int mypos = omp_get_thread_num();
        int mFrom = rangeM[mypos];
        int mTo = rangeM[mypos + 1];
        int minI = 0;
        int minJ = 0;
        int minL = 0;
        double *c = c_buffer_tmp[mypos];
       
        double *buffer = buffers[mypos];
        double *bufaa = buffer;
        double *bufbb = buffer + GEMM_P * GEMM_Q * 2;
        int gemmR = GEMM_R;
        int gemmQ = GEMM_Q;
        int gemmP = GEMM_P;
        int ls, is, js;
        int ldc = 0;
        for (ls = 0; ls < K; ls += minL) {
            minL = K - ls;
            if (minL >= gemmQ) {
                minL = gemmQ;
            }
            for (is = mFrom; is < mTo; is += minI) {
                minI = mTo - is;
                if (minI >= gemmP) {
                    minI = gemmP;
                }
                for (int js = 0; js < N; js += minJ) {
                    minJ = N - js;
                    if (minJ > gemmR) {
                        minJ = gemmR;
                    }
                    double *bufTmp = bufbb;
                    cblas_zgemm_pack(CblasColMajor,
                        CblasB,
                        CblasNoTrans,
                        minI,
                        minJ,
                        minL,
                        b + (ls + ldb * js) * COMPSIZE,
                        ldb,
                        bufTmp);
                    int minii = GEMM_P;
                    for (int iis = is; iis < is + minI; iis += minii) {
                        minii = MIN(is + minI - iis, GEMM_P);
                        double *tmpAbuf = bufaa;
                        cblas_zgemm_pack(CblasColMajor,
                            CblasA,
                            CblasNoTrans,
                            minii,
                            minJ,
                            minL,
                            a + (iis + lda * ls) * COMPSIZE,
                            lda,
                            tmpAbuf);
                        int minjj = gemmR;
                        int start = js;
                        int end = js + minJ;
                        for (int jjs = js + mypos * GEMM_UNROLL_N; jjs < end; jjs += minjj) {
                            minjj = MIN(end - jjs, GEMM_UNROLL_N);
                            int off = (jjs - start) * minL * COMPSIZE;
                            double *tmpBbuf = bufTmp + off;
                            ldc = minii;
                            memset(c, 0, (GEMM_P) * (GEMM_UNROLL_N) * sizeof(double) * COMPSIZE);
                            cblas_zgemm_compute(CblasColMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                minii,
                                minjj,
                                minL,
                                alpha,
                                tmpAbuf,
                                lda,
                                tmpBbuf,
                                ldb,
                                beta,
                                c,
                                ldc);
                            for (int mcount = iis; mcount < iis + minii; mcount++) {
                                for (int ncount = 0; ncount < minjj; ncount++) {
                                    int index = ((ncount) * (minii) + (mcount - iis)) * 2;
                                    double real = c[index];
                                    double imag = c[index + 1];
                                    r[mcount] += (real * real - imag * imag) * 1.0;
                                }
                            }
                        }
                        start = js;
                        end = MIN(js + mypos * GEMM_UNROLL_N, js + minJ);
                        for (int jjs = start; jjs < end; jjs += minjj) {
                            minjj = MIN(end - jjs, GEMM_UNROLL_N);
                            int off = (jjs - start) * minL * COMPSIZE;
                            double *tmpBbuf = bufTmp + off;
                            ldc = minii;
                            memset(c, 0, (GEMM_P) * (GEMM_UNROLL_N) * sizeof(double) * COMPSIZE);
                            cblas_zgemm_compute(CblasColMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                minii,
                                minjj,
                                minL,
                                alpha,
                                tmpAbuf,
                                lda,
                                tmpBbuf,
                                ldb,
                                beta,
                                c,
                                ldc);
                            for (int mcount = iis; mcount < iis + minii; mcount++) {
                                for (int ncount = 0; ncount < minjj; ncount++) {
                                    int index = ((ncount) * (minii) + (mcount - iis)) * 2;
                                    double real = c[index];
                                    double imag = c[index + 1];
                                    r[mcount] += (real * real - imag * imag) * 1.0;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
int main(int argc, char *argv[])
{
    int M = 65536, N = 32768, K = 128, check = 0, nthreads = 1;
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    nthreads = atoi(argv[4]);
    check = atoi(argv[5]);
    if (nthreads <= 0 && nthreads > MAX_CPU_CORES) {
        nthreads = 1;
    }
    for (int i = 0; i < BUFFER_NUMBERS; i++) {
        krestrain_malloc((void **)(buffers + i), BUFFER_SIZE);
        krestrain_malloc((void **)(c_buffer_tmp + i), (GEMM_P) * (GEMM_UNROLL_N) * sizeof(double) * COMPSIZE);
    }
    double *A = (double *)malloc(M * K * 2 * sizeof(double));
    double *B = (double *)malloc(K * N * 2 * sizeof(double));
    double *R = (double *)malloc(sizeof(double) * M);
    double *R0 = (double *)malloc(sizeof(double) * M);
    srand(0);
    for (int i = 0; i < M * K * 2;) {
        A[i] = i;
        A[i + 1] = 0;
        i += 2;
    }
    for (int i = 0; i < N * K * 2;) {
        B[i] = 1;
        B[i + 1] = 0;
        i += 2;
    }
    for (int i = 0; i < M; i++) {
        R[i] = 0;
        R0[i] = 0;
    }

    int lda = M, ldb = K;
    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    struct timeval start, end;
    gettimeofday(&start, NULL);
    zgemm_restrain(M, N, K, &alpha, A, lda, B, ldb, &beta, R, nthreads);
    gettimeofday(&end, NULL);
    long seconds = end.tv_sec - start.tv_sec;
    long microseconds = end.tv_usec - start.tv_usec;
    double duration = seconds * 1000.0 + microseconds / 1000.0;

    double Gflops = ((8 * (long long)M * N * K + 4 * (long long)M * N) / duration) / 1e6;
    printf("M: %d, N: %d, K: %d, duration time: %0.6f ms, Gflops: %0.6f\n", M, N, K, duration, Gflops);
    if (check == 1) {
        double *C = (double *)malloc(M * N * 2 * sizeof(double));
        if(C == NULL) {
            printf("Out of memory for C matrix\n");
        }
        for (int i = 0; i < N * M * 2;) {
            C[i] = 0;
            C[i + 1] = 0;
            i += 2;
        }
        printf("Checking...\n");
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &alpha, A, M, B, K, &beta, C, M);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                int index = (i + j * M) * 2;
                double real = C[index];
                double imag = C[index + 1];
                R0[i] += (real * real - imag * imag);
            }
        }
        int check_res = 1;
        for (int i = 0; i < M; i++) {
            if (fabs(R[i] - R0[i]) > 1e-7) {
                printf("check:  R[%d]: %f, R0[%d]: %f \n", i, R[i], i, R0[i]);
                check_res = 0;
                break;
            }
        }
        if (check_res) {
            printf("check success!\n");
        } else {
            printf("check fail!\n");
        }
        free(C);
    }
    free(A);
    free(B);
    free(R);
    for (int i = 0; i < BUFFER_NUMBERS; i++) {
        krestrain_free(buffers[i]);
        krestrain_free(c_buffer_tmp[i]);
    }
}