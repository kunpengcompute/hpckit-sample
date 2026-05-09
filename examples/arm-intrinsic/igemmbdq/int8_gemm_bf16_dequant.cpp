#include <random>
#include <memory>
#include <iostream>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <arm_sve.h>
#include <arm_sme.h>
#include <arm_bf16.h>
#include <arm_neon.h>

using MatrixTilingBlock = std::tuple<int64_t, int64_t, int64_t>;

inline bfloat16_t to_bf16(float x)
{
    return vcvth_bf16_f32(x);
}

inline float to_float(bfloat16_t x)
{
    return vcvtah_f32_bf16(x);
}

/** @brief dequant from bf16 to int8 and save to memory */
__attribute__((always_inline)) inline void int8_gemm11_save_bf16_dequant(const float *rscales, const float *cscales, int ldc,
    bfloat16_t *c, int m_num, int n_num) __arm_inout("za") __arm_streaming
{
    svbool_t pg_c_16 = svwhilelt_b16(0, n_num / 4);
    svbool_t pg32_all = svptrue_b32();
    svbfloat16_t zero = svdup_bf16(0);
    for (int i = 0; i < m_num / 4; i++) {
        svint32_t out_v0 = svdup_s32(0);
        out_v0 = svread_hor_za32_s32_m(out_v0, pg32_all, 0, i);
        svfloat32_t rscales_v = svld1(pg32_all, rscales);
        float cscales_v = *(cscales + i);
        svfloat32_t out_v0_f = svcvt_f32_s32_x(pg32_all, out_v0);
        out_v0_f = svmul_x(pg32_all, out_v0_f, svmul_x(pg32_all, rscales_v, cscales_v));
        svst1(pg_c_16, c + (i + 0) * ldc, svuzp1(svcvt_bf16_f32_x(pg32_all, out_v0_f), zero));
    }
}

/** @brief int8 mopa for one za tile */
__attribute__((always_inline)) inline void int8_gemm_1VL_1VL(const int8_t *a, const int8_t *b, int k_num, int m_num,
    int n_num) __arm_inout("za") __arm_streaming
{   
    auto prefetch_dis = 64;
    svbool_t pg_8_m = svwhilelt_b8(0, m_num);
    svbool_t pg_8_n = svwhilelt_b8(0, n_num);
    for (int ki = 0; ki < (k_num / 4); ki++) {
        svprfb(pg_8_n, b + (ki + prefetch_dis) * n_num, SV_PLDL1STRM);
        svint8_t m_data = svld1(pg_8_m, a + ki * m_num);
        svint8_t n_data = svld1(pg_8_n, b + ki * n_num);
        svmopa_za32_s8_m(0, pg_8_m, pg_8_n, m_data, n_data);
    }
}

inline void int8_gemm_bf16_dequant_macro_kernel(int bm, int em, int bn, int en, int k, int k_offset, int k_num, int ldc, int8_t *a,
    int8_t *b, bfloat16_t *c, const float *rscales, const float *cscales) __arm_inout("za") __arm_streaming
{
    for (int mi = bm; mi < em;) {
        int m_num = std::min(64, (em - mi) * 4), ni = bn;
        for (; ni < en; ni += 16) {
            int n_num = std::min(64, (en - ni) * 4);
            svzero_za();
            int8_gemm_1VL_1VL(a + mi * k + k_offset * m_num / 4, b + ni * k + k_offset * n_num / 4, k_num, m_num,
                n_num);
            int8_gemm11_save_bf16_dequant(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num, n_num);
        }
        mi += 16;
    }
}

__arm_new("za") void int8_gemmbf16_dequant_kernel_filter(int m, int n, int ldc, int k, int8_t *a, int8_t *b, bfloat16_t *c,
    const float *rscales, const float *cscales) __arm_streaming
{
    for (int mi = 0; mi < m;) {
        int m_num = std::min(64, (m - mi) * 4), ni = 0;
        for (; ni < n; ni += 16) {
            int n_num = std::min(64, (n - ni) * 4);
            svzero_za();
            int8_gemm_1VL_1VL(a + mi * k, b + ni * k, k, m_num,
                n_num);
            int8_gemm11_save_bf16_dequant(rscales + ni, cscales + mi, ldc, c + mi * ldc + ni, m_num, n_num);
        }
        mi += 16;
    }
}

void reduce(int64_t m, int64_t n, int64_t bk, bfloat16_t *origin_c, bfloat16_t *tmpc, int64_t start, int64_t end)
{
    svbfloat16_t zero = svdup_bf16(0);
    for (int64_t i = start; i < end; i += svcnth()) {
        svbool_t p = svwhilelt_b16(i, end);
        svbfloat16_t s = svld1(p, tmpc + i);
        svfloat32_t s0 = svreinterpret_f32(svzip1(zero, s));
        svfloat32_t s1 = svreinterpret_f32(svzip2(zero, s));
        for (int64_t k = 1; k < bk; ++k) {
            svbfloat16_t v = svld1(p, tmpc + i + k * m * n);
            s0 = svadd_m(svptrue_b32(), s0, svreinterpret_f32(svzip1(zero, v)));
            s1 = svadd_m(svptrue_b32(), s1, svreinterpret_f32(svzip2(zero, v)));
        }
        svst1(p, origin_c + i, svuzp1(svcvt_bf16_x(svptrue_b32(), s0), svcvt_bf16_x(svptrue_b32(), s1)));
    }
}

inline std::tuple<int64_t, int64_t, int64_t> get_idxs(int64_t tid, int64_t bm, int64_t bn, int64_t bk)
{
    int64_t idk = tid % bk;
    tid /= bk;
    int64_t idm = tid % bm;
    tid /= bm;
    int64_t idn = tid % bn;
    tid /= bn;
    return std::tuple(idm, idn, idk);
}

void int8_gemm_bf16_dequant_thread_task(int64_t thread_id, int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr, int8_t *weight_ptr,
    float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc, int64_t num_threads)
{
    int64_t tile_m = std::get<0>(t);
    int64_t tile_n = std::get<1>(t);
    int64_t tile_k = std::get<2>(t);
    int64_t blocks_in_m = m / tile_m;
    int64_t blocks_in_n = (n + tile_n - 1) / tile_n;
    int64_t blocks_in_k = k / tile_k;

    auto [idx_m, idx_n, idx_k] = get_idxs(thread_id, blocks_in_m, blocks_in_n, blocks_in_k);
    int8_t *a = act_ptr + idx_m * tile_m * k + idx_k * tile_m * tile_k;
    int8_t *b = weight_ptr + idx_n * tile_n * k + idx_k * tile_n * tile_k;

    bfloat16_t *c = output_ptr + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
    bfloat16_t *nowc = tmpc + n * m * idx_k + idx_m * tile_m * n + idx_n * tile_n;
    bfloat16_t *target_c = (blocks_in_k == 1) ? c : nowc;
    float *cscales = act_scale_ptr + idx_m * tile_m;
    float *rscales = weight_scale_ptr + idx_n * tile_n;
    bfloat16_t alpha = 1;
    bfloat16_t beta = 0;

    int8_gemmbf16_dequant_kernel_filter(tile_m, std::min(n - idx_n * tile_n, tile_n), n, std::get<2>(t), a, b, target_c, rscales, cscales);

    int64_t reduce_start = (m * n / num_threads * thread_id + 31) / 32 * 32;
    int64_t reduce_end = std::min(m * n, (m * n / num_threads * (thread_id + 1) + 31) / 32 * 32);
    bool k_splited = (blocks_in_k > 1);
    if (k_splited) {
        reduce(m, n, blocks_in_k, output_ptr, tmpc, reduce_start, reduce_end);
    }
}

void int8_gemm_bf16_dequant(int64_t m, int64_t n, int64_t k, MatrixTilingBlock t, int8_t *act_ptr, int8_t *weight_ptr,
    float *act_scale_ptr, float *weight_scale_ptr, bfloat16_t *output_ptr, bfloat16_t *tmpc)
{
    int64_t num_threads = 1;
    
    for(int64_t thread_id = 0;thread_id < num_threads; thread_id++) {
        int8_gemm_bf16_dequant_thread_task(thread_id, m, n, k, t, act_ptr, weight_ptr, act_scale_ptr, 
            weight_scale_ptr, output_ptr, tmpc, num_threads);
    }
}

__arm_new("za") void int8_gemm_pack_kernel(int m, int n, int8_t *src, int lds, int8_t *dst, int ldd) __arm_streaming
{   
    auto prefetch_dis = 64;
    for (int mi = 0; mi < m; mi += 16) {
        int m_num = std::min(16, m - mi);
        int dst_off = mi * ldd;
        svbool_t pg_m = svwhilelt_b32(mi, m);
        for (int ni = 0; ni < n; ni += 64) {
            svzero_za();
            svbool_t pg_n = svwhilelt_b32(0, (n - ni) / 4);
            for (int i = 0; i < 16 && mi + i < m; i++){
                svld1_hor_za32(0, i, pg_n, src + (mi + i) * lds + ni);
            }
            for (int i = 0; i < 16 && ni + i * 4 < n; i++){
                svst1_ver_za32(0, i, pg_m, dst + dst_off + (ni + i * 4) * m_num);
            }
        }
    }
}

template <typename scalar_t, bool with_idx>
void gemm_pack_thread_task(
    int64_t thread_id, int64_t m, int64_t n, int64_t tm, int64_t tn, scalar_t *i_ptr, scalar_t *o_ptr, int *idx, int64_t ldi, int thread_nums)
{
    int64_t blocks_m = m / tm;
    int64_t blocks_n = n / tn;
    int64_t threads_blocks = blocks_m * blocks_n;
    int64_t remain_threads = thread_nums / threads_blocks;

    int64_t subblock_m = (tm + remain_threads * 16 - 1) / (remain_threads * 16) * 16;
    int64_t threads_tm = (tm + subblock_m - 1) / subblock_m;
    int64_t threads_tn = remain_threads / threads_tm * 4 / sizeof(scalar_t);
    int64_t subblock_n = (tn + threads_tn - 1) / threads_tn * 4 / sizeof(scalar_t);
    threads_tn = std::min((tn + subblock_n - 1) / subblock_n, remain_threads / threads_tm);

    int64_t threads_subblocks = threads_tm * threads_tn;
    if (n <= 512){
        subblock_m = tm, subblock_n = tn, threads_tm = threads_tn = threads_subblocks = 1;
    }

    if(thread_id < threads_blocks * threads_subblocks){
        int x = (thread_id / threads_subblocks) / blocks_n;
        int y = (thread_id / threads_subblocks) % blocks_n;
        int subx = (thread_id % threads_subblocks) / threads_tn;
        int suby = (thread_id % threads_subblocks) % threads_tn;
        int bm = std::min(subblock_m, tm - subx * subblock_m);
        int bn = std::min(subblock_n, tn - suby * subblock_n);
        int x_off = x * tm + subx * subblock_m;
        int y_off = y * tn + suby * subblock_n;
        int o_off =
            thread_id / threads_subblocks * tm * tn + subx * subblock_m * tn + suby * subblock_n * std::min(bm, 16);
        int8_gemm_pack_kernel(bm, bn, i_ptr + x_off * ldi + y_off, ldi, o_ptr + o_off, tn);
    }
}

template <typename scalar_t, bool with_idx>
void gemm_pack(int64_t m, int64_t n, int64_t tm, int64_t tn, scalar_t *i_ptr, scalar_t *o_ptr, int *idx, int64_t ldi)
{
    int64_t num_threads = 1;
    for(int64_t thread_id = 0;thread_id < num_threads; thread_id++) {
        gemm_pack_thread_task<scalar_t, with_idx>(thread_id, m, n, tm, tn, i_ptr, o_ptr, idx, ldi, num_threads);
    }
}

void int8_gemm_pack(int64_t m, int64_t n, int64_t tm, int64_t tn, int8_t *i_ptr, int8_t *o_ptr)
{
    gemm_pack<int8_t, false>(m, n, tm, tn, i_ptr, o_ptr, nullptr, n);
}

void test_int8_gemm_bf16_dequant(std::vector<int64_t> &cas)
{
    int64_t m = cas[0];
    int64_t n = cas[1];
    int64_t k = cas[2];
    
    if (k % 4) {
        std::cerr << "K must be a multiple of 4." << std::endl;
        exit(-1);
    }

    MatrixTilingBlock t(m, n, k);

    std::unique_ptr<int8_t[]> a(new int8_t[m * k]);
    std::unique_ptr<int8_t[]> pack_a(new int8_t[m * k]);
    std::unique_ptr<int8_t[]> b(new int8_t[k * n]);
    std::unique_ptr<int8_t[]> pack_b(new int8_t[k * n]);
    std::unique_ptr<float[]> quant[2];
    quant[0].reset(new float[n]);
    quant[1].reset(new float[m]);
    float *quantArgs[2] = {quant[0].get(), quant[1].get()};
    std::unique_ptr<bfloat16_t[]> c(new bfloat16_t[m * n]);
    std::unique_ptr<bfloat16_t[]> tmpc(new bfloat16_t[m * n * 32]);
    std::unique_ptr<bfloat16_t[]> expect(new bfloat16_t[m * n]);

    std::mt19937 rnd(time(0));
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < k; j++) {
            a[i * k + j] = rnd() % 8;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < k; j++) {
            b[i * k + j] = rnd() % 8;
        }
    }
    for (int64_t i = 0; i < n; i++) {
        quant[0][i] = (rnd() % 2 + 1) * 1.0 / k;
    }
    for (int64_t i = 0; i < m; i++) {
        quant[1][i] = (rnd() % 2 + 1) * 1.0 / 2;
    }
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            c[i * n + j] = expect[i * n + j] = rnd() % 128;
        }
    }
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            int32_t temp = 0;
            for (int64_t ki = 0; ki < k; ki++) {
                temp += (int32_t)a[i * k + ki] * b[j * k + ki];
            }
            expect[i * n + j] = to_bf16(temp * quantArgs[0][j] * quantArgs[1][i]);
        }
    }

    int8_gemm_pack(m, k, std::get<0>(t), std::get<2>(t), a.get(), pack_a.get());
    int8_gemm_pack(n, k, std::get<1>(t), std::get<2>(t), b.get(), pack_b.get());
    int8_gemm_bf16_dequant(m, n, k, t, pack_a.get(), pack_b.get(), quantArgs[1], quantArgs[0], c.get(), tmpc.get());
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float cans = c[i * n + j];
            float eans = expect[i * n + j];
            if (std::abs(cans - eans) > 0.1) {
                std::cerr << "(" << i << ", " << j << ") is " << cans << ", expect " << eans << ", "
                          << "shape: (" << m << ", " << n << ", " << k << ")"
                          << "\n";
                exit(1);
            }
        }
    }
}

void read_args(std::vector<std::vector<int64_t>> &cases, int64_t args_num, int64_t argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "Unexpected number of args." << std::endl;
        exit(-1);
    } else {
        cases.clear();
        for (int64_t i = 1; i < argc;) {
            std::vector<int64_t> cas;
            for (int64_t j = 0; j < args_num; ++j, ++i) {
                cas.push_back(std::stoi(argv[i]));
            }
            cases.push_back(cas);
        }
    }
}

int main(int argc, char **argv)
{
    std::vector<std::vector<int64_t>> cases;
    read_args(cases, 3, argc, argv);
    for (auto cas : cases) {
        test_int8_gemm_bf16_dequant(cas);
    }
}