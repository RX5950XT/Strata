// engine/src/inkling2/cuda/ink_attn_dispatch.cuh - fork: the attention step of one visit as a free function, so the
// same code runs on the main GPU (runner) or on a second GPU that holds the global layers' KV (ink_remote_attn.cuh).
// The body is the runner's original step 7, with its buffers passed in InkAttnBufs.
#pragma once
#include "ink_cuda.cuh"
#include "ink_attn.cuh"
#include "ink_pf_flash.cuh"
#include "ink_dec_tc.cuh"
#include <cstring>

namespace strata::ink {

constexpr int kInkPfTile = 4096;  // prefill keys per tile (longer ranges use the online-softmax tiles)
constexpr int kInkPfQCh  = 256;   // prefill query rows per chunk

struct InkAttnBufs {
    cudaStream_t st = nullptr;
    cublasHandle_t cublas = nullptr;
    int sm_cc = 0;
    float *q = nullptr, *r = nullptr, *rel_table = nullptr, *sinks = nullptr, *gate = nullptr, *out = nullptr;
    int32_t *pos = nullptr, *slots = nullptr;
    uint8_t *anc = nullptr, *dep = nullptr;
    float *tk = nullptr, *tv = nullptr;         // tree mode: K/V of the tree tokens
    float *pm = nullptr, *pl = nullptr, *po = nullptr;  // split-K partials
    __nv_bfloat16 *pf_q = nullptr, *pf_k = nullptr, *pf_v = nullptr, *pf_p = nullptr;
    float *pf_s = nullptr, *pf_o = nullptr, *pf_ot = nullptr, *pf_m = nullptr;
    float *mq_pm = nullptr, *mq_pl = nullptr, *mq_po = nullptr;  // [tree_cap, 24, kInkMqSplits(, 128)]; null = off
};

// fork: splits for the wide kernel (64; INK_DEC_SPLITS=8..64 overrides for experiments). Measured on the 3060 Ti at
// 128K: 64 splits 25.6 tok/s, 48 24.9, 32 24.6, one full wave (57) 24.1 - more CTAs in flight beat wave fitting.
// The same count for every M keeps tree verify and plain decode on the same key partition.
inline int ink_flash_wide_splits() {
    static const int n = [] {
        const char* e = std::getenv("INK_DEC_SPLITS");
        return e ? std::max(8, std::min(kInkMqSplits, std::atoi(e))) : kInkMqSplits;
    }();
    return n;
}

// fork: decode / tree attention for a global visit, split-K with queries fastest in the grid
inline void ink_flash_wide(const InkAttnBufs& b, const InkKvVisitDesc& kv, int mode, int M, int slot0, int p0_val) {
    const int nsp = ink_flash_wide_splits();
    static const bool dec16_off = std::getenv("INK_DEC16_OFF") != nullptr;   // A/B: the 32-lane kernel
    if (ink_dec_tc_ok(b.sm_cc, kv, M)) {
        ink_dec_tc_launch(b.st, b.q, b.rel_table, b.sinks, kv, slot0, b.pos, mode == 2, b.tk, b.tv, b.anc,
                          b.mq_pm, b.mq_pl, b.mq_po, M, nsp);
    } else if (!dec16_off) {
        k_ink_flash_dec16<<<dim3(M, nsp, kInkKVHeads), 128, 0, b.st>>>(
            b.q, b.rel_table, b.sinks, kv, slot0, b.pos,
            (mode == 2) ? b.tk : nullptr, (mode == 2) ? b.tv : nullptr, (mode == 2) ? b.anc : nullptr,
            p0_val, b.mq_pm, b.mq_pl, b.mq_po, M, nsp);
    } else
    k_ink_flash_gqa_splitk<kInkMqSplits, true><<<dim3(M, nsp, kInkKVHeads), 128, 0, b.st>>>(
        b.q, b.rel_table, b.sinks, kv, nullptr, slot0, b.pos,
        (mode == 2) ? b.tk : nullptr, (mode == 2) ? b.tv : nullptr,
        (mode == 2) ? b.anc : nullptr, (mode == 2) ? b.dep : nullptr,
        p0_val, b.mq_pm, b.mq_pl, b.mq_po, M, nsp);
    k_ink_flash_mq_merge<<<dim3(kInkHeads, M), 128, 0, b.st>>>(b.mq_pm, b.mq_pl, b.mq_po, b.gate, b.out, M, nsp);
}

inline void ink_pf_attn_tiled(const InkAttnBufs& b, const InkKvVisitDesc& kv, int slot0, int p0_val, int M, bool is_swa, int k_start,
                          bool use_fp16_pf, cudaDataType_t pf_dt, cublasGemmAlgo_t pf_algo) {
    constexpr int GQ = kInkHeads / kInkKVHeads;
    const float sc_qk = 0.08838834764831845f;
    const float one = 1.0f, zero = 0.0f;
    if (use_fp16_pf) k_ink_pf_pack_q<true><<<dim3(kInkHeads, M), 128, 0, b.st>>>(b.q, b.pf_q, M);
    else             k_ink_pf_pack_q<false><<<dim3(kInkHeads, M), 128, 0, b.st>>>(b.q, b.pf_q, M);
    float* run_m = b.pf_m;
    float* run_l = b.pf_m + (size_t)kInkHeads * kInkPfQCh;
    float* til_m = b.pf_m + (size_t)2 * kInkHeads * kInkPfQCh;
    float* til_l = b.pf_m + (size_t)3 * kInkHeads * kInkPfQCh;
    for (int q0 = 0; q0 < M; q0 += kInkPfQCh) {
        const int C = std::min(kInkPfQCh, M - q0);
        const int kb = is_swa ? std::max(k_start, p0_val + q0 - kInkSwaWin + 1) : k_start;
        const int ke = p0_val + q0 + C;
        k_ink_pf_online_init<<<dim3(C, kInkHeads), 128, 0, b.st>>>(b.sinks, q0, C, run_m, run_l, b.pf_o);
        for (int t0 = kb; t0 < ke; t0 += kInkPfTile) {
            const int tl = std::min(kInkPfTile, ke - t0);
            if (use_fp16_pf) k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, tl), 128, 0, b.st>>>(kv, slot0, t0, tl, b.pf_k, b.pf_v);
            else             k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, tl), 128, 0, b.st>>>(kv, slot0, t0, tl, b.pf_k, b.pf_v);
            for (int r = 0; r < GQ; ++r) {
                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    b.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    tl, C, kInkHeadDim, &sc_qk,
                    b.pf_k, pf_dt, kInkHeadDim, (long long)tl * kInkHeadDim,
                    b.pf_q + ((size_t)r * M + q0) * kInkHeadDim, pf_dt, kInkHeadDim, (long long)GQ * M * kInkHeadDim,
                    &zero, b.pf_s + (size_t)r * C * tl, CUDA_R_32F, tl, (long long)GQ * C * tl,
                    kInkKVHeads, CUBLAS_COMPUTE_32F, pf_algo));
            }
            if (use_fp16_pf) k_ink_pf_tile_softmax<true><<<dim3(C, kInkHeads), 256, 0, b.st>>>(b.pf_s, b.rel_table, b.pf_p, til_m, til_l, q0, C, t0, tl, p0_val, is_swa);
            else             k_ink_pf_tile_softmax<false><<<dim3(C, kInkHeads), 256, 0, b.st>>>(b.pf_s, b.rel_table, b.pf_p, til_m, til_l, q0, C, t0, tl, p0_val, is_swa);
            for (int r = 0; r < GQ; ++r) {
                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    b.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    kInkHeadDim, C, tl, &one,
                    b.pf_v, pf_dt, kInkHeadDim, (long long)tl * kInkHeadDim,
                    b.pf_p + (size_t)r * C * tl, pf_dt, tl, (long long)GQ * C * tl,
                    &zero, b.pf_ot + (size_t)r * C * kInkHeadDim, CUDA_R_32F, kInkHeadDim, (long long)GQ * C * kInkHeadDim,
                    kInkKVHeads, CUBLAS_COMPUTE_32F, pf_algo));
            }
            k_ink_pf_online_merge<<<dim3(C, kInkHeads), 128, 0, b.st>>>(b.pf_o, b.pf_ot, run_m, run_l, til_m, til_l, C);
        }
        k_ink_pf_online_finish<<<dim3(kInkHeads, C), 128, 0, b.st>>>(b.pf_o, run_l, b.gate, b.out, q0, C);
    }
}

// The older paths (cuBLAS prefill, split-K decode, tree); also the reference for INK_PF_FLASH_CHECK.
inline void ink_attention_cublas(const InkAttnBufs& b, const InkKvVisitDesc& kv, const float* rel_proj,
                                 int mode, int M, int slot0, int p0_val, bool is_swa) {
    dim3 gr(kInkHeads, M);
    k_ink_rel_table<<<gr, 256, 0, b.st>>>(b.r, rel_proj, b.rel_table, M);
    static const bool mq_off = std::getenv("INK_MQ_OFF") != nullptr;   // fork: A/B switch for the wide split-K
    if ((M <= 16 || mode == 2) && !is_swa && mode != 1 && b.mq_po != nullptr && M <= 32 && !mq_off) {
        ink_flash_wide(b, kv, mode, M, slot0, p0_val);
    } else if (M <= 16 || mode == 2) {
        dim3 gsp(kInkKVHeads, kInkAttnSplits, M);
        k_ink_flash_gqa_splitk<<<gsp, 128, 0, b.st>>>(
            b.q, b.rel_table, b.sinks,
            kv, (mode == 1) ? b.slots : nullptr, slot0, b.pos,
            (mode == 2) ? b.tk : nullptr, (mode == 2) ? b.tv : nullptr,
            (mode == 2) ? b.anc : nullptr, (mode == 2) ? b.dep : nullptr,
            p0_val, b.pm, b.pl, b.po, M);
        k_ink_flash_gqa_merge<<<gr, 128, 0, b.st>>>(b.pm, b.pl, b.po, b.gate, b.out, M);
    } else if (mode == 0 && M >= 128 && (b.sm_cc >= 70 || b.sm_cc == 60) && b.pf_q != nullptr) {
        // High-speed Causal/SWA-Sliced Tensor Core Prefill Attention (BF16 on sm_80+, FP16 on sm_60..75!)
        const bool use_fp16_pf = (b.sm_cc < 80);
        const cudaDataType_t pf_dt = use_fp16_pf ? CUDA_R_16F : CUDA_R_16BF;
        const cublasGemmAlgo_t pf_algo = (b.sm_cc >= 70) ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT;
        const int k_start = is_swa ? std::max(0, p0_val - kInkSwaWin + 1) : 0;
        const int K_len = (p0_val + M) - k_start;
        if (K_len > kInkPfTile) {
            ink_pf_attn_tiled(b, kv, slot0, p0_val, M, is_swa, k_start, use_fp16_pf, pf_dt, pf_algo);
        } else {
        if (use_fp16_pf) {
            k_ink_pf_pack_q<true><<<dim3(kInkHeads, M), 128, 0, b.st>>>(b.q, b.pf_q, M);
            k_ink_pf_pack_kv<true><<<dim3(kInkKVHeads, K_len), 128, 0, b.st>>>(kv, slot0, k_start, K_len, b.pf_k, b.pf_v);
        } else {
            k_ink_pf_pack_q<false><<<dim3(kInkHeads, M), 128, 0, b.st>>>(b.q, b.pf_q, M);
            k_ink_pf_pack_kv<false><<<dim3(kInkKVHeads, K_len), 128, 0, b.st>>>(kv, slot0, k_start, K_len, b.pf_k, b.pf_v);
        }
        const int Q_CH = kInkPfQCh;
        constexpr int GQ = kInkHeads / kInkKVHeads;  // 3 query heads per KV head
        const float sc_qk = 0.08838834764831845f;
        const float one = 1.0f, zero = 0.0f;
        for (int q0 = 0; q0 < M; q0 += Q_CH) {
            const int C = std::min(Q_CH, M - q0);
            const int k_sub_beg = is_swa ? std::max(k_start, p0_val + q0 - kInkSwaWin + 1) : k_start;
            const int k_sub_end = p0_val + q0 + C;
            const int k_off = k_sub_beg - k_start;
            const int K_sub = std::max(1, k_sub_end - k_sub_beg);
            // S[h, C, K_sub] = (1/sqrt(128)) * Q[h, q0..q0+C, 128] @ K[h/3, k_off..k_off+K_sub, 128]^T,
            // one batched GEMM per sub-head r (heads 3g+r share KV head g)
            for (int r = 0; r < GQ; ++r) {
                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    b.cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                    K_sub, C, kInkHeadDim, &sc_qk,
                    b.pf_k + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                    b.pf_q + ((size_t)r * M + q0) * kInkHeadDim, pf_dt, kInkHeadDim, (long long)GQ * M * kInkHeadDim,
                    &zero, b.pf_s + (size_t)r * C * K_sub, CUDA_R_32F, K_sub, (long long)GQ * C * K_sub,
                    kInkKVHeads, CUBLAS_COMPUTE_32F, pf_algo));
            }
            if (use_fp16_pf) {
                k_ink_pf_softmax_rel_sink<true><<<dim3(C, kInkHeads), 256, 0, b.st>>>(
                    b.pf_s, b.rel_table, b.sinks, b.pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
            } else {
                k_ink_pf_softmax_rel_sink<false><<<dim3(C, kInkHeads), 256, 0, b.st>>>(
                    b.pf_s, b.rel_table, b.sinks, b.pf_p, q0, C, k_sub_beg, K_sub, p0_val, is_swa);
            }
            // O[h, C, 128] = P[h, C, K_sub] @ V[h, k_off..k_off+K_sub, 128]
            for (int r = 0; r < GQ; ++r) {
                INK_CUBLAS_CK(cublasGemmStridedBatchedEx(
                    b.cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                    kInkHeadDim, C, K_sub, &one,
                    b.pf_v + (size_t)k_off * kInkHeadDim, pf_dt, kInkHeadDim, (long long)K_len * kInkHeadDim,
                    b.pf_p + (size_t)r * C * K_sub, pf_dt, K_sub, (long long)GQ * C * K_sub,
                    &zero, b.pf_o + (size_t)r * C * kInkHeadDim, CUDA_R_32F, kInkHeadDim, (long long)GQ * C * kInkHeadDim,
                    kInkKVHeads, CUBLAS_COMPUTE_32F, pf_algo));
            }
            k_ink_pf_unpack_o_gate<<<dim3(kInkHeads, C), 128, 0, b.st>>>(b.pf_o, b.gate, b.out, q0, C);
        }
        }
    } else {
        dim3 g3(kInkKVHeads, M);
        k_ink_flash_gqa_gqa3<<<g3, 128, 0, b.st>>>(
            b.q, b.rel_table, b.sinks, b.gate,
            kv, (mode == 1) ? b.slots : nullptr, slot0, b.pos,
            b.out, M);
    }
}

__global__ void k_ink_pf_flash_diff(const float* a, const float* b, size_t n, unsigned int* res) {
    float d = 0.0f, m = 0.0f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        d = fmaxf(d, fabsf(a[i] - b[i]));
        m = fmaxf(m, fabsf(b[i]));
    }
    atomicMax(res, __float_as_uint(d));
    atomicMax(res + 1, __float_as_uint(m));
}

inline void ink_pf_flash_compare(cudaStream_t st, const float* got, const float* ref, size_t n, int M, int p0, bool is_swa) {
    static unsigned int* res = nullptr;
    if (!res) cudaMallocManaged(&res, 2 * sizeof(unsigned int));
    cudaStreamSynchronize(st);
    res[0] = res[1] = 0;
    k_ink_pf_flash_diff<<<120, 256, 0, st>>>(got, ref, n, res);
    cudaStreamSynchronize(st);
    float d, m;
    std::memcpy(&d, &res[0], 4);
    std::memcpy(&m, &res[1], 4);
    std::fprintf(stderr, "[pf-flash-check] p0=%d M=%d %s max|diff|=%.3g max|ref|=%.3g\n", p0, M, is_swa ? "swa" : "glob", d, m);
}

// mode: 0 = single-slot sequential (prefill or M=1 decode), 1 = batched multi-slot, 2 = tree verify
inline void ink_attention(const InkAttnBufs& b, const InkKvVisitDesc& kv, const float* rel_proj,
                          int mode, int M, int slot0, int p0_val, bool is_swa) {
    // fork: the fused prefill kernel computes the relative bias itself, so it needs no rel_table
    static const bool fl_check = std::getenv("INK_PF_FLASH_CHECK") != nullptr;  // debug: compare with the old path
    if (mode == 0 && M >= 128 &&
        ink_pf_flash_launch(b.st, b.sm_cc, b.q, b.r, rel_proj, b.sinks, b.gate, b.out, kv, slot0, p0_val, M,
                            is_swa ? std::max(0, p0_val - kInkSwaWin + 1) : 0, is_swa)) {
        if (!fl_check) return;
        static float* ref = nullptr;
        static size_t ref_n = 0;
        const size_t n = (size_t)M * kInkHeads * kInkHeadDim;
        if (n > ref_n) { if (ref) cudaFree(ref); cudaMalloc(&ref, n * sizeof(float)); ref_n = n; }
        InkAttnBufs b2 = b;
        b2.out = ref;
        ink_attention_cublas(b2, kv, rel_proj, 0, M, slot0, p0_val, is_swa);
        ink_pf_flash_compare(b.st, b.out, ref, n, M, p0_val, is_swa);
        return;
    }
    ink_attention_cublas(b, kv, rel_proj, mode, M, slot0, p0_val, is_swa);
}

}  // namespace strata::ink
