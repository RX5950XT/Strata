// engine/src/inkling2/cuda/ink_pf_flash.cuh - fork: fused prefill attention (flash attention 2 style, FP16/BF16 tensor cores,
// sm_80+). One CTA = 64 query positions x the 3 query heads of one KV head, so each K/V tile is read and dequantized once
// for 3 heads. The relative-position bias (rank 16, distances < 1024) is computed in FP32 inside the kernel from r and
// rel_proj, so no [M, 24, 1025] table is written. Semantics match k_ink_pf_tile_softmax + k_ink_pf_online_*:
// score = q.k / sqrt(128) + bias(dist), causal (+ SWA window), the per-query sink adds exp(sink) to the denominator,
// out = O / l * sigmoid(gate).
#pragma once
#include "ink_attn.cuh"

namespace strata::ink {

constexpr int kPfFlBM      = 64;                                  // query positions per CTA
constexpr int kPfFlBN      = 64;                                  // keys per tile
constexpr int kPfFlGQ      = kInkHeads / kInkKVHeads;             // 3 query heads per KV head
constexpr int kPfFlWarps   = kPfFlGQ * (kPfFlBM / 16);            // 12: warp w = head w/4, rows (w%4)*16..+16
constexpr int kPfFlThreads = kPfFlWarps * 32;                     // 384
constexpr int kPfFlRowB    = kInkHeadDim * 2;                     // 256 bytes per BF16 row
constexpr int kPfFlQBytes  = kPfFlGQ * kPfFlBM * kPfFlRowB;       // 48 KB
constexpr int kPfFlKBytes  = kPfFlBN * kPfFlRowB;                 // 16 KB (K and V each)
constexpr int kPfFlPCols   = 128;                                 // distance slice per tile (127 used)
constexpr int kPfFlPBytes  = kInkDRel * kPfFlPCols * 4;           // 8 KB
constexpr int kPfFlSmem    = kPfFlQBytes + 2 * kPfFlKBytes + kPfFlPBytes;  // 88 KB

// byte offset of 16-byte chunk c (0..15) of row r in a swizzled [rows][128] BF16 tile (conflict-free ldmatrix)
__device__ __forceinline__ uint32_t pffl_off(int r, int c) { return (uint32_t)(r * kPfFlRowB + ((c ^ (r & 7)) << 4)); }

__device__ __forceinline__ void pffl_ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void pffl_ldsm_x4_t(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
// F16 = FP16 inputs (10-bit mantissa, values clamped to +-65504), else BF16; FP32 accumulation either way
template <bool F16>
__device__ __forceinline__ void pffl_mma(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
    if constexpr (F16)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                     "{%0,%1,%2,%3};\n"
                     : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
template <bool F16>
__device__ __forceinline__ uint32_t pffl_pack(float lo, float hi) {
    if constexpr (F16) {
        const __half2 v = __floats2half2_rn(fminf(fmaxf(lo, -65504.0f), 65504.0f), fminf(fmaxf(hi, -65504.0f), 65504.0f));
        return *reinterpret_cast<const uint32_t*>(&v);
    } else {
        const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<const uint32_t*>(&v);
    }
}

// One K/V tile staged in registers: FP8 rows are 8 chunks of 16 B, FP16 rows 16 chunks.
template <bool FP8>
struct PfFlStage {
    static constexpr int kChRow = FP8 ? 8 : 16;
    static constexpr int kTotal = 2 * kPfFlBN * kChRow;                       // K then V
    static constexpr int kPer   = (kTotal + kPfFlThreads - 1) / kPfFlThreads;
    uint4 raw[kPer];
    float sc[kPer];
};

template <bool FP8>
__device__ __forceinline__ void pffl_fetch(PfFlStage<FP8>& s, const InkKvVisitDesc& kv, int slot0, int kh, int t0, int ke) {
    using S = PfFlStage<FP8>;
#pragma unroll
    for (int i = 0; i < S::kPer; ++i) {
        const int c = threadIdx.x + i * kPfFlThreads;
        s.raw[i] = make_uint4(0, 0, 0, 0);
        s.sc[i] = 0.0f;
        if (c >= S::kTotal) continue;
        const bool isv = c >= S::kTotal / 2;
        const int cc = isv ? c - S::kTotal / 2 : c;
        const int j = cc / S::kChRow, part = cc % S::kChRow;
        const int kp = t0 + j;
        if (kp >= ke) continue;
        const int pi = kv.is_swa ? (kp % kv.cap_pos) : kp;
        const size_t rowi = ((size_t)slot0 * kv.cap_pos + pi) * kInkKVHeads + kh;
        if constexpr (FP8) {
            const uint8_t* base = (isv ? kv.v_f8 : kv.k_f8) + rowi * kInkHeadDim;
            s.raw[i] = __ldg(reinterpret_cast<const uint4*>(base) + part);
            s.sc[i] = __ldg((isv ? kv.v_sc : kv.k_sc) + rowi);
        } else {
            const __half* base = (isv ? kv.v_f16 : kv.k_f16) + rowi * kInkHeadDim;
            s.raw[i] = __ldg(reinterpret_cast<const uint4*>(base) + part);
        }
    }
}

template <bool FP8, bool F16>
__device__ __forceinline__ void pffl_store(const PfFlStage<FP8>& s, unsigned char* ks, unsigned char* vs) {
    using S = PfFlStage<FP8>;
#pragma unroll
    for (int i = 0; i < S::kPer; ++i) {
        const int c = threadIdx.x + i * kPfFlThreads;
        if (c >= S::kTotal) continue;
        const bool isv = c >= S::kTotal / 2;
        const int cc = isv ? c - S::kTotal / 2 : c;
        const int j = cc / S::kChRow, part = cc % S::kChRow;
        unsigned char* dst = isv ? vs : ks;
        if constexpr (FP8) {
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&s.raw[i]);
            uint32_t w[8];
#pragma unroll
            for (int e = 0; e < 8; ++e)
                w[e] = pffl_pack<F16>(e4m3_to_f32(b[2 * e]) * s.sc[i], e4m3_to_f32(b[2 * e + 1]) * s.sc[i]);
            *reinterpret_cast<uint4*>(dst + pffl_off(j, 2 * part))     = make_uint4(w[0], w[1], w[2], w[3]);
            *reinterpret_cast<uint4*>(dst + pffl_off(j, 2 * part + 1)) = make_uint4(w[4], w[5], w[6], w[7]);
        } else {
            const __half2* h = reinterpret_cast<const __half2*>(&s.raw[i]);
            uint32_t w[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 f = __half22float2(h[e]);
                w[e] = pffl_pack<F16>(f.x, f.y);
            }
            *reinterpret_cast<uint4*>(dst + pffl_off(j, part)) = make_uint4(w[0], w[1], w[2], w[3]);
        }
    }
}

// grid (ceil(M / 64), 8 KV heads), 384 threads, kPfFlSmem dynamic shared memory.
// q/r/sinks/gate/out are indexed by the chunk row m (0..M-1); row m is at position p0 + m.
template <bool FP8, bool F16>
__global__ void __launch_bounds__(kPfFlThreads, 1) k_ink_pf_flash(
    const float* __restrict__ q,         // [M, 24, 128]
    const float* __restrict__ r,         // [M, 24, 16]
    const float* __restrict__ rel_proj,  // [16, 1024]
    const float* __restrict__ sinks,     // [M, 24]
    const float* __restrict__ gate,      // [M, 24, 128] or null
    float* __restrict__ out,             // [M, 24, 128]
    InkKvVisitDesc kv, int slot0, int p0, int M, int k_start, int is_swa) {
    extern __shared__ __align__(128) unsigned char pffl_smem[];
    unsigned char* qs = pffl_smem;
    unsigned char* ks = qs + kPfFlQBytes;
    unsigned char* vs = ks + kPfFlKBytes;
    float* ps = reinterpret_cast<float*>(vs + kPfFlKBytes);     // [16][128] rel_proj slice * log2(e)

    constexpr float kLog2e = 1.4426950408889634f;
    const float sc_log2 = 0.08838834764831845f * kLog2e;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int hr = warp >> 2, rb = warp & 3;
    const int kh = blockIdx.y, h = kh * kPfFlGQ + hr;
    const int m0 = (gridDim.x - 1 - blockIdx.x) * kPfFlBM;      // longest rows first
    const int nrows = min(kPfFlBM, M - m0);
    const int qlo = p0 + m0, qhi = p0 + m0 + nrows - 1;
    const int kb = is_swa ? max(k_start, qlo - kInkSwaWin + 1) : k_start;
    const int ke = qhi + 1;

    PfFlStage<FP8> stg;
    pffl_fetch<FP8>(stg, kv, slot0, kh, kb, ke);

    // Q -> BF16 smem [3 heads][64 rows][128]
    for (int c = tid; c < kPfFlGQ * kPfFlBM * 16; c += kPfFlThreads) {
        const int row = c >> 4, ch = c & 15;
        const int qh = row / kPfFlBM, qi = row % kPfFlBM;
        uint4 w = make_uint4(0, 0, 0, 0);
        if (qi < nrows) {
            const float4* src = reinterpret_cast<const float4*>(
                q + ((size_t)(m0 + qi) * kInkHeads + kh * kPfFlGQ + qh) * kInkHeadDim + ch * 8);
            const float4 a = __ldg(src), b = __ldg(src + 1);
            w = make_uint4(pffl_pack<F16>(a.x, a.y), pffl_pack<F16>(a.z, a.w), pffl_pack<F16>(b.x, b.y), pffl_pack<F16>(b.z, b.w));
        }
        *reinterpret_cast<uint4*>(qs + pffl_off(row, ch)) = w;
    }

    // this thread's two rows: warp-local g and g + 8
    const int pi0 = rb * 16 + g, pi1 = pi0 + 8;
    const int qp0 = p0 + m0 + pi0, qp1 = p0 + m0 + pi1;
    const int mr0 = min(m0 + pi0, M - 1), mr1 = min(m0 + pi1, M - 1);    // clamped for loads
    const float* r0p = r + ((size_t)mr0 * kInkHeads + h) * kInkDRel;
    const float* r1p = r + ((size_t)mr1 * kInkHeads + h) * kInkDRel;
    float row_m[2] = {sinks[(size_t)mr0 * kInkHeads + h] * kLog2e, sinks[(size_t)mr1 * kInkHeads + h] * kLog2e};
    float row_l[2] = {t == 0 ? 1.0f : 0.0f, t == 0 ? 1.0f : 0.0f};     // the sink counted once per quad
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

    const uint32_t qs_a = (uint32_t)__cvta_generic_to_shared(qs);
    const uint32_t ks_a = (uint32_t)__cvta_generic_to_shared(ks);
    const uint32_t vs_a = (uint32_t)__cvta_generic_to_shared(vs);
    const int qrow = hr * kPfFlBM + rb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;

    for (int t0 = kb; t0 < ke; t0 += kPfFlBN) {
        const int kmax = min(t0 + kPfFlBN, ke) - 1;
        const bool need_bias = kmax > qlo - kInkRelExtent;         // some distance < 1024
        const bool need_mask = kmax > qlo || t0 + kPfFlBN > ke || (is_swa && qhi - t0 >= kInkSwaWin);
        const int dbase = qlo - (t0 + kPfFlBN - 1);                // distance of (row 0, key 63)
        __syncthreads();                                           // previous tile fully consumed
        pffl_store<FP8, F16>(stg, ks, vs);
        if (need_bias) {
            for (int c = tid; c < kInkDRel * kPfFlPCols; c += kPfFlThreads) {
                const int cr = c / kPfFlPCols, d = dbase + c % kPfFlPCols;
                ps[c] = (d >= 0 && d < kInkRelExtent) ? rel_proj[(size_t)cr * kInkRelExtent + d] * kLog2e : 0.0f;
            }
        }
        __syncthreads();
        if (t0 + kPfFlBN < ke) pffl_fetch<FP8>(stg, kv, slot0, kh, t0 + kPfFlBN, ke);

        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; ++i) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            uint32_t a0, a1, a2, a3;
            pffl_ldsm_x4(qs_a + pffl_off(qrow, kk * 2 + (lane >> 4)), a0, a1, a2, a3);
#pragma unroll
            for (int nb2 = 0; nb2 < 4; ++nb2) {
                uint32_t b0, b1, b2, b3;
                const int kr = nb2 * 16 + (lane & 7) + (lane >> 4) * 8;
                pffl_ldsm_x4(ks_a + pffl_off(kr, kk * 2 + ((lane >> 3) & 1)), b0, b1, b2, b3);
                pffl_mma<F16>(s[2 * nb2], a0, a1, a2, a3, b0, b1);
                pffl_mma<F16>(s[2 * nb2 + 1], a0, a1, a2, a3, b2, b3);
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) { s[i][0] *= sc_log2; s[i][1] *= sc_log2; s[i][2] *= sc_log2; s[i][3] *= sc_log2; }

        if (need_bias) {
            // element (row pi, key j): distance index into ps = (pi - j) + 63
            const int d0 = pi0 - 2 * t + (kPfFlBN - 1);
#pragma unroll
            for (int c4 = 0; c4 < kInkDRel; c4 += 4) {
                const float4 ra = __ldg(reinterpret_cast<const float4*>(r0p + c4));
                const float4 rbv = __ldg(reinterpret_cast<const float4*>(r1p + c4));
                const float rr0[4] = {ra.x, ra.y, ra.z, ra.w}, rr1[4] = {rbv.x, rbv.y, rbv.z, rbv.w};
#pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    const float* pc = ps + (c4 + cc) * kPfFlPCols;
#pragma unroll
                    for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
                        for (int e = 0; e < 2; ++e) {
                            const int di = d0 - 8 * nb - e;               // row g
                            s[nb][e]     = fmaf(rr0[cc], pc[di], s[nb][e]);
                            s[nb][2 + e] = fmaf(rr1[cc], pc[di + 8], s[nb][2 + e]);
                        }
                    }
                }
            }
        }
        if (need_mask) {
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int kp = t0 + nb * 8 + 2 * t + e;
                    const bool ok0 = kp <= qp0 && kp < ke && (!is_swa || qp0 - kp < kInkSwaWin);
                    const bool ok1 = kp <= qp1 && kp < ke && (!is_swa || qp1 - kp < kInkSwaWin);
                    if (!ok0) s[nb][e] = -INFINITY;
                    if (!ok1) s[nb][2 + e] = -INFINITY;
                }
            }
        }

        // online softmax (row max over the quad = 64 keys)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            float mx = row_m[hh];
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) mx = fmaxf(mx, fmaxf(s[nb][2 * hh], s[nb][2 * hh + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float alpha = exp2f(row_m[hh] - mx);
            row_m[hh] = mx;
            float ls = 0.0f;
#pragma unroll
            for (int nb = 0; nb < 8; ++nb) {
                s[nb][2 * hh]     = exp2f(s[nb][2 * hh] - mx);
                s[nb][2 * hh + 1] = exp2f(s[nb][2 * hh + 1] - mx);
                ls += s[nb][2 * hh] + s[nb][2 * hh + 1];
            }
            row_l[hh] = row_l[hh] * alpha + ls;
#pragma unroll
            for (int nd = 0; nd < 16; ++nd) { o[nd][2 * hh] *= alpha; o[nd][2 * hh + 1] *= alpha; }
        }

        // O += P V
#pragma unroll
        for (int k2 = 0; k2 < 4; ++k2) {
            const uint32_t a0 = pffl_pack<F16>(s[2 * k2][0], s[2 * k2][1]);
            const uint32_t a1 = pffl_pack<F16>(s[2 * k2][2], s[2 * k2][3]);
            const uint32_t a2 = pffl_pack<F16>(s[2 * k2 + 1][0], s[2 * k2 + 1][1]);
            const uint32_t a3 = pffl_pack<F16>(s[2 * k2 + 1][2], s[2 * k2 + 1][3]);
            const int vr = k2 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int nd2 = 0; nd2 < 8; ++nd2) {
                uint32_t b0, b1, b2, b3;
                pffl_ldsm_x4_t(vs_a + pffl_off(vr, nd2 * 2 + (lane >> 4)), b0, b1, b2, b3);
                pffl_mma<F16>(o[2 * nd2], a0, a1, a2, a3, b0, b1);
                pffl_mma<F16>(o[2 * nd2 + 1], a0, a1, a2, a3, b2, b3);
            }
        }
    }

    // finish: O / l * sigmoid(gate)
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
        float l = row_l[hh];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        const int pi = hh ? pi1 : pi0;
        if (pi >= nrows) continue;
        const float inv = 1.0f / l;
        const size_t ob = ((size_t)(m0 + pi) * kInkHeads + h) * kInkHeadDim;
#pragma unroll
        for (int nd = 0; nd < 16; ++nd) {
            const int d = nd * 8 + 2 * t;
            float2 v = make_float2(o[nd][2 * hh] * inv, o[nd][2 * hh + 1] * inv);
            if (gate) {
                const float2 gt = *reinterpret_cast<const float2*>(gate + ob + d);
                v.x *= 1.0f / (1.0f + __expf(-gt.x));
                v.y *= 1.0f / (1.0f + __expf(-gt.y));
            }
            *reinterpret_cast<float2*>(out + ob + d) = v;
        }
    }
}

// The fused kernel handles every prefill (M >= 128) on this card, so the cuBLAS path's buffers (K/V pack, score tiles,
// a [M, 24, 1025] bias table) are not needed. INK_PF_FLASH_CHECK runs both paths, so it keeps them.
inline bool ink_pf_flash_needs_old_bufs(int sm_cc) {
    static const bool off = std::getenv("INK_PF_FLASH_OFF") != nullptr;
    static const bool check = std::getenv("INK_PF_FLASH_CHECK") != nullptr;
    return off || check || sm_cc < 80;
}
// rows of rel_table still needed: decode, batched decode, tree and the 17..127-token path
inline int ink_rel_table_rows(int sm_cc, int max_batch) {
    return ink_pf_flash_needs_old_bufs(sm_cc) ? max_batch : std::min(max_batch, 128);
}

template <bool FP8, bool F16>
inline void ink_pf_flash_go(cudaStream_t st, const float* q, const float* r, const float* rel_proj, const float* sinks,
                            const float* gate, float* out, const InkKvVisitDesc& kv, int slot0, int p0, int M,
                            int k_start, bool is_swa) {
    static bool attr_set[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 64 && !attr_set[dev]) {
        cudaFuncSetAttribute(k_ink_pf_flash<FP8, F16>, cudaFuncAttributeMaxDynamicSharedMemorySize, kPfFlSmem);
        attr_set[dev] = true;
    }
    const dim3 grid((M + kPfFlBM - 1) / kPfFlBM, kInkKVHeads);
    k_ink_pf_flash<FP8, F16><<<grid, kPfFlThreads, kPfFlSmem, st>>>(q, r, rel_proj, sinks, gate, out, kv, slot0, p0, M,
                                                                     k_start, is_swa ? 1 : 0);
}

// Host launcher. Returns false when the kernel does not apply (pre-sm_80); the caller keeps its older path.
// INK_PF_FLASH_OFF=1 turns it off, INK_PF_FLASH_BF16=1 uses BF16 inputs instead of FP16.
inline bool ink_pf_flash_launch(cudaStream_t st, int sm_cc, const float* q, const float* r, const float* rel_proj,
                                const float* sinks, const float* gate, float* out, const InkKvVisitDesc& kv,
                                int slot0, int p0, int M, int k_start, bool is_swa) {
    static const bool off = std::getenv("INK_PF_FLASH_OFF") != nullptr;
    static const bool bf16 = std::getenv("INK_PF_FLASH_BF16") != nullptr;
    if (off || sm_cc < 80 || M <= 0) return false;
    if (kv.is_fp8) {
        if (bf16) ink_pf_flash_go<true, false>(st, q, r, rel_proj, sinks, gate, out, kv, slot0, p0, M, k_start, is_swa);
        else      ink_pf_flash_go<true, true>(st, q, r, rel_proj, sinks, gate, out, kv, slot0, p0, M, k_start, is_swa);
    } else {
        if (bf16) ink_pf_flash_go<false, false>(st, q, r, rel_proj, sinks, gate, out, kv, slot0, p0, M, k_start, is_swa);
        else      ink_pf_flash_go<false, true>(st, q, r, rel_proj, sinks, gate, out, kv, slot0, p0, M, k_start, is_swa);
    }
    return true;
}

}  // namespace strata::ink
