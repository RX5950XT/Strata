// engine/src/inkling2/cuda/ink_dec_tc.cuh - fork: decode / tree-verify attention for global visits on FP16 tensor cores
// (mma.sync m16n8k16, sm_80+). The rows of one MMA tile are (query head, query) pairs of one KV head, so up to 5
// queries x 3 heads share one warp and every K/V byte is read once for all of them; 6..16 queries use 2 or 3 warps
// (one per 16 rows) that read the same K/V through L1. K/V go straight from global memory into the MMA fragments
// (no shared memory): the head dimension and the output dimension are permuted so each thread reads whole 16/32-byte
// pieces of a key row. FP8 E4M3 becomes FP16 with two bit operations (value * 2^-8, folded into the scales).
//
// Same partial layout as k_ink_flash_dec16 ([M, 24, kInkMqSplits(, 128)], merged by k_ink_flash_mq_merge), but one key
// partition for all queries: splits cover the cache keys [0, kc), kc = q_pos[M-1] + 1 (mode 0, causal mask per row) or
// q_pos[0] (tree: the tree keys are added by k_ink_flash_tree_tail as one more split).
#pragma once
#include "ink_attn.cuh"
#include "ink_pf_flash.cuh"

namespace strata::ink {

// two E4M3 bytes held in bits 15..8 and 31..24 -> FP16x2 bit patterns of value * 2^-8 (exact, subnormals included).
// KV writes clamp to +-448, so the NaN codes 0x7F / 0xFF never occur.
__device__ __forceinline__ uint32_t dtc_f8x2(uint32_t t) { return (t & 0x80008000u) | ((t >> 1) & 0x3F803F80u); }

// T = MMA row tiles (warps): rows r = 16 * warp + i map to query head r / M and query r % M (rows >= 3M unused).
// FP8: E4M3 KV with per-row scales, else FP16 KV. DRAFT: the DSpark draft block (keys [0, *q_pos), no causal mask,
// no relative bias, no sink). grid (nsp, 8 KV heads), 32 * T threads.
template <int T, bool FP8, bool DRAFT>
__global__ void __launch_bounds__(32 * T) k_ink_flash_dec_tc(
    const float* __restrict__ q, const float* __restrict__ rel_table, const float* __restrict__ sinks,
    InkKvVisitDesc kv, int slot0, const int32_t* __restrict__ q_pos, int tree,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_o, int M, int nsp) {
    constexpr float kLog2e = 1.4426950408889634f;
    const int sp = blockIdx.x, kh = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int kc = (DRAFT || tree) ? q_pos[0] : q_pos[M - 1] + 1;
    const int per_sp = (((kc + nsp - 1) / nsp) + 15) & ~15;
    const int s_beg = sp * per_sp, s_end = min(kc, s_beg + per_sp);

    // this thread's two rows (g and g + 8 of the warp's tile)
    int qp[2], hrow[2], mrow[2];
    bool valid[2];
    const float* rt[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int R = warp * 16 + g + 8 * i;
        valid[i] = R < 3 * M;
        const int rr = valid[i] ? R / M : 0, mm = valid[i] ? R % M : 0;
        hrow[i] = 3 * kh + rr;
        mrow[i] = mm;
        qp[i] = !valid[i] ? -1 : DRAFT ? INT_MAX : q_pos[mm];       // -1: every key masked
        rt[i] = rel_table + ((size_t)mm * kInkHeads + hrow[i]) * (kInkRelExtent + 1);
    }
    const int qp_lo = min(valid[0] ? qp[0] : INT_MAX, valid[1] ? qp[1] : INT_MAX);

    // Q fragments: k-step kk, k index 2t+e -> dim 32t + 4kk + e, 2t+8+e -> dim 32t + 4kk + 2 + e
    uint32_t qa[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        float4 v[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            v[i] = valid[i] ? __ldg(reinterpret_cast<const float4*>(
                                  q + ((size_t)mrow[i] * kInkHeads + hrow[i]) * kInkHeadDim + 32 * t + 4 * kk))
                            : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        qa[kk][0] = pffl_pack<true>(v[0].x, v[0].y);
        qa[kk][1] = pffl_pack<true>(v[1].x, v[1].y);
        qa[kk][2] = pffl_pack<true>(v[0].z, v[0].w);
        qa[kk][3] = pffl_pack<true>(v[1].z, v[1].w);
    }

    float row_m[2], row_l[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const bool first = !DRAFT && (sp == 0) && valid[i];
        row_m[i] = first ? sinks[(size_t)mrow[i] * kInkHeads + hrow[i]] * kLog2e : -1e30f;
        row_l[i] = (first && t == 0) ? 1.0f : 0.0f;    // the sink counted once per quad
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;

    const float sc_k = (FP8 ? 256.0f : 1.0f) * 0.08838834764831845f * kLog2e;   // FP8: undo the 2^-8 of the FP16 pattern
    const size_t row0 = (size_t)slot0 * kv.cap_pos;
    for (int base = s_beg; base < s_end; base += 16) {
        // K: keys base + g, base + 8 + g, bytes 32t..32t+31; V: keys base + 2t (+1, +8, +9), bytes 16g..16g+15
        constexpr int KW = FP8 ? 2 : 4;          // uint4 per thread per key: K 32 dims, V 16 dims
        constexpr int VW = FP8 ? 1 : 2;
        uint4 kr[2 * KW], vr[4 * VW];
        float ks[2][2], vs[2][2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int kp = base + 8 * j + g;
            const size_t rk = (row0 + kp) * kInkKVHeads + kh;
#pragma unroll
            for (int w = 0; w < KW; ++w) {
                if (kp < s_end) {
                    kr[KW * j + w] = FP8 ? __ldg(reinterpret_cast<const uint4*>(kv.k_f8 + rk * kInkHeadDim) + 2 * t + w)
                                         : __ldg(reinterpret_cast<const uint4*>(kv.k_f16 + rk * kInkHeadDim) + 4 * t + w);
                } else {
                    kr[KW * j + w] = make_uint4(0, 0, 0, 0);
                }
            }
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int vp = base + 8 * j + 2 * t + e;
                const size_t ri = (row0 + vp) * kInkKVHeads + kh;
#pragma unroll
                for (int w = 0; w < VW; ++w) {
                    if (vp < s_end) {
                        vr[VW * (2 * j + e) + w] = FP8 ? __ldg(reinterpret_cast<const uint4*>(kv.v_f8 + ri * kInkHeadDim) + g)
                                                       : __ldg(reinterpret_cast<const uint4*>(kv.v_f16 + ri * kInkHeadDim) + 2 * g + w);
                    } else {
                        vr[VW * (2 * j + e) + w] = make_uint4(0, 0, 0, 0);
                    }
                }
                if constexpr (FP8) {
                    ks[j][e] = vp < s_end ? __ldg(kv.k_sc + ri) * sc_k : 0.0f;
                    vs[j][e] = vp < s_end ? __ldg(kv.v_sc + ri) * 256.0f : 0.0f;
                } else {
                    ks[j][e] = sc_k;
                    vs[j][e] = 1.0f;
                }
            }
        }

        // S = Q K^T (16 rows x 16 keys): n-tile j col c = key base + 8j + c
        float s[2][4];
#pragma unroll
        for (int j = 0; j < 2; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                uint32_t b0, b1;
                if constexpr (FP8) {
                    const uint4 w4 = kr[2 * j + (kk >> 2)];
                    const uint32_t w = (kk & 3) == 0 ? w4.x : (kk & 3) == 1 ? w4.y : (kk & 3) == 2 ? w4.z : w4.w;
                    b0 = dtc_f8x2(__byte_perm(w, 0, 0x1000));
                    b1 = dtc_f8x2(__byte_perm(w, 0, 0x3020));
                } else {   // FP16: dims 32t + 4kk .. +3 are words 2kk, 2kk + 1 of the thread's 64 bytes
                    const uint4 w4 = kr[4 * j + (kk >> 1)];
                    b0 = (kk & 1) ? w4.z : w4.x;
                    b1 = (kk & 1) ? w4.w : w4.y;
                }
                pffl_mma<true>(s[j], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
            }
        }

        const bool tail = !DRAFT && base + 16 > qp_lo + 1 - kInkRelExtent;   // some key within 1024 of a row (bias, causal)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int kp = base + 8 * j + 2 * t + e;
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    float v = s[j][2 * i + e] * ks[j][e];
                    if (tail) {
                        const int dist = qp[i] - kp;
                        if (dist < 0) v = -INFINITY;
                        else if (dist < kInkRelExtent) v = fmaf(__ldg(rt[i] + dist), kLog2e, v);
                    }
                    if (kp >= s_end || !valid[i]) v = -INFINITY;
                    s[j][2 * i + e] = v;
                }
            }
        }

        // online softmax per row over the quad's 16 keys
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float mx = fmaxf(fmaxf(s[0][2 * i], s[0][2 * i + 1]), fmaxf(s[1][2 * i], s[1][2 * i + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            mx = fmaxf(mx, row_m[i]);
            const float alpha = exp2f(row_m[i] - mx);
            row_m[i] = mx;
            float ls = 0.0f;
#pragma unroll
            for (int j = 0; j < 2; ++j) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const float p = exp2f(s[j][2 * i + e] - mx);
                    ls += p;
                    s[j][2 * i + e] = p * vs[j][e];     // the V scale rides on P
                }
            }
            row_l[i] = row_l[i] * alpha + ls;
#pragma unroll
            for (int nd = 0; nd < 16; ++nd) { o[nd][2 * i] *= alpha; o[nd][2 * i + 1] *= alpha; }
        }

        // O += P V: k index 2t(+1) = key base + 2t(+1), 2t+8(+1) = key base + 8 + 2t(+1); n-tile nd col n = dim 16n + nd
        const uint32_t pa0 = pffl_pack<true>(s[0][0], s[0][1]), pa1 = pffl_pack<true>(s[0][2], s[0][3]);
        const uint32_t pa2 = pffl_pack<true>(s[1][0], s[1][1]), pa3 = pffl_pack<true>(s[1][2], s[1][3]);
        if constexpr (FP8) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                const uint32_t x0 = w == 0 ? vr[0].x : w == 1 ? vr[0].y : w == 2 ? vr[0].z : vr[0].w;
                const uint32_t x1 = w == 0 ? vr[1].x : w == 1 ? vr[1].y : w == 2 ? vr[1].z : vr[1].w;
                const uint32_t x2 = w == 0 ? vr[2].x : w == 1 ? vr[2].y : w == 2 ? vr[2].z : vr[2].w;
                const uint32_t x3 = w == 0 ? vr[3].x : w == 1 ? vr[3].y : w == 2 ? vr[3].z : vr[3].w;
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t sel = (uint32_t)(b | (b << 4) | (b << 8) | ((4 + b) << 12));
                    pffl_mma<true>(o[4 * w + b], pa0, pa1, pa2, pa3,
                                   dtc_f8x2(__byte_perm(x0, x1, sel)), dtc_f8x2(__byte_perm(x2, x3, sel)));
                }
            }
        } else {   // FP16: dim 16g + nd is half nd of the key's 32 bytes; pair the same half of two keys
#pragma unroll
            for (int w = 0; w < 8; ++w) {
                const uint4 a0 = vr[0 + (w >> 2)], a1 = vr[2 + (w >> 2)], a2 = vr[4 + (w >> 2)], a3 = vr[6 + (w >> 2)];
                const int c = w & 3;
                const uint32_t x0 = c == 0 ? a0.x : c == 1 ? a0.y : c == 2 ? a0.z : a0.w;
                const uint32_t x1 = c == 0 ? a1.x : c == 1 ? a1.y : c == 2 ? a1.z : a1.w;
                const uint32_t x2 = c == 0 ? a2.x : c == 1 ? a2.y : c == 2 ? a2.z : a2.w;
                const uint32_t x3 = c == 0 ? a3.x : c == 1 ? a3.y : c == 2 ? a3.z : a3.w;
                pffl_mma<true>(o[2 * w], pa0, pa1, pa2, pa3, __byte_perm(x0, x1, 0x5410), __byte_perm(x2, x3, 0x5410));
                pffl_mma<true>(o[2 * w + 1], pa0, pa1, pa2, pa3, __byte_perm(x0, x1, 0x7632), __byte_perm(x2, x3, 0x7632));
            }
        }
    }

    // partials: thread holds dims 32t + nd (o[nd][0/2]) and 32t + 16 + nd (o[nd][1/3])
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        float l = row_l[i];
        l += __shfl_xor_sync(0xffffffffu, l, 1);
        l += __shfl_xor_sync(0xffffffffu, l, 2);
        if (!valid[i]) continue;
        const size_t idx = ((size_t)mrow[i] * kInkHeads + hrow[i]) * kInkMqSplits + sp;
        if (t == 0) { part_m[idx] = row_m[i] * 0.6931471805599453f; part_l[idx] = l; }
        float* dst = part_o + idx * kInkHeadDim + 32 * t;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            reinterpret_cast<float4*>(dst)[c] =
                make_float4(o[4 * c][2 * i], o[4 * c + 1][2 * i], o[4 * c + 2][2 * i], o[4 * c + 3][2 * i]);
            reinterpret_cast<float4*>(dst + 16)[c] =
                make_float4(o[4 * c][2 * i + 1], o[4 * c + 1][2 * i + 1], o[4 * c + 2][2 * i + 1], o[4 * c + 3][2 * i + 1]);
        }
    }
}

// Tree keys q_pos[0] .. q_pos[m] (the ancestors of node m, K/V in tree_k / tree_v) as split `slot` of the partials.
// grid (M, 8 KV heads), 96 threads: warp r = query head 3 * kh + r, lane = dims 4 lane .. 4 lane + 3.
__global__ void __launch_bounds__(96) k_ink_flash_tree_tail(
    const float* __restrict__ q, const float* __restrict__ rel_table, const int32_t* __restrict__ q_pos,
    const float* __restrict__ tree_k, const float* __restrict__ tree_v, const uint8_t* __restrict__ tree_anc,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_o, int M, int slot) {
    const int m = blockIdx.x, kh = blockIdx.y, r = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (m >= M) return;
    const int h = 3 * kh + r;
    const int tp0 = q_pos[0], qp = q_pos[m];
    const int n = min(qp - tp0 + 1, kInkMaxTreeDepth);
    const float4 qv = reinterpret_cast<const float4*>(q + ((size_t)m * kInkHeads + h) * kInkHeadDim)[lane];
    const float* rt = rel_table + ((size_t)m * kInkHeads + h) * (kInkRelExtent + 1);
    float sc[kInkMaxTreeDepth];
    float mx = -1e30f;
#pragma unroll
    for (int j = 0; j < kInkMaxTreeDepth; ++j) {
        sc[j] = -INFINITY;
        if (j < n) {
            const int node = (int)tree_anc[m * kInkMaxTreeDepth + j];
            const float4 kv4 = reinterpret_cast<const float4*>(tree_k + ((size_t)node * kInkKVHeads + kh) * kInkHeadDim)[lane];
            float d = qv.x * kv4.x + qv.y * kv4.y + qv.z * kv4.z + qv.w * kv4.w;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            const int dist = min(max(qp - (tp0 + j), 0), kInkRelExtent);
            sc[j] = d * 0.08838834764831845f + rt[dist];
            mx = fmaxf(mx, sc[j]);
        }
    }
    float l = 0.0f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int j = 0; j < kInkMaxTreeDepth; ++j) {
        if (j < n) {
            const int node = (int)tree_anc[m * kInkMaxTreeDepth + j];
            const float p = __expf(sc[j] - mx);
            const float4 v4 = reinterpret_cast<const float4*>(tree_v + ((size_t)node * kInkKVHeads + kh) * kInkHeadDim)[lane];
            l += p;
            acc.x = fmaf(p, v4.x, acc.x); acc.y = fmaf(p, v4.y, acc.y);
            acc.z = fmaf(p, v4.z, acc.z); acc.w = fmaf(p, v4.w, acc.w);
        }
    }
    const size_t idx = ((size_t)m * kInkHeads + h) * kInkMqSplits + slot;
    if (lane == 0) { part_m[idx] = mx; part_l[idx] = l; }
    reinterpret_cast<float4*>(part_o + idx * kInkHeadDim)[lane] = acc;
}

// DSpark draft block: every draft query sees all B draft keys (kd / vd [B, 8, 128] FP32), written as split `slot`.
// grid (B, 8 KV heads), 96 threads: warp r = query head 3 * kh + r.
__global__ void __launch_bounds__(96) k_dspark_tc_tail(
    const float* __restrict__ q, const float* __restrict__ kd, const float* __restrict__ vd,
    float* __restrict__ part_m, float* __restrict__ part_l, float* __restrict__ part_o, int B, int slot) {
    const int b = blockIdx.x, kh = blockIdx.y, r = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (b >= B) return;
    const int h = 3 * kh + r;
    const float4 qv = reinterpret_cast<const float4*>(q + ((size_t)b * kInkHeads + h) * kInkHeadDim)[lane];
    float sc[kInkDsparkBlock];
    float mx = -1e30f;
#pragma unroll
    for (int j = 0; j < kInkDsparkBlock; ++j) {
        sc[j] = -INFINITY;
        if (j < B) {
            const float4 k4 = reinterpret_cast<const float4*>(kd + ((size_t)j * kInkKVHeads + kh) * kInkHeadDim)[lane];
            float d = qv.x * k4.x + qv.y * k4.y + qv.z * k4.z + qv.w * k4.w;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);
            sc[j] = d * 0.08838834764831845f;
            mx = fmaxf(mx, sc[j]);
        }
    }
    float l = 0.0f;
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int j = 0; j < kInkDsparkBlock; ++j) {
        if (j < B) {
            const float p = __expf(sc[j] - mx);
            const float4 v4 = reinterpret_cast<const float4*>(vd + ((size_t)j * kInkKVHeads + kh) * kInkHeadDim)[lane];
            l += p;
            acc.x = fmaf(p, v4.x, acc.x); acc.y = fmaf(p, v4.y, acc.y);
            acc.z = fmaf(p, v4.z, acc.z); acc.w = fmaf(p, v4.w, acc.w);
        }
    }
    const size_t idx = ((size_t)b * kInkHeads + h) * kInkMqSplits + slot;
    if (lane == 0) { part_m[idx] = mx; part_l[idx] = l; }
    reinterpret_cast<float4*>(part_o + idx * kInkHeadDim)[lane] = acc;
}

// DSpark draft attention of a global layer: ctx keys [0, *apos) (FP16, not a ring) + the B draft keys -> out [B, 24, 128].
// Partials need [B, 24, kInkMqSplits(, 128)] floats. Same result as k_dspark_block_attn up to FP16 rounding of q and p.
inline void ink_dspark_attn_tc(cudaStream_t st, const float* q, const float* kd, const float* vd,
                               const __half* ctx_k, const __half* ctx_v, int cap, const int32_t* apos, int B,
                               float* pm, float* pl, float* po, float* out) {
    InkKvVisitDesc kv{};
    kv.k_f16 = const_cast<__half*>(ctx_k);
    kv.v_f16 = const_cast<__half*>(ctx_v);
    kv.cap_pos = cap;
    kv.is_swa = false;
    kv.is_fp8 = false;
    const int nsp = kInkMqSplits;
    const dim3 grid(nsp - 1, kInkKVHeads);
    const int T = (3 * B + 15) / 16;
    if (T == 1)      k_ink_flash_dec_tc<1, false, true><<<grid, 32, 0, st>>>(q, nullptr, nullptr, kv, 0, apos, 0, pm, pl, po, B, nsp - 1);
    else if (T == 2) k_ink_flash_dec_tc<2, false, true><<<grid, 64, 0, st>>>(q, nullptr, nullptr, kv, 0, apos, 0, pm, pl, po, B, nsp - 1);
    else             k_ink_flash_dec_tc<3, false, true><<<grid, 96, 0, st>>>(q, nullptr, nullptr, kv, 0, apos, 0, pm, pl, po, B, nsp - 1);
    k_dspark_tc_tail<<<dim3(B, kInkKVHeads), 96, 0, st>>>(q, kd, vd, pm, pl, po, B, nsp - 1);
    k_ink_flash_mq_merge<<<dim3(kInkHeads, B), 128, 0, st>>>(pm, pl, po, nullptr, out, B, nsp);
}

// true when the tensor-core path applies (sm_80+, FP8 KV, <= 16 queries); INK_DEC_TC_OFF=1 keeps k_ink_flash_dec16
inline bool ink_dec_tc_ok(int sm_cc, const InkKvVisitDesc& kv, int M) {
    static const bool off = std::getenv("INK_DEC_TC_OFF") != nullptr;
    return !off && sm_cc >= 80 && kv.is_fp8 && !kv.is_swa && M >= 1 && M <= 16;
}

// nsp = total partial slots the merge reads; tree mode uses the last one for the tree keys
inline void ink_dec_tc_launch(cudaStream_t st, const float* q, const float* rel_table, const float* sinks,
                              const InkKvVisitDesc& kv, int slot0, const int32_t* q_pos, bool tree,
                              const float* tree_k, const float* tree_v, const uint8_t* tree_anc,
                              float* pm, float* pl, float* po, int M, int nsp) {
    const int nsp_tc = tree ? nsp - 1 : nsp;
    const int T = (3 * M + 15) / 16;
    const dim3 grid(nsp_tc, kInkKVHeads);
    if (T == 1)      k_ink_flash_dec_tc<1, true, false><<<grid, 32, 0, st>>>(q, rel_table, sinks, kv, slot0, q_pos, tree ? 1 : 0, pm, pl, po, M, nsp_tc);
    else if (T == 2) k_ink_flash_dec_tc<2, true, false><<<grid, 64, 0, st>>>(q, rel_table, sinks, kv, slot0, q_pos, tree ? 1 : 0, pm, pl, po, M, nsp_tc);
    else             k_ink_flash_dec_tc<3, true, false><<<grid, 96, 0, st>>>(q, rel_table, sinks, kv, slot0, q_pos, tree ? 1 : 0, pm, pl, po, M, nsp_tc);
    if (tree)
        k_ink_flash_tree_tail<<<dim3(M, kInkKVHeads), 96, 0, st>>>(q, rel_table, q_pos, tree_k, tree_v, tree_anc, pm, pl, po, M, nsp - 1);
}

}  // namespace strata::ink
