// engine/src/inkling2/cuda/ink_remote_attn.cuh - fork: the 8 global-attention visits run on a second GPU.
//
// Why: at 200K context the global layers' KV (8 visits x 16.9 KB per token = 3.4 GB in FP8) does not fit next to the
// weights on a 16 GB card, and the two consumer GPUs here have no peer access. The second card holds that KV and
// computes those 8 attentions; everything else stays on the main GPU.
//
// How (per global visit, also inside the main GPU's CUDA graphs):
//   main GPU:  k_ra_pack  writes q, r, sinks, gate, positions and the new K/V rows into mapped host memory,
//              k_ra_ring  rings the doorbell, k_ra_wait_copy spins until the answer is back and copies it into place.
//   host thread (bound to the second GPU): sees the ring, copies the request up, appends the K/V rows to its KV cache,
//              runs the same ink_attention() code the main GPU would, copies the result down and acks.
// Tree-verify K/V are kept in a stage buffer on the second GPU so accepted tokens are committed there too
// (ink_commit_kv in ink_spec.cuh). Positions and tree shape travel with each request, so graph replays stay valid.
// Two request/answer buffers: a prompt chunk can run as two halves whose global visits alternate on the two GPUs
// (the main GPU computes one half while the second GPU does the other half's attention). Requests are numbered;
// the host thread serves them in order and the second GPU acks with its own running count.
#pragma once
#include "ink_attn_dispatch.cuh"

#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::ink {

struct InkRaReq { volatile int32_t visit, mode, M, slot0, p0, buf; };
struct alignas(64) InkRaBell {
    volatile uint32_t ring;          // number of the newest request
    uint32_t pad0[15];
    volatile uint32_t ack;           // number of the newest finished request
    uint32_t pad1[15];
    InkRaReq req[2];                 // request n uses req[n & 1] (at most two are ever outstanding)
    volatile int32_t err;
};

// Byte offsets inside the request buffer, fixed for max_batch tokens (copies use the first M rows of each part).
struct InkRaLayout {
    size_t q = 0, r = 0, sinks = 0, gate = 0, pos = 0, slots = 0, anc = 0, dep = 0, tk = 0, tv = 0, kv = 0, total = 0;
    int kv_row = 0;  // bytes per token of K+V(+scales) as stored in the cache
    // compact for exactly MB tokens: one copy moves a whole request
    void build(int MB, bool fp8) {
        auto a = [](size_t x) { return (x + 255) & ~(size_t)255; };
        kv_row = fp8 ? (2 * kInkKVDim + 2 * kInkKVHeads * 4) : (2 * kInkKVDim * 2);
        size_t o = 0;
        q = o;     o += a((size_t)MB * kInkQDim * 4);
        r = o;     o += a((size_t)MB * kInkHeads * kInkDRel * 4);
        sinks = o; o += a((size_t)MB * kInkHeads * 4);
        gate = o;  o += a((size_t)MB * kInkQDim * 4);
        pos = o;   o += a((size_t)MB * 4);
        slots = o; o += a((size_t)MB * 4);
        anc = o;   o += a((size_t)MB * kInkMaxTreeDepth);
        dep = o;   o += a((size_t)MB);
        tk = o;    o += a((size_t)MB * kInkKVDim * 4);
        tv = o;    o += a((size_t)MB * kInkKVDim * 4);
        kv = o;    o += a((size_t)MB * kv_row);
        total = o;
    }
};

// ---- main GPU side ----
// grid (M), block 256: gather one token's attention inputs into the mapped request buffer
__global__ void __launch_bounds__(256) k_ra_pack(
    uint8_t* __restrict__ req, InkRaLayout L,
    const float* __restrict__ q, const float* __restrict__ r, const float* __restrict__ sinks, const float* __restrict__ gate,
    const int32_t* __restrict__ pos, const int32_t* __restrict__ slots, int slot0,
    const uint8_t* __restrict__ anc, const uint8_t* __restrict__ dep,
    const float* __restrict__ tk, const float* __restrict__ tv,   // tree mode
    InkKvVisitDesc ring, int mode, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    float* dq = reinterpret_cast<float*>(req + L.q) + (size_t)m * kInkQDim;
    float* dg = reinterpret_cast<float*>(req + L.gate) + (size_t)m * kInkQDim;
    for (int i = tid; i < kInkQDim; i += 256) { dq[i] = q[(size_t)m * kInkQDim + i]; dg[i] = gate[(size_t)m * kInkQDim + i]; }
    float* dr = reinterpret_cast<float*>(req + L.r) + (size_t)m * kInkHeads * kInkDRel;
    for (int i = tid; i < kInkHeads * kInkDRel; i += 256) dr[i] = r[(size_t)m * kInkHeads * kInkDRel + i];
    if (tid < kInkHeads) reinterpret_cast<float*>(req + L.sinks)[(size_t)m * kInkHeads + tid] = sinks[(size_t)m * kInkHeads + tid];
    const int sl = (mode == 1 && slots) ? slots[m] : slot0;
    if (tid == 0) {
        reinterpret_cast<int32_t*>(req + L.pos)[m] = pos[m];
        reinterpret_cast<int32_t*>(req + L.slots)[m] = sl;
        if (mode == 2) req[L.dep + m] = dep[m];
    }
    if (mode == 2) {
        if (tid < kInkMaxTreeDepth) req[L.anc + (size_t)m * kInkMaxTreeDepth + tid] = anc[(size_t)m * kInkMaxTreeDepth + tid];
        float* dk = reinterpret_cast<float*>(req + L.tk) + (size_t)m * kInkKVDim;
        float* dv = reinterpret_cast<float*>(req + L.tv) + (size_t)m * kInkKVDim;
        for (int i = tid; i < kInkKVDim; i += 256) { dk[i] = tk[(size_t)m * kInkKVDim + i]; dv[i] = tv[(size_t)m * kInkKVDim + i]; }
    } else {
        // the K/V row this token wrote into the small ring cache (same bytes the full cache would hold)
        const int p_idx = pos[m] % ring.cap_pos;
        const size_t base = ((size_t)sl * ring.cap_pos + p_idx) * kInkKVDim;
        uint8_t* row = req + L.kv + (size_t)m * L.kv_row;
        if (ring.is_fp8) {
            for (int i = tid; i < kInkKVDim; i += 256) { row[i] = ring.k_f8[base + i]; row[kInkKVDim + i] = ring.v_f8[base + i]; }
            float* sc = reinterpret_cast<float*>(row + 2 * kInkKVDim);
            const size_t sidx = ((size_t)sl * ring.cap_pos + p_idx) * kInkKVHeads;
            if (tid < kInkKVHeads) { sc[tid] = ring.k_sc[sidx + tid]; sc[kInkKVHeads + tid] = ring.v_sc[sidx + tid]; }
        } else {
            __half* h = reinterpret_cast<__half*>(row);
            for (int i = tid; i < kInkKVDim; i += 256) { h[i] = ring.k_f16[base + i]; h[kInkKVDim + i] = ring.v_f16[base + i]; }
        }
    }
    __threadfence_system();
}

__global__ void k_ra_ring(InkRaBell* bell, uint32_t* seq_dev, uint32_t* cur_seq, int buf, int visit, int mode, int M, int slot0, int p0) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const uint32_t s = ++(*seq_dev);
    InkRaReq& r = bell->req[s & 1];
    r.visit = visit; r.mode = mode; r.M = M; r.slot0 = slot0; r.p0 = p0; r.buf = buf;
    __threadfence_system();
    cur_seq[buf] = s;
    bell->ring = s;
    __threadfence_system();
}

// grid (blocks), block 256: wait for the answer (thread 0 of each block spins), then copy it into out [M, 3072]
__global__ void __launch_bounds__(256) k_ra_wait_copy(const InkRaBell* bell, const uint32_t* cur_seq,
                                                      const float* __restrict__ ans, float* __restrict__ out, int M) {
    __shared__ int ok;
    if (threadIdx.x == 0) {
        const uint32_t want = *cur_seq;
        const long long t0 = clock64();
        int good = 1;
        while ((int)(bell->ack - want) < 0) {
            if (clock64() - t0 > 60ll * 3000000000ll) { good = 0; break; }  // ~60 s: never hang the stream forever
        }
        __threadfence_system();
        ok = good;
    }
    __syncthreads();
    if (!ok) return;
    const size_t n4 = (size_t)M * kInkQDim / 4;
    const float4* src = reinterpret_cast<const float4*>(ans);
    float4* dst = reinterpret_cast<float4*>(out);
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n4; i += (size_t)gridDim.x * 256) dst[i] = __ldcv(src + i);
}

// ---- second GPU side ----
// Keeps the KV GPU at working clocks while requests flow. Measured on the RTX 3060 Ti: tiny bursty work left it in
// P8 and one M=1 attention took 1.1-1.3 ms instead of tens of microseconds (same fix as engine/src/kernels/cuda/keep_warm.cu).
__global__ void k_ra_keep_warm(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        __nanosleep(100000);
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
// last node of every request: the answer is already in host memory (stream order), tell the main GPU.
// Requests are served in order, so the running count on this GPU is the request's number.
__global__ void k_ra_ack(InkRaBell* bell, uint32_t* count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence_system();
        bell->ack = ++(*count);
        __threadfence_system();
    }
}

// grid (M), block 256: copy each token's K/V row (+scales) into the full cache at its position
__global__ void __launch_bounds__(256) k_ra_import_kv(const uint8_t* __restrict__ rows, int kv_row,
                                                      const int32_t* __restrict__ pos, const int32_t* __restrict__ slots,
                                                      InkKvVisitDesc kv, int M) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const int tid = threadIdx.x;
    const int p = pos[m], sl = slots[m];
    const size_t base = ((size_t)sl * kv.cap_pos + p) * kInkKVDim;
    const uint8_t* row = rows + (size_t)m * kv_row;
    if (kv.is_fp8) {
        for (int i = tid; i < kInkKVDim; i += 256) { kv.k_f8[base + i] = row[i]; kv.v_f8[base + i] = row[kInkKVDim + i]; }
        const float* sc = reinterpret_cast<const float*>(row + 2 * kInkKVDim);
        const size_t sidx = ((size_t)sl * kv.cap_pos + p) * kInkKVHeads;
        if (tid < kInkKVHeads) { kv.k_sc[sidx + tid] = sc[tid]; kv.v_sc[sidx + tid] = sc[kInkKVHeads + tid]; }
    } else {
        const __half* h = reinterpret_cast<const __half*>(row);
        for (int i = tid; i < kInkKVDim; i += 256) { kv.k_f16[base + i] = h[i]; kv.v_f16[base + i] = h[kInkKVDim + i]; }
    }
}


struct InkRemoteAttn {
    int gpu_main = 0, gpu_kv = 1;
    int max_slots = 1, max_ctx = 0, max_batch = 0, tree_cap = 0;
    bool fp8 = true;
    InkRaLayout L;                     // layout for max_batch tokens (buffer sizes)

    // shared host memory (mapped, portable)
    InkRaBell* h_bell = nullptr;
    InkRaBell* d_bell = nullptr;       // main GPU view
    InkRaBell* d1_bell = nullptr;      // KV GPU view
    uint8_t* h_req[2] = {};
    uint8_t* d_req[2] = {};            // main GPU view
    float* h_ans[2] = {};
    float* d_ans[2] = {};              // main GPU view
    uint32_t* d_seq = nullptr;         // main GPU: requests posted
    uint32_t* d_cur_seq = nullptr;     // main GPU: [2] number of the last request per buffer
    uint32_t* d1_ack_count = nullptr;  // KV GPU: requests finished

    // KV GPU
    cudaStream_t st1 = nullptr;
    cudaStream_t warm1 = nullptr;          // lowest priority: keep-warm kernels only
    cudaEvent_t warm_done = nullptr;
    cublasHandle_t cublas1 = nullptr;
    int sm_cc1 = 0;
    InkKvVisitDesc kv1[kInkVisits]{};     // only global visits have storage
    InkKvVisitDesc* d1_desc_arr = nullptr;
    float* d1_stage_k = nullptr;          // [48, tree_cap, 1024]
    float* d1_stage_v = nullptr;
    float* d1_rel_proj[kInkPhysLayers] = {};
    uint8_t* d1_req = nullptr;            // device copy of one request (compact layout)
    int32_t* d1_path = nullptr;
    InkAttnBufs b1;                       // work buffers (rel_table, partials, prefill tiles, out)
    std::vector<void*> d1_allocs;
    struct GraphKey { int visit, mode, M; };
    std::vector<std::pair<GraphKey, cudaGraphExec_t>> graphs;

    std::thread worker;
    std::atomic<bool> stop{false};
    std::atomic<int> failed{0};
    std::mutex gpu1_mu;
    size_t gpu1_bytes = 0;

    template <typename T> T* d1alloc(size_t n) {
        T* p = nullptr;
        INK_CK(cudaMalloc(&p, std::max<size_t>(n, 1) * sizeof(T)));
        d1_allocs.push_back(p);
        gpu1_bytes += n * sizeof(T);
        return p;
    }
    InkRaLayout layout(int M) const { InkRaLayout l; l.build(M, fp8); return l; }
    static bool check_on() { static const bool on = std::getenv("INK_RA_CHECK") != nullptr; return on; }

    // rel_proj_host[ip] = host copy of layer ip rel_logits_proj for global layers, else empty
    void init(int gpu_main_, int gpu_kv_, int slots, int ctx, int mb, int tcap, bool fp8_, const bool* is_swa_visit,
              const std::vector<std::vector<float>>& rel_proj_host) {
        gpu_main = gpu_main_; gpu_kv = gpu_kv_; max_slots = slots; max_ctx = ctx; max_batch = mb; tree_cap = tcap; fp8 = fp8_;
        L.build(max_batch, fp8);

        INK_CK(cudaSetDevice(gpu_main));
        INK_CK(cudaHostAlloc(&h_bell, sizeof(InkRaBell), cudaHostAllocMapped | cudaHostAllocPortable));
        std::memset((void*)h_bell, 0, sizeof(InkRaBell));
        INK_CK(cudaHostGetDevicePointer((void**)&d_bell, (void*)h_bell, 0));
        for (int b = 0; b < 2; ++b) {
            INK_CK(cudaHostAlloc(&h_req[b], L.total, cudaHostAllocMapped | cudaHostAllocPortable));
            INK_CK(cudaHostAlloc(&h_ans[b], (size_t)max_batch * kInkQDim * 4, cudaHostAllocMapped | cudaHostAllocPortable));
            INK_CK(cudaHostGetDevicePointer((void**)&d_req[b], h_req[b], 0));
            INK_CK(cudaHostGetDevicePointer((void**)&d_ans[b], h_ans[b], 0));
        }
        INK_CK(cudaMalloc(&d_seq, 4)); INK_CK(cudaMemset(d_seq, 0, 4));
        INK_CK(cudaMalloc(&d_cur_seq, 8)); INK_CK(cudaMemset(d_cur_seq, 0, 8));

        INK_CK(cudaSetDevice(gpu_kv));
        cudaDeviceProp prop{};
        INK_CK(cudaGetDeviceProperties(&prop, gpu_kv));
        sm_cc1 = prop.major * 10 + prop.minor;
        INK_CK(cudaHostGetDevicePointer((void**)&d1_bell, (void*)h_bell, 0));
        int prio_lo = 0, prio_hi = 0;
        INK_CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi));
        INK_CK(cudaStreamCreateWithPriority(&st1, cudaStreamNonBlocking, prio_hi));
        INK_CK(cudaStreamCreateWithPriority(&warm1, cudaStreamNonBlocking, prio_lo));
        INK_CK(cudaEventCreateWithFlags(&warm_done, cudaEventDisableTiming));
        INK_CK(cudaEventRecord(warm_done, warm1));
        INK_CUBLAS_CK(cublasCreate(&cublas1));
        INK_CUBLAS_CK(cublasSetStream(cublas1, st1));
        for (int v = 0; v < kInkVisits; ++v) {
            if (is_swa_visit[v]) continue;
            InkKvVisitDesc& d = kv1[v];
            d.cap_pos = max_ctx; d.is_swa = false; d.is_fp8 = fp8;
            const size_t elems = (size_t)max_slots * max_ctx * kInkKVDim;
            if (fp8) {
                d.k_f8 = d1alloc<uint8_t>(elems); d.v_f8 = d1alloc<uint8_t>(elems);
                d.k_sc = d1alloc<float>((size_t)max_slots * max_ctx * kInkKVHeads);
                d.v_sc = d1alloc<float>((size_t)max_slots * max_ctx * kInkKVHeads);
            } else {
                d.k_f16 = d1alloc<__half>(elems); d.v_f16 = d1alloc<__half>(elems);
            }
        }
        d1_desc_arr = d1alloc<InkKvVisitDesc>(kInkVisits);
        d1_ack_count = d1alloc<uint32_t>(1);
        INK_CK(cudaMemset(d1_ack_count, 0, 4));
        INK_CK(cudaMemcpy(d1_desc_arr, kv1, sizeof(kv1), cudaMemcpyHostToDevice));
        d1_stage_k = d1alloc<float>((size_t)kInkVisits * tree_cap * kInkKVDim);
        d1_stage_v = d1alloc<float>((size_t)kInkVisits * tree_cap * kInkKVDim);
        for (int ip = 0; ip < kInkPhysLayers; ++ip) {
            if (rel_proj_host[ip].empty()) continue;
            d1_rel_proj[ip] = d1alloc<float>(rel_proj_host[ip].size());
            INK_CK(cudaMemcpy(d1_rel_proj[ip], rel_proj_host[ip].data(), rel_proj_host[ip].size() * 4, cudaMemcpyHostToDevice));
        }
        const size_t MB = (size_t)max_batch;
        d1_req = d1alloc<uint8_t>(L.total);
        d1_path = d1alloc<int32_t>(kInkMaxTree);
        b1.st = st1; b1.cublas = cublas1; b1.sm_cc = sm_cc1;
        b1.rel_table = d1alloc<float>((size_t)ink_rel_table_rows(sm_cc1, max_batch) * kInkHeads * (kInkRelExtent + 1));
        b1.out = d1alloc<float>(MB * kInkQDim);
        b1.pm = d1alloc<float>((size_t)tree_cap * kInkHeads * kInkAttnSplits);
        b1.pl = d1alloc<float>((size_t)tree_cap * kInkHeads * kInkAttnSplits);
        b1.po = d1alloc<float>((size_t)tree_cap * kInkHeads * kInkAttnSplits * kInkHeadDim);
        b1.mq_pm = d1alloc<float>((size_t)tree_cap * kInkHeads * kInkMqSplits);
        b1.mq_pl = d1alloc<float>((size_t)tree_cap * kInkHeads * kInkMqSplits);
        b1.mq_po = d1alloc<float>((size_t)tree_cap * kInkHeads * kInkMqSplits * kInkHeadDim);
        if (ink_pf_flash_needs_old_bufs(sm_cc1)) {   // fork: the fused prefill kernel needs none of these
            const size_t rows = (size_t)kInkHeads * kInkPfQCh;
            b1.pf_q = d1alloc<__nv_bfloat16>(MB * kInkHeads * kInkHeadDim);
            b1.pf_k = d1alloc<__nv_bfloat16>((size_t)kInkKVHeads * kInkPfTile * kInkHeadDim);
            b1.pf_v = d1alloc<__nv_bfloat16>((size_t)kInkKVHeads * kInkPfTile * kInkHeadDim);
            b1.pf_s = d1alloc<float>(rows * kInkPfTile);
            b1.pf_p = d1alloc<__nv_bfloat16>(rows * kInkPfTile);
            b1.pf_o = d1alloc<float>(rows * kInkHeadDim);
            b1.pf_ot = d1alloc<float>(rows * kInkHeadDim);
            b1.pf_m = d1alloc<float>(rows * 4);
        }
        INK_CK(cudaDeviceSynchronize());
        INK_CK(cudaSetDevice(gpu_main));

        stop.store(false);
        worker = std::thread([this]() { serve(); });
    }

    // Main GPU: enqueue (inside or outside graph capture) one remote attention for a global visit, in buffer buf.
    void post(cudaStream_t st, int buf, int visit, int mode, int M, int slot0, int p0,
              const float* q, const float* r, const float* sinks, const float* gate,
              const int32_t* pos, const int32_t* slots, const uint8_t* anc, const uint8_t* dep,
              const float* tk, const float* tv, const InkKvVisitDesc& ring) {
        const InkRaLayout Lm = layout(M);
        k_ra_pack<<<M, 256, 0, st>>>(d_req[buf], Lm, q, r, sinks, gate, pos, slots, slot0, anc, dep, tk, tv, ring, mode, M);
        k_ra_ring<<<1, 32, 0, st>>>(d_bell, d_seq, d_cur_seq, buf, visit, mode, M, slot0, p0);
    }
    // Main GPU: wait for the last request posted in buffer buf and copy its answer into out [M, 3072]
    void wait(cudaStream_t st, int buf, int M, float* out) {
        const int nb = std::max(1, std::min(64, (M * kInkQDim / 4 + 255) / 256));
        k_ra_wait_copy<<<nb, 256, 0, st>>>(d_bell, d_cur_seq + buf, d_ans[buf], out, M);
    }
    void post_and_wait(cudaStream_t st, int visit, int mode, int M, int slot0, int p0,
                       const float* q, const float* r, const float* sinks, const float* gate,
                       const int32_t* pos, const int32_t* slots, const uint8_t* anc, const uint8_t* dep,
                       const float* tk, const float* tv, const InkKvVisitDesc& ring, float* out) {
        post(st, 0, visit, mode, M, slot0, p0, q, r, sinks, gate, pos, slots, anc, dep, tk, tv, ring);
        wait(st, 0, M, out);
    }

    // KV GPU: commit tree-accepted K/V (host path) for the global visits. Called from the main thread when the
    // KV GPU is idle (the last request was acked by its final kernel, so its stream has drained).
    void commit(const int32_t* host_path, int p0, int n, int slot0, void (*launch_fn)(const InkKvVisitDesc*, const float*, const float*,
                const int32_t*, int, int, int, int, cudaStream_t)) {
        std::lock_guard<std::mutex> lk(gpu1_mu);
        INK_CK(cudaSetDevice(gpu_kv));
        INK_CK(cudaMemcpyAsync(d1_path, host_path, (size_t)n * 4, cudaMemcpyHostToDevice, st1));
        launch_fn(d1_desc_arr, d1_stage_k, d1_stage_v, d1_path, p0, n, slot0, tree_cap, st1);
        INK_CK(cudaStreamSynchronize(st1));
        INK_CK(cudaSetDevice(gpu_main));
    }

    void serve() {
        cudaSetDevice(gpu_kv);
        uint32_t seen = 0;
        auto last = std::chrono::steady_clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            if (h_bell->ring == seen) {
                const auto idle = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - last).count();
                if (idle > 50) { std::this_thread::sleep_for(std::chrono::microseconds(50)); continue; }
                warm();     // still busy (between requests): keep the clocks up
                continue;   // hot spin while active: every microsecond here is on the critical path
            }
            const uint32_t seq = ++seen;    // serve every request in order (two may be waiting)
            const auto t_ring = std::chrono::steady_clock::now();
            std::atomic_thread_fence(std::memory_order_acquire);
            const InkRaReq& rq = h_bell->req[seq & 1];
            const int visit = rq.visit, mode = rq.mode, M = rq.M, slot0 = rq.slot0, p0 = rq.p0, buf = rq.buf & 1;
            {
                std::lock_guard<std::mutex> lk(gpu1_mu);
                const bool ok = launch(buf, visit, mode, M, slot0, p0);
                cudaStreamQuery(st1);   // WDDM batches submissions: push this request to the GPU now
                if (!ok) {   // answer stays stale; ack so the main GPU never hangs (and keep the GPU count in step)
                    h_bell->err = 1;
                    failed.store(1);
                    cudaStreamSynchronize(st1);
                    cudaMemcpy(d1_ack_count, &seq, 4, cudaMemcpyHostToDevice);
                    std::atomic_thread_fence(std::memory_order_release);
                    h_bell->ack = seq;
                }
            }
            // fork: INK_RA_PROF=1 prints where a hand-off spends its time (every 2000 requests)
            static const bool prof = std::getenv("INK_RA_PROF") != nullptr;
            if (prof) {
                const auto t_sub = std::chrono::steady_clock::now();
                while ((int)(h_bell->ack - seq) < 0) {}
                const auto t_ack = std::chrono::steady_clock::now();
                auto us = [](auto a, auto b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
                prof_gap += us(last, t_ring); prof_sub += us(t_ring, t_sub); prof_gpu += us(t_sub, t_ack); ++prof_n;
                if (prof_n == 2000) {
                    std::fprintf(stderr, "[ra] per request: main GPU gap %.0f us, submit %.0f us, KV GPU %.0f us\n",
                                 prof_gap / prof_n, prof_sub / prof_n, prof_gpu / prof_n);
                    prof_gap = prof_sub = prof_gpu = 0; prof_n = 0;
                }
                last = t_ack;
                continue;
            }
            last = std::chrono::steady_clock::now();
        }
    }
    double prof_gap = 0, prof_sub = 0, prof_gpu = 0;
    int prof_n = 0;

    // The whole request as stream work on the KV GPU, ending with the ack kernel (no host sync needed).
    void enqueue(int buf, int visit, int mode, int M, int slot0, int p0) {
        const InkRaLayout Lm = layout(M);
        cudaMemcpyAsync(d1_req, h_req[buf], Lm.total, cudaMemcpyHostToDevice, st1);
        InkAttnBufs b = b1;
        b.q = reinterpret_cast<float*>(d1_req + Lm.q);
        b.r = reinterpret_cast<float*>(d1_req + Lm.r);
        b.sinks = reinterpret_cast<float*>(d1_req + Lm.sinks);
        b.gate = reinterpret_cast<float*>(d1_req + Lm.gate);
        b.pos = reinterpret_cast<int32_t*>(d1_req + Lm.pos);
        b.slots = reinterpret_cast<int32_t*>(d1_req + Lm.slots);
        b.anc = d1_req + Lm.anc;
        b.dep = d1_req + Lm.dep;
        b.tk = reinterpret_cast<float*>(d1_req + Lm.tk);
        b.tv = reinterpret_cast<float*>(d1_req + Lm.tv);
        if (mode == 2) {
            cudaMemcpyAsync(d1_stage_k + (size_t)visit * tree_cap * kInkKVDim, b.tk, (size_t)M * kInkKVDim * 4, cudaMemcpyDeviceToDevice, st1);
            cudaMemcpyAsync(d1_stage_v + (size_t)visit * tree_cap * kInkKVDim, b.tv, (size_t)M * kInkKVDim * 4, cudaMemcpyDeviceToDevice, st1);
        } else {
            k_ra_import_kv<<<M, 256, 0, st1>>>(d1_req + Lm.kv, Lm.kv_row, b.pos, b.slots, kv1[visit], M);
        }
        ink_attention(b, kv1[visit], d1_rel_proj[ink_visit_phys(visit)], mode, M, slot0, p0, false);
        if (check_on() && M <= 32) {   // fork debug (INK_RA_CHECK=1, eager only): new kernel vs the split-K kernel
            std::vector<float> fast((size_t)M * kInkQDim), ref((size_t)M * kInkQDim);
            cudaMemcpyAsync(fast.data(), b.out, fast.size() * 4, cudaMemcpyDeviceToHost, st1);
            InkAttnBufs bref = b;
            bref.mq_po = nullptr;
            ink_attention(bref, kv1[visit], d1_rel_proj[ink_visit_phys(visit)], mode, M, slot0, p0, false);
            cudaMemcpyAsync(ref.data(), b.out, ref.size() * 4, cudaMemcpyDeviceToHost, st1);
            cudaStreamSynchronize(st1);
            double md = 0, mr = 0;
            for (size_t i = 0; i < ref.size(); ++i) { md = std::max(md, (double)std::fabs(fast[i] - ref[i])); mr = std::max(mr, (double)std::fabs(ref[i])); }
            static int shown = 0;
            if (M <= 16 && shown++ < 10) std::fprintf(stderr, "[ra-check] visit %d mode %d M %d: max |diff| %.3g (max |ref| %.3g)%c", visit, mode, M, md, mr, 10);
        }
        cudaMemcpyAsync(h_ans[buf], b.out, (size_t)M * kInkQDim * 4, cudaMemcpyDeviceToHost, st1);
        k_ra_ack<<<1, 32, 0, st1>>>(d1_bell, d1_ack_count);
    }

    // 1 ms keep-warm kernel, relaunched once the previous one has finished
    void warm() {
        const cudaError_t w = cudaEventQuery(warm_done);
        if (w == cudaSuccess) {
            k_ra_keep_warm<<<1, 1, 0, warm1>>>(1000000ull);
            cudaEventRecord(warm_done, warm1);
            cudaStreamQuery(warm1);
        } else if (w != cudaErrorNotReady) {
            cudaGetLastError();
        }
    }

    bool launch(int buf, int visit, int mode, int M, int slot0, int p0) {
        if (visit < 0 || visit >= kInkVisits || M <= 0 || M > max_batch || (!kv1[visit].k_f8 && !kv1[visit].k_f16)) return false;
        // prefill (tensor-core path, cuBLAS) runs eagerly; decode / tree / batched requests replay a cached graph
        if ((mode == 0 && M >= 128) || check_on() || buf != 0) { enqueue(buf, visit, mode, M, slot0, p0); return cudaGetLastError() == cudaSuccess; }
        for (auto& g : graphs) {
            if (g.first.visit == visit && g.first.mode == mode && g.first.M == M) return cudaGraphLaunch(g.second, st1) == cudaSuccess;
        }
        cudaGraph_t gr = nullptr;
        cudaGraphExec_t ex = nullptr;
        if (cudaStreamBeginCapture(st1, cudaStreamCaptureModeThreadLocal) != cudaSuccess) return false;
        enqueue(buf, visit, mode, M, slot0, p0);
        if (cudaStreamEndCapture(st1, &gr) != cudaSuccess || cudaGraphInstantiate(&ex, gr, nullptr, nullptr, 0) != cudaSuccess) return false;
        cudaGraphDestroy(gr);
        graphs.push_back({GraphKey{visit, mode, M}, ex});
        return cudaGraphLaunch(ex, st1) == cudaSuccess;
    }

    void shutdown() {
        if (worker.joinable()) { stop.store(true); worker.join(); }
        if (!h_bell) return;
        cudaSetDevice(gpu_kv);
        cudaStreamSynchronize(st1);
        for (auto& g : graphs) cudaGraphExecDestroy(g.second);
        graphs.clear();
        for (void* p : d1_allocs) cudaFree(p);
        d1_allocs.clear();
        if (cublas1) cublasDestroy(cublas1);
        if (st1) cudaStreamDestroy(st1);
        if (warm1) { cudaStreamSynchronize(warm1); cudaStreamDestroy(warm1); }
        if (warm_done) cudaEventDestroy(warm_done);
        cudaSetDevice(gpu_main);
        cudaFree(d_seq); cudaFree(d_cur_seq);
        cudaFreeHost((void*)h_bell);
        for (int b = 0; b < 2; ++b) { cudaFreeHost(h_req[b]); cudaFreeHost(h_ans[b]); }
        h_bell = nullptr;
    }
    ~InkRemoteAttn() { shutdown(); }
};

}  // namespace strata::ink
