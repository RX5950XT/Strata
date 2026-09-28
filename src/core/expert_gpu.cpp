#include "strata/core/expert_gpu.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <immintrin.h>
#include <stdexcept>
#include <thread>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <timeapi.h>
#pragma comment(lib, "winmm.lib")

namespace strata::kernels {
bool keep_warm(uint64_t ns, void* stream);   // src/kernels/cuda/keep_warm.cu
}
#endif

namespace strata::core {
namespace {
constexpr uint32_t kMagic = 0x55504745;   // "EGPU"
constexpr int kCap = 128, kMaxTok = 8, kWidth = 2560, kCandCap = 16384, kSwapCap = 256, kStage = 24;
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

// The job's tables and rows.  The worker page-locks this and moves its head (tables + the used rows of x) to VRAM
// with ONE H2D copy and the written rows back with ONE D2H copy: kernels reading the tables over PCIe (and writing
// 4-byte results) made ~40K tiny PCIe transactions a layer, 0.46-0.71 ms on the GPU for a few experts.
struct EgpuIo {
    unsigned long long ptr[kCap];            ///< the worker fills these (slot or staging addresses)
    int32_t start[kCap + 1], dst[kCap], tok[kCap], count, count2;
    // Fixed graph arguments: the PCIe table starts at a different g1 every job.
    unsigned long long ptr2[kStage];
    int32_t start2[kStage + 1];
    alignas(256) float x[kMaxTok * kWidth];
    alignas(256) float out[kCap * kWidth];
};

// Shared memory between the engine and its worker (a pagefile-backed section, zeroed at creation).  Plain fields
// are written by one side before a release store of a counter and read by the other after the acquire load.
struct EgpuChannel {
    uint32_t magic;
    std::atomic<int32_t> phase;   ///< 0 started, 1 VRAM measured, 2 candidates posted, 3 ready, -1 failed
    char error[512], name[128];
    uint64_t vram_free, vram_total, pinned;
    int32_t n_cand, slots;
    struct Cand { int32_t layer, expert; } cand[kCandCap];   ///< rank order; the worker holds the first `slots`
    std::atomic<uint32_t> req, ack, sleeping, stop, swap_req, swap_ack, active;
    std::atomic<uint32_t> pf_req, pf_ack;
    int32_t pf_status;
    // the job: groups [0, groups) are resident (item = slot), [groups, groups + groups2) are read over PCIe
    // (item = expert id of `layer`)
    int32_t layer, nt, k, groups, groups2, entries, lo, hi, item[kCap];
    int32_t status;
    float gpu_ms;
    double submit_ms;
    int32_t swap_n, swap_slot[kSwapCap], swap_layer[kSwapCap], swap_expert[kSwapCap];
    alignas(4096) EgpuIo io;
};

#ifdef _WIN32
namespace {
// PF_PB experts per copy batch, two batch areas in the decode staging (kStage = 24 blobs): the copy of batch n+1
// overlaps the compute of batch n.
constexpr int PF_K = 10, PF_NE = 512, PF_RMAX = 4096, PF_PB = 12, PF_SB = 2;
struct alignas(256) PfHeader { int32_t layer, tokens, mask[PF_NE]; };
struct PfView {
    PfHeader* head;
    uint16_t* x;
    int32_t* ids;
    float *w, *out;
    PfView(void* base, int cap) : head((PfHeader*) base) {
        x = (uint16_t*) ((uint8_t*) base + sizeof(PfHeader));
        ids = (int32_t*) (x + (size_t) cap * kWidth);
        w = (float*) (ids + (size_t) cap * PF_K);
        out = w + (size_t) cap * PF_K;
    }
};
size_t pf_bytes(int cap) {
    return (sizeof(PfHeader) + (size_t) cap * (6 * kWidth + 8 * PF_K) + 4095) & ~(size_t) 4095;
}
std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }

void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

bool native_format_ok(const strata::kernels::cpu::ExpertLayout& layout, std::string& err) {
    for (const auto& f : layout.fmt) {
        const bool gu = f.gu_type == 16 || f.gu_type == 17 || f.gu_type == 18 || f.gu_type == 21 || f.gu_type == 22 ||
                        f.gu_type == 23 || f.gu_type == 29 || f.gu_type == 42;
        const bool down = f.d_type == 20 || f.d_type == 23 || f.d_type == 42;
        if (!gu || !down || f.n_embd != kWidth || f.n_ff != strata::kernels::cpu::FF) {
            err = "unsupported native expert geometry or format";
            return false;
        }
    }
    return true;
}

// ================================ THE WORKER PROCESS ================================
class Worker {
public:
    int run(int argc, char** argv);
    ~Worker() { for (const auto& g : graphs_) if (g.exec) cudaGraphExecDestroy(g.exec); }

private:
    EgpuChannel* ch_ = nullptr;
    const uint8_t* arena_ = nullptr;
    HANDLE wake_ = nullptr;
    ExpertCache cache_;
    cudaStream_t stream_ = nullptr, copy_ = nullptr, warm_ = nullptr, swap_ = nullptr;
    cudaEvent_t start_ = nullptr, done_ = nullptr, copied_ = nullptr, warm_done_ = nullptr;
    EgpuIo* dev_ = nullptr;
    void* scratch_ = nullptr;
    uint8_t* quant_ = nullptr;
    float* scales_ = nullptr;
    uint8_t* stage_[kStage] = {};
    struct Graph { int gu, down, nt, k; cudaGraphExec_t exec = nullptr; };
    std::vector<Graph> graphs_;   // a null exec remembers a failed build; keep using the direct path
    int pf_cap_ = 0;
    void *pf_host_ = nullptr, *pf_dev_ = nullptr;
    uint16_t *pf_xs_ = nullptr, *pf_h_ = nullptr, *pf_gu_w_ = nullptr, *pf_d_w_ = nullptr;
    float *pf_gu_ = nullptr, *pf_dm_ = nullptr;
    int32_t* pf_src_ = nullptr;          ///< per row: its token
    float* pf_rw_ = nullptr;             ///< per row: its router weight
    cudaEvent_t pf_used_[2] = {}, pf_copied_[2] = {};
    strata::prefill::Gemm pf_gemm_;
    int fail(const std::string& why) {
        std::snprintf(ch_->error, sizeof ch_->error, "%s", why.c_str());
        ch_->phase.store(-1, std::memory_order_release);
        return 1;
    }
    bool wait_phase(int want);
    void warm();
    cudaError_t submit();
    void swaps();
    void prefill_init(const char* name, int cap);
    cudaError_t prefill_submit();
};

bool Worker::wait_phase(int want) {
    for (int i = 0; i < 120000; ++i) {
        if (ch_->phase.load(std::memory_order_acquire) >= want) return true;
        if (ch_->stop.load(std::memory_order_acquire)) return false;
        Sleep(1);
    }
    return false;
}

int Worker::run(int argc, char** argv) {
    using namespace strata::kernels;
    if (argc != 10 && argc != 11) return 2;
    HANDLE map = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, widen(argv[2]).c_str());
    if (!map) return 3;
    ch_ = (EgpuChannel*) MapViewOfFile(map, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(EgpuChannel));
    if (!ch_ || ch_->magic != kMagic) return 3;
    wake_ = OpenEventW(SYNCHRONIZE, FALSE, (widen(argv[2]) + L"-wake").c_str());
    const uint64_t arena_bytes = std::strtoull(argv[4], nullptr, 10), pin_bytes = std::strtoull(argv[9], nullptr, 10);
    const int64_t n_layers = std::atoll(argv[6]), n_expert = std::atoll(argv[7]);
    const uint64_t reserve = (uint64_t) std::atoll(argv[8]) << 20;
    int pf_cap = 0;
    if (argc == 11) {
        char* end = nullptr;
        const long long cap = std::strtoll(argv[10], &end, 10);
        if (end == argv[10] || *end || cap < 0 || cap > INT32_MAX / PF_K)
            return fail("invalid prefill chunk");
        pf_cap = (int) cap;
    }
    std::string err;
    const char* stage = "start CUDA";
    try {
        checked(cudaSetDevice(0));
        cudaDeviceProp prop{};
        checked(cudaGetDeviceProperties(&prop, 0));
        if (prop.major < 8) return fail("compute capability must be at least 8.0");
        std::snprintf(ch_->name, sizeof ch_->name, "%s", prop.name);
        stage = "load the expert layout";
        if (!cpu::expert_layout_load(argv[5], n_layers, n_expert, err)) return fail(err);
        const auto& layout = cpu::expert_layout();
        if (layout.native && !native_format_ok(layout, err)) return fail(err);
        stage = "map the expert arena";
        HANDLE amap = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, widen(argv[3]).c_str());
        if (!amap) return fail("cannot open the arena section (error " + std::to_string(GetLastError()) + ")");
        arena_ = (const uint8_t*) MapViewOfFile(amap, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, (SIZE_T) arena_bytes);
        if (!arena_) return fail("cannot map the arena section (error " + std::to_string(GetLastError()) + ")");
        stage = "allocate streams and buffers";
        checked(cudaHostRegister(&ch_->io, sizeof(EgpuIo), cudaHostRegisterDefault));
        int least = 0, greatest = 0;
        checked(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        checked(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatest));
        checked(cudaStreamCreateWithPriority(&copy_, cudaStreamNonBlocking, greatest));
        checked(cudaStreamCreateWithPriority(&warm_, cudaStreamNonBlocking, least));
        checked(cudaStreamCreateWithPriority(&swap_, cudaStreamNonBlocking, least));
        checked(cudaEventCreate(&start_));
        checked(cudaEventCreate(&done_));
        checked(cudaEventCreateWithFlags(&copied_, cudaEventDisableTiming));
        checked(cudaEventCreateWithFlags(&warm_done_, cudaEventDisableTiming));
        checked(cudaMalloc((void**) &dev_, sizeof(EgpuIo)));
        checked(cudaMalloc(&scratch_, std::max(moe_hit_grouped_scratch_bytes(kCap, kWidth, cpu::FF),
                                               (uint64_t) native_expert_scratch_bytes(kCap, cpu::FF))));
        checked(cudaMalloc(&quant_, kMaxTok * (kWidth / 32) * 36));
        checked(cudaMalloc(&scales_, kMaxTok * (kWidth / 32) * sizeof(float)));
        const size_t stage_bytes = ((size_t) layout.max_blob + 255) & ~(size_t) 255;
        checked(cudaMalloc((void**) &stage_[0], stage_bytes * kStage));
        for (int q = 1; q < kStage; ++q) stage_[q] = stage_[0] + (size_t) q * stage_bytes;
        if (pf_cap > 0) {
            stage = "allocate prefill buffers";
            if (n_expert != PF_NE) return fail("prefill requires 512 experts");
            prefill_init(argv[2], pf_cap);
        }
        size_t free = 0, total = 0;
        checked(cudaMemGetInfo(&free, &total));
        ch_->vram_free = free;
        ch_->vram_total = total;
        ch_->phase.store(1, std::memory_order_release);
        if (!wait_phase(2)) return 0;

        // The engine's candidates, in rank order, as many as fit above the reserve.  Under WDDM a cudaMalloc takes
        // its memory when it is first written, so the slots are zeroed and THEN the reserve is checked; a shortfall
        // (the image encoder sharing the card, the desktop) drops that much from the end and tries again.
        stage = "allocate the slots";
        const uint64_t room = free > reserve ? (uint64_t) free - reserve : 0;
        std::vector<int64_t> sizes;
        uint64_t used = 0;
        for (int i = 0; i < ch_->n_cand && i < kCandCap; ++i) {
            const uint64_t bytes = layout.blob_bytes(ch_->cand[i].layer), step = (bytes + 255) & ~255ull;
            if (used + step > room) break;
            used += step;
            sizes.push_back((int64_t) bytes);
        }
        if (sizes.empty()) return fail("no slots fit after the reserve");
        for (int attempt = 0;; ++attempt) {
            if (!cache_.open_sized(sizes, n_layers, n_expert, err)) return fail(err);
            checked(cudaMemset(cache_.device_slot(0), 0, (size_t) cache_.bytes()));
            checked(cudaDeviceSynchronize());
            checked(cudaMemGetInfo(&free, &total));
            if (free >= reserve) break;
            uint64_t drop = reserve - free + (64ull << 20), dropped = 0;
            while (!sizes.empty() && dropped < drop) {
                dropped += ((uint64_t) sizes.back() + 255) & ~255ull;
                sizes.pop_back();
            }
            cache_.close();
            if (sizes.empty() || attempt == 3)
                return fail("only " + std::to_string(free >> 20) + " MiB free once the slots are written");
        }
        ch_->vram_free = free;
        // This process's share of the machine's page-lock budget, whole layers from the start of the arena: its
        // PCIe reads and its swaps DMA straight from here.  A layer past it is copied (swaps) or left to GPU0/CPU.
        stage = "pin the arena";
        uint64_t pinned = 0;
        for (int64_t l = 0; l < n_layers; ++l) {
            const uint64_t off = layout.layer_offset(l), n = layout.blob_bytes(l) * (uint64_t) n_expert;
            if (off + n > pin_bytes || off + n > arena_bytes) break;
            if (cudaHostRegister((void*) (arena_ + off), (size_t) n, cudaHostRegisterDefault) != cudaSuccess) {
                cudaGetLastError();
                break;
            }
            pinned = off + n;
        }
        ch_->pinned = pinned;
        stage = "fill the slots";
        for (size_t i = 0; i < sizes.size(); ++i) {
            const auto& c = ch_->cand[i];
            const int32_t slot = cache_.admit(c.layer, c.expert);
            if (!cache_.fill_slot_blocking(slot, arena_ + layout.blob_offset(c.layer, c.expert), err, sizes[i])) return fail(err);
        }
        if (!cache_.verify_slot(0, arena_ + layout.blob_offset(ch_->cand[0].layer, ch_->cand[0].expert), err, sizes[0]))
            return fail(err);
        // Every expert format once (and one PCIe read), so a kernel that cannot run on this card fails here and not
        // at the first layer.
        stage = "kernel self-test";
        std::vector<int> seen;
        for (size_t i = 0; i <= sizes.size(); ++i) {
            const bool pcie = i == sizes.size();
            if (pcie && pinned == 0) break;
            const int32_t layer = pcie ? 0 : ch_->cand[i].layer;
            const int mark = layout.native ? layout.fmt[(size_t) layer].gu_type * 64 + layout.fmt[(size_t) layer].d_type : 0;
            if (!pcie && std::find(seen.begin(), seen.end(), mark) != seen.end()) continue;
            seen.push_back(mark);
            std::memset(ch_->io.x, 0, sizeof(float) * kWidth);
            ch_->layer = layer; ch_->nt = 1; ch_->k = 1; ch_->entries = 1; ch_->lo = ch_->hi = 0;
            ch_->groups = pcie ? 0 : 1; ch_->groups2 = pcie ? 1 : 0; ch_->item[0] = pcie ? 0 : (int32_t) i;
            ch_->io.start[0] = 0; ch_->io.start[1] = 1; ch_->io.dst[0] = 0; ch_->io.tok[0] = 0;
            ch_->io.count = ch_->groups; ch_->io.count2 = ch_->groups2;
            const cudaError_t e = submit();
            if (e != cudaSuccess) return fail(std::string("kernel self-test: ") + cudaGetErrorString(e));
        }
        ch_->slots = (int32_t) sizes.size();
        ch_->phase.store(3, std::memory_order_release);
    } catch (const std::exception& error) {
        return fail(std::string(stage) + ": " + error.what());
    }
    // THE LOOP.  Spin while layers keep coming (one every ~1 ms), sleep on the event once they stop.  While a
    // request runs (and for a second after), 1 ms keep-warm kernels go out every millisecond even with no layer to
    // compute: the prompt leaves this GPU idle for seconds, and from P8 it took ~1.5 s of the reply to get its clocks
    // and its PCIe link back (measured: 88 us per expert group at the start of a reply, 7 us once warm).  15 ms
    // apart was not enough: the card fell back to P8 after ~7 s.
    timeBeginPeriod(1);
    std::thread swapper(&Worker::swaps, this);
    uint32_t served = ch_->ack.load(std::memory_order_relaxed);
    double last = now_ms();
    while (!ch_->stop.load(std::memory_order_acquire)) {
        const uint32_t pf_want = ch_->pf_req.load(std::memory_order_acquire);
        if (pf_want != ch_->pf_ack.load(std::memory_order_relaxed)) {
            ch_->pf_status = (int32_t) prefill_submit();
            ch_->pf_ack.store(pf_want, std::memory_order_release);
            last = now_ms();
            continue;
        }
        const uint32_t want = ch_->req.load(std::memory_order_acquire);
        if (want == served) {
            const double t = now_ms();
            if (t - last < 2.0) { _mm_pause(); continue; }
            if (ch_->active.load(std::memory_order_acquire)) last = t - 2.0;
            const bool hot = t - last < 1000.0;
            if (hot) warm();
            ch_->sleeping.store(1, std::memory_order_seq_cst);
            if (ch_->req.load(std::memory_order_seq_cst) == served &&
                ch_->pf_req.load(std::memory_order_seq_cst) == ch_->pf_ack.load() && !ch_->stop.load())
                WaitForSingleObject(wake_, hot ? 1 : 100);
            ch_->sleeping.store(0, std::memory_order_seq_cst);
            continue;
        }
        ch_->status = (int32_t) submit();
        served = want;
        last = now_ms();
        ch_->ack.store(served, std::memory_order_release);
    }
    swapper.join();
    return 0;
}

void Worker::prefill_init(const char* name, int cap) {
    pf_cap_ = cap;
    HANDLE map = OpenFileMappingW(FILE_MAP_READ | FILE_MAP_WRITE, FALSE, (widen(name) + L"-prefill").c_str());
    if (!map) throw std::runtime_error("cannot open prefill section");
    pf_host_ = MapViewOfFile(map, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, pf_bytes(cap));
    CloseHandle(map);
    if (!pf_host_) throw std::runtime_error("cannot map prefill section");
    checked(cudaHostRegister(pf_host_, pf_bytes(cap), cudaHostRegisterDefault));
    auto alloc = [&](auto** p, size_t count) {
        checked(cudaMalloc((void**) p, count * sizeof(**p)));
        checked(cudaMemset(*p, 0, count * sizeof(**p)));
    };
    checked(cudaMalloc(&pf_dev_, pf_bytes(cap)));
    checked(cudaMemset(pf_dev_, 0, pf_bytes(cap)));
    alloc(&pf_src_, (size_t) cap * PF_K);
    alloc(&pf_rw_, (size_t) cap * PF_K);
    alloc(&pf_xs_, (size_t) PF_RMAX * kWidth);
    alloc(&pf_gu_, (size_t) PF_RMAX * 1280);
    alloc(&pf_h_, (size_t) PF_RMAX * 640);
    alloc(&pf_dm_, (size_t) PF_RMAX * kWidth);
    alloc(&pf_gu_w_, (size_t) PF_SB * 1280 * kWidth);
    alloc(&pf_d_w_, (size_t) PF_SB * kWidth * 640);
    for (int a = 0; a < 2; ++a) {
        checked(cudaEventCreateWithFlags(&pf_used_[a], cudaEventDisableTiming));
        checked(cudaEventCreateWithFlags(&pf_copied_[a], cudaEventDisableTiming));
    }
    std::string err;
    if (!pf_gemm_.init(stream_, 0, err)) throw std::runtime_error(err);
    // WDDM commits on first touch; include the workspace before measuring room for slots.
    pf_gemm_.f16(pf_xs_, pf_gu_w_, pf_gu_, 1, 1280, kWidth);
    checked(cudaDeviceSynchronize());
}

cudaError_t Worker::prefill_submit() {
    using namespace strata::prefill;
    using namespace strata::kernels;
    if (!pf_host_ || !pf_dev_) return cudaErrorInvalidValue;
    try {
        const PfView h(pf_host_, pf_cap_), d(pf_dev_, pf_cap_);
        const int T = h.head->tokens, l = h.head->layer;
        const auto& lay = cpu::expert_layout();
        if (T <= 0 || T > pf_cap_ || l < 0 || l >= lay.n_layers) return cudaErrorInvalidValue;
        const size_t pairs = (size_t) T * PF_K, bb = (size_t) lay.blob_bytes(l);
        std::vector<int32_t> cnt(PF_NE, 0), off(PF_NE, 0), order, src;
        std::vector<float> rw;
        for (size_t i = 0; i < pairs; ++i) {
            const int e = h.ids[i];
            if (e < 0 || e >= PF_NE) return cudaErrorInvalidValue;
            if (h.head->mask[e] != -1) ++cnt[e];
        }
        int rows = 0;
        for (int e = 0; e < PF_NE; ++e) {
            const int s = h.head->mask[e];
            if (s < -2 || s >= ch_->slots) return cudaErrorInvalidValue;
            if (s == -2 && lay.layer_offset(l) + bb * PF_NE > ch_->pinned) return cudaErrorInvalidValue;
            off[e] = rows;
            rows += cnt[e];
            if (cnt[e]) order.push_back(e);
        }
        src.resize(rows);
        rw.resize(rows);
        auto fill = off;
        for (size_t i = 0; i < pairs; ++i) {
            const int e = h.ids[i];
            if (h.head->mask[e] == -1) continue;
            const int r = fill[e]++;
            src[r] = (int32_t) (i / PF_K);
            rw[r] = h.w[i];
        }
        checked(cudaMemcpyAsync(d.x, h.x, (size_t) T * kWidth * 2, cudaMemcpyHostToDevice, stream_));
        checked(cudaMemcpyAsync(d.w, h.w, pairs * 4, cudaMemcpyHostToDevice, stream_));
        checked(cudaMemcpyAsync(d.ids, h.ids, pairs * 4, cudaMemcpyHostToDevice, stream_));
        if (rows) {
            checked(cudaMemcpyAsync(pf_src_, src.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, stream_));
            checked(cudaMemcpyAsync(pf_rw_, rw.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, stream_));
        }
        checked(cudaMemsetAsync(d.out, 0, (size_t) T * kWidth * 4, stream_));
        // Staging: the decode staging (decode and prefill are served one at a time by this thread), in two areas.
        IqDequantBatch tables[2];
        auto stage = [&](size_t first, int a) {
            const int nb = (int) std::min((size_t) PF_PB, order.size() - first);
            uint8_t* area = stage_[a * PF_PB];
            if (first >= 2 * (size_t) PF_PB) checked(cudaStreamWaitEvent(copy_, pf_used_[a], 0));
            for (int i = 0; i < nb;) {
                const int e = order[first + i], s = h.head->mask[e];
                if (s >= 0) { tables[a].blob[i++] = cache_.device_slot(s); continue; }
                int end = i + 1;
                while (end < nb && h.head->mask[order[first + end]] == -2 &&
                       order[first + end] == order[first + end - 1] + 1) ++end;
                checked(cudaMemcpyAsync(area + (size_t) i * bb, arena_ + lay.blob_offset(l, e),
                                        (size_t) (end - i) * bb, cudaMemcpyHostToDevice, copy_));
                for (; i < end; ++i) tables[a].blob[i] = area + (size_t) i * bb;
            }
            checked(cudaEventRecord(pf_copied_[a], copy_));
        };
        if (!order.empty()) stage(0, 0);
        for (size_t first = 0; first < order.size(); first += PF_PB) {
            const int nb = (int) std::min((size_t) PF_PB, order.size() - first);
            const int a = (int) (first / PF_PB % 2);
            const IqDequantBatch batch = tables[a];
            checked(cudaStreamWaitEvent(stream_, pf_copied_[a], 0));
            if (first + PF_PB < order.size()) stage(first + PF_PB, a ^ 1);
            for (int s0 = 0; s0 < nb; s0 += PF_SB) {
                const int ns = std::min(PF_SB, nb - s0);
                IqDequantBatch sub;
                for (int i = 0; i < ns; ++i) sub.blob[i] = batch.blob[s0 + i];
                if (lay.native) {
                    const auto& f = lay.fmt[(size_t) l];
                    iq_dequant_gu_f16_batch(f.gu_type, sub, ns, f.up_off, f.n_ff, f.n_embd, pf_gu_w_, stream_);
                    iq_dequant_f16_batch(f.d_type, sub, ns, f.down_off, f.n_embd * f.n_ff, pf_d_w_, stream_);
                } else {
                    for (int i = 0; i < ns; ++i)
                        blob_dequant_f16(sub.blob[i], pf_gu_w_ + (size_t) i * 1280 * kWidth,
                                         pf_d_w_ + (size_t) i * kWidth * 640, stream_);
                }
                if (s0 + ns == nb) checked(cudaEventRecord(pf_used_[a], stream_));
                const int start = off[order[first + s0]];
                const int last = order[first + s0 + ns - 1], end = off[last] + cnt[last];
                for (int begin = start; begin < end; begin += PF_RMAX) {
                    const int nr = std::min(PF_RMAX, end - begin);
                    gather_rows16(d.x, pf_src_ + begin, pf_xs_, nr, kWidth, stream_);
                    auto spans = [&](bool down) {
                        for (int i = 0; i < ns; ++i) {
                            const int e = order[first + s0 + i];
                            const int lo = std::max(begin, off[e]), hi = std::min(begin + nr, off[e] + cnt[e]);
                            if (hi <= lo) continue;
                            const size_t r = lo - begin;
                            if (!down) pf_gemm_.f16(pf_xs_ + r * kWidth, pf_gu_w_ + (size_t) i * 1280 * kWidth,
                                                    pf_gu_ + r * 1280, hi - lo, 1280, kWidth);
                            else pf_gemm_.f16(pf_h_ + r * 640, pf_d_w_ + (size_t) i * kWidth * 640,
                                               pf_dm_ + r * kWidth, hi - lo, kWidth, 640);
                        }
                    };
                    spans(false);
                    swiglu_interleaved(pf_gu_, pf_h_, nr, stream_);
                    spans(true);
                    for (int i = 0; i < ns; ++i) {   // one expert per launch: its tokens are distinct
                        const int e = order[first + s0 + i];
                        const int lo = std::max(begin, off[e]), hi = std::min(begin + nr, off[e] + cnt[e]);
                        if (hi > lo) moe_partial(pf_dm_ + (size_t) (lo - begin) * kWidth, pf_src_ + lo, pf_rw_ + lo, d.out,
                                                 hi - lo, stream_);
                    }
                }
            }
        }
        checked(cudaMemcpyAsync(h.out, d.out, (size_t) T * kWidth * 4, cudaMemcpyDeviceToHost, stream_));
        checked(cudaStreamSynchronize(stream_));
        return cudaSuccess;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "expert GPU prefill: %s\n", error.what());
        cudaStreamSynchronize(copy_);
        cudaStreamSynchronize(stream_);
        return cudaErrorLaunchFailure;
    }
}

// Swaps run on their own thread and stream, off the layer path: a copy from an unpinned layer is synchronous for
// the thread that issues it.
void Worker::swaps() {
    cudaSetDevice(0);
    const auto& layout = strata::kernels::cpu::expert_layout();
    uint32_t done = ch_->swap_ack.load(std::memory_order_relaxed);
    while (!ch_->stop.load(std::memory_order_acquire)) {
        const uint32_t want = ch_->swap_req.load(std::memory_order_acquire);
        if (want == done) { Sleep(1); continue; }
        for (int i = 0; i < ch_->swap_n && i < kSwapCap; ++i) {
            const int32_t layer = ch_->swap_layer[i];
            cudaMemcpyAsync(cache_.device_slot(ch_->swap_slot[i]), arena_ + layout.blob_offset(layer, ch_->swap_expert[i]),
                            (size_t) layout.blob_bytes(layer), cudaMemcpyHostToDevice, swap_);
        }
        cudaStreamSynchronize(swap_);
        cudaGetLastError();
        done = want;
        ch_->swap_ack.store(done, std::memory_order_release);
    }
}

// Keep the card at its working clocks (keep_warm.cu): 1 ms kernels, relaunched once the previous one is done; a
// 200 ms one held the expert kernels behind it (p90 193 ms) under WDDM.
void Worker::warm() {
    const cudaError_t w = cudaEventQuery(warm_done_);
    if (w == cudaSuccess && strata::kernels::keep_warm(1000000ull, warm_)) cudaEventRecord(warm_done_, warm_);
    else if (w != cudaErrorNotReady) cudaGetLastError();
}

// One layer: the PCIe reads start first on their own stream, then the tables and x go over in one copy, the resident
// groups run while the reads land, the PCIe groups after them, and the written rows come back in one copy.
cudaError_t Worker::submit() {
    using namespace strata::kernels;
    const double t0 = now_ms();
    try {
        const auto& layout = cpu::expert_layout();
        warm();
        const int64_t layer = ch_->layer;
        const int g1 = ch_->groups, g2 = std::min(ch_->groups2, kStage), nt = ch_->nt, entries = ch_->entries;
        const size_t bb = (size_t) layout.blob_bytes(layer);
        for (int q = 0; q < g2; ++q) {
            checked(cudaMemcpyAsync(stage_[q], arena_ + layout.blob_offset(layer, ch_->item[g1 + q]), bb,
                                    cudaMemcpyHostToDevice, copy_));
            ch_->io.ptr[g1 + q] = (unsigned long long) stage_[q];
        }
        checked(cudaEventRecord(copied_, copy_));
        for (int g = 0; g < g1; ++g) ch_->io.ptr[g] = (unsigned long long) cache_.device_slot(ch_->item[g]);
        std::copy_n(ch_->io.ptr + g1, g2, ch_->io.ptr2);
        std::copy_n(ch_->io.start + g1, g2 + 1, ch_->io.start2);
        const int cap = std::min(nt * std::max(1, (int) ch_->k), kCap);   // every dst row is < nt * k
        auto enqueue = [&](bool graph) {
            checked(cudaMemcpyAsync(dev_, &ch_->io, offsetof(EgpuIo, x) + (size_t) nt * kWidth * sizeof(float),
                                    cudaMemcpyHostToDevice, stream_));
            auto run = [&](const unsigned long long* ptr, const int32_t* start, int n, const int32_t* count) {
                const int ne = graph ? cap : entries;   // both calls must carve scratch at the same offsets
                if (layout.native) {
                    const auto& f = layout.fmt[(size_t) layer];
                    native_expert_grouped(native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff), ptr,
                                          start, count, dev_->dst, dev_->tok, n, ne, quant_, scratch_,
                                          dev_->out, stream_, true);
                } else {
                    moe_grouped_s2(ptr, start, count, dev_->dst, dev_->tok, n, ne, quant_, scales_,
                                   scratch_, dev_->out, stream_, true);
                }
            };
            if (layout.native) quantize_q8_1_rows(dev_->x, nt, kWidth, quant_, stream_, true);
            else quantize_q8_0_scaled(dev_->x, quant_, scales_, (int64_t) nt * kWidth, stream_, true);
            if (graph || g1) run(dev_->ptr, dev_->start, graph ? cap : g1, &dev_->count);
            if (graph || g2) {
                checked(cudaStreamWaitEvent(stream_, copied_, graph ? cudaEventWaitExternal : 0));
                run(graph ? dev_->ptr2 : dev_->ptr + g1, graph ? dev_->start2 : dev_->start + g1,
                    graph ? cap : g2, &dev_->count2);
            }
            const size_t lo = graph ? 0 : (size_t) ch_->lo * kWidth;
            const size_t rows = (size_t) (graph ? cap : ch_->hi - ch_->lo + 1) * kWidth;
            checked(cudaMemcpyAsync(ch_->io.out + lo, dev_->out + lo, rows * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        };
        cudaGraphExec_t exec = nullptr;
        // One graph per (format, nt, k); a job outside those bounds keeps the direct path.
        if (entries <= cap && ch_->hi < cap) {
            const int gu = layout.native ? layout.fmt[(size_t) layer].gu_type : -1;
            const int down = layout.native ? layout.fmt[(size_t) layer].d_type : -1;
            auto it = std::find_if(graphs_.begin(), graphs_.end(), [&](const Graph& g) {
                return g.gu == gu && g.down == down && g.nt == nt && g.k == ch_->k;
            });
            if (it == graphs_.end()) {
                graphs_.push_back({gu, down, nt, ch_->k});
                it = graphs_.end() - 1;
                cudaGraph_t graph = nullptr;
                bool capturing = false;
                try {
                    // The swap thread is independent of this capture.
                    checked(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
                    capturing = true;
                    enqueue(true);
                    const cudaError_t end = cudaStreamEndCapture(stream_, &graph);
                    capturing = false;
                    checked(end);
                    checked(cudaGraphInstantiate(&it->exec, graph, 0));
                } catch (const std::exception& error) {
                    if (capturing) cudaStreamEndCapture(stream_, &graph);
                    if (it->exec) cudaGraphExecDestroy(it->exec);
                    it->exec = nullptr;
                    cudaGetLastError();
                    std::fprintf(stderr, "expert GPU graph (%d,%d,%d): %s; using direct path\n",
                                 gu, down, nt, error.what());
                }
                if (graph) cudaGraphDestroy(graph);
            }
            exec = it->exec;
        }
        checked(cudaEventRecord(start_, stream_));
        if (exec) checked(cudaGraphLaunch(exec, stream_));
        else enqueue(false);
        checked(cudaEventRecord(done_, stream_));
    } catch (const std::exception&) {
        const cudaError_t e = cudaGetLastError();
        cudaStreamSynchronize(stream_);
        cudaStreamSynchronize(copy_);
        return e != cudaSuccess ? e : cudaErrorLaunchFailure;
    }
    ch_->submit_ms = now_ms() - t0;
    cudaError_t status;
    do { status = cudaEventQuery(done_); } while (status == cudaErrorNotReady);
    float gpu = 0;
    if (status == cudaSuccess) cudaEventElapsedTime(&gpu, start_, done_);
    else cudaStreamSynchronize(stream_);
    ch_->gpu_ms = gpu;
    return status;
}
}  // namespace

int expert_gpu_worker_main(int argc, char** argv) {
    Worker w;
    return w.run(argc, argv);
}

// ================================ THE ENGINE'S SIDE ================================
ExpertGpu::~ExpertGpu() { close(); }

bool ExpertGpu::alive() const { return proc_ && WaitForSingleObject((HANDLE) proc_, 0) == WAIT_TIMEOUT; }

void ExpertGpu::close() {
    if (ch_) {
        ch_->stop.store(1, std::memory_order_release);
        if (wake_) SetEvent((HANDLE) wake_);
    }
    if (proc_) {
        if (WaitForSingleObject((HANDLE) proc_, 5000) != WAIT_OBJECT_0) TerminateProcess((HANDLE) proc_, 1);
        CloseHandle((HANDLE) proc_);
    }
    if (job_) CloseHandle((HANDLE) job_);
    if (wake_) CloseHandle((HANDLE) wake_);
    if (pf_registered_) cudaHostUnregister(pf_);
    if (pf_) UnmapViewOfFile(pf_);
    if (pf_map_) CloseHandle((HANDLE) pf_map_);
    pf_ = pf_map_ = nullptr;
    pf_cap_ = 0;
    pf_registered_ = pf_pending_ = false;
    pf_req_ = 0;
    if (ch_) UnmapViewOfFile(ch_);
    if (map_) CloseHandle((HANDLE) map_);
    ch_ = nullptr;
    map_ = proc_ = job_ = wake_ = nullptr;
    enabled_ = pending_ = swapping_ = false;
}

bool ExpertGpu::open(const Options& opt, const ExpertCache& primary, const std::string& arena_name, uint64_t arena_bytes,
                     uint64_t main_pinned, const std::string& pack_dir, int64_t n_layers, int64_t n_expert,
                     const std::vector<std::pair<int32_t, int32_t>>& profile, std::string& err) {
    close();
    auto bail = [&](const std::string& why) { err = why; close(); return false; };
    if (profile.empty()) return bail("an expert profile is required");
    if (opt.reserve_mib < 0 || opt.slots < -1 || opt.prefill_chunk < 0 || opt.prefill_chunk > INT32_MAX / PF_K)
        return bail("invalid reserve, slot cap or prefill chunk");
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    pcie_num_ = std::max(0, std::min(256, opt.pcie_num));
    // the channel, the wake event, and a job object that takes the worker down with this process
    const std::string name = "Local\\strata-egpu-" + std::to_string(GetCurrentProcessId());
    map_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, 0, (DWORD) sizeof(EgpuChannel),
                              widen(name).c_str());
    if (!map_) return bail("cannot create the worker channel");
    ch_ = (EgpuChannel*) MapViewOfFile((HANDLE) map_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, sizeof(EgpuChannel));
    if (!ch_) return bail("cannot map the worker channel");
    ch_->magic = kMagic;
    if (opt.prefill_chunk > 0) {
        if (n_expert != PF_NE) return bail("prefill requires 512 experts");
        pf_cap_ = opt.prefill_chunk;
        const uint64_t bytes = pf_bytes(pf_cap_);
        pf_map_ = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, (DWORD) (bytes >> 32),
                                     (DWORD) bytes, widen(name + "-prefill").c_str());
        if (!pf_map_) return bail("cannot create prefill section");
        pf_ = MapViewOfFile((HANDLE) pf_map_, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, (SIZE_T) bytes);
        if (!pf_) return bail("cannot map prefill section");
        const cudaError_t rc = cudaHostRegister(pf_, (size_t) bytes, cudaHostRegisterDefault);
        if (rc != cudaSuccess) return bail(std::string("cannot pin prefill section: ") + cudaGetErrorString(rc));
        pf_registered_ = true;
    }
    wake_ = CreateEventW(nullptr, FALSE, FALSE, widen(name + "-wake").c_str());
    job_ = CreateJobObjectW(nullptr, nullptr);
    if (!wake_ || !job_) return bail("cannot create the worker's event or job object");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION kill{};
    kill.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject((HANDLE) job_, JobObjectExtendedLimitInformation, &kill, sizeof kill);
    // The page-lock budget: ~78 GiB in total on a 96 GB PC (docs/DUAL-GPU.md), so 3/4 of RAM less what this
    // process pinned, with 2 GiB of margin.
    uint64_t pin = 0;
    if (opt.pin_gib >= 0) {
        pin = (uint64_t) (opt.pin_gib * 1073741824.0);
    } else {
        MEMORYSTATUSEX ms{};
        ms.dwLength = sizeof ms;
        GlobalMemoryStatusEx(&ms);
        const uint64_t budget = ms.ullTotalPhys / 4 * 3;
        pin = budget > main_pinned + (2ull << 30) ? budget - main_pinned - (2ull << 30) : 0;
    }
    wchar_t exe[MAX_PATH];
    if (!GetModuleFileNameW(nullptr, exe, MAX_PATH)) return bail("cannot find strata.exe");
    std::wstring cmd = L"\"" + std::wstring(exe) + L"\" --expert-gpu-worker " + widen(name) + L" " + widen(arena_name) +
                       L" " + std::to_wstring(arena_bytes) + L" \"" + widen(pack_dir) + L"\" " +
                       std::to_wstring(n_layers) + L" " + std::to_wstring(n_expert) + L" " +
                       std::to_wstring(opt.reserve_mib) + L" " + std::to_wstring(pin) + L" " +
                       std::to_wstring(opt.prefill_chunk);
    // The worker sees only its GPU.  Its environment is this one with CUDA_VISIBLE_DEVICES swapped for the call.
    wchar_t old_env[4096];
    const DWORD had = GetEnvironmentVariableW(L"CUDA_VISIBLE_DEVICES", old_env, 4096);
    SetEnvironmentVariableW(L"CUDA_VISIBLE_DEVICES", widen(opt.device).c_str());
    STARTUPINFOW si{};
    si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    const BOOL made = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW | CREATE_SUSPENDED,
                                     nullptr, nullptr, &si, &pi);
    SetEnvironmentVariableW(L"CUDA_VISIBLE_DEVICES", had && had < 4096 ? old_env : nullptr);
    if (!made) return bail("cannot start the worker process (error " + std::to_string(GetLastError()) + ")");
    AssignProcessToJobObject((HANDLE) job_, pi.hProcess);
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    proc_ = pi.hProcess;
    auto wait_phase = [&](int want, double seconds) -> bool {
        const double t0 = now_ms();
        while (ch_->phase.load(std::memory_order_acquire) < want) {
            if (ch_->phase.load(std::memory_order_acquire) < 0) return false;
            if (!alive() || now_ms() - t0 > seconds * 1000.0) return false;
            Sleep(1);
        }
        return true;
    };
    auto why = [&](const char* what) {
        if (ch_->phase.load() < 0) return std::string(ch_->error);
        DWORD code = 0;
        GetExitCodeProcess((HANDLE) proc_, &code);
        return std::string(what) + (alive() ? " (timed out)" : " (worker exited, code " + std::to_string(code) + ")");
    };
    if (!wait_phase(1, 120)) return bail(why("the worker did not start"));
    // The profile's pairs GPU0 did not take, in rank order.
    const auto& layout = strata::kernels::cpu::expert_layout();
    int n = 0;
    for (const auto& pair : profile) {
        if (n >= kCandCap || (opt.slots >= 0 && n >= opt.slots)) break;
        if (primary.slot_of(pair.first, pair.second) >= 0) continue;
        ch_->cand[n++] = {pair.first, pair.second};
    }
    if (n == 0) return bail("the profile has no experts GPU0 does not already hold");
    ch_->n_cand = n;
    ch_->phase.store(2, std::memory_order_release);
    if (!wait_phase(3, 600)) return bail(why("the worker did not get ready"));
    res_.assign((size_t) (n_layers * n_expert), kNotResident);
    slot_layer_.assign((size_t) ch_->slots, 0);
    for (int i = 0; i < ch_->slots; ++i) {
        res_[(size_t) ch_->cand[i].layer * (size_t) n_expert + (size_t) ch_->cand[i].expert] = i;
        slot_layer_[(size_t) i] = ch_->cand[i].layer;
    }
    pinned_ = ch_->pinned;
    name_ = ch_->name;
    req_ = ch_->req.load();
    swap_req_ = ch_->swap_req.load();
    enabled_ = true;
    uint64_t bytes = 0;
    for (int i = 0; i < ch_->slots; ++i) bytes += layout.blob_bytes(ch_->cand[i].layer);
    std::fprintf(stderr, "strata generate: expert GPU: %d slots on %s (worker process), %.2f GiB, %llu MiB free of %llu MiB; "
                         "it pinned %.1f GiB of the arena for its own PCIe reads (share %d/256 of the misses)\n",
                 ch_->slots, name_.c_str(), (double) bytes / 1073741824.0, (unsigned long long) (ch_->vram_free >> 20),
                 (unsigned long long) (ch_->vram_total >> 20), (double) pinned_ / 1073741824.0, pcie_num_);
    return true;
}

void ExpertGpu::set_active(bool on) {
    if (!enabled_) return;
    ch_->active.store(on ? 1 : 0, std::memory_order_release);
    if (on && ch_->sleeping.load()) SetEvent((HANDLE) wake_);
}

bool ExpertGpu::contains(int64_t layer, int32_t expert) const {
    return enabled_ && res_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] >= 0;
}

bool ExpertGpu::layer_pinned(int64_t layer) const {
    if (!enabled_ || layer < 0 || layer >= n_layers_) return false;
    const auto& lay = strata::kernels::cpu::expert_layout();
    return lay.layer_offset(layer) + lay.blob_bytes(layer) * (uint64_t) n_expert_ <= pinned_;
}

bool ExpertGpu::prefill_ready(int64_t tokens) const {
    return enabled_ && pf_ && !pf_pending_ && tokens > 0 && tokens <= pf_cap_;
}
uint16_t* ExpertGpu::prefill_input() { return PfView(pf_, pf_cap_).x; }
float* ExpertGpu::prefill_weights() { return PfView(pf_, pf_cap_).w; }
const float* ExpertGpu::prefill_output() const { return PfView(pf_, pf_cap_).out; }

bool ExpertGpu::prefill_launch(int64_t layer, int tokens, const int32_t* ids, const int32_t* mask) {
    if (!prefill_ready(tokens) || layer < 0 || layer >= n_layers_ || !ids || !mask) return false;
    if (!alive()) { disable("the worker process exited before prefill"); return false; }
    poll_swaps();
    const PfView h(pf_, pf_cap_);
    for (int e = 0; e < PF_NE; ++e) {
        if (mask[e] < -2 || (mask[e] == -2 && !layer_pinned(layer))) return false;
        const int slot = res_[(size_t) layer * n_expert_ + e];
        if (mask[e] >= 0 && slot < 0) return false;
        h.head->mask[e] = mask[e] >= 0 ? slot : mask[e];
    }
    for (int i = 0; i < tokens * PF_K; ++i) if (ids[i] < 0 || ids[i] >= PF_NE) return false;
    h.head->layer = (int32_t) layer;
    h.head->tokens = tokens;
    std::memcpy(h.ids, ids, (size_t) tokens * PF_K * 4);
    pf_pending_ = true;
    ch_->pf_req.store(++pf_req_, std::memory_order_seq_cst);
    if (ch_->sleeping.load(std::memory_order_seq_cst)) SetEvent((HANDLE) wake_);
    return true;
}

bool ExpertGpu::prefill_wait() {
    if (!pf_pending_) return false;
    for (uint32_t i = 1; ch_->pf_ack.load(std::memory_order_acquire) != pf_req_; ++i) {
        _mm_pause();
        if ((i & 0xffff) == 0 && !alive()) {
            pf_pending_ = false;
            disable("the worker process exited during prefill");
            return false;
        }
    }
    pf_pending_ = false;
    if (ch_->pf_status == (int32_t) cudaSuccess) return true;
    disable(cudaGetErrorString((cudaError_t) ch_->pf_status));
    return false;
}

void ExpertGpu::disable(const char* reason) {
    failure_ = reason;
    if (enabled_) std::fprintf(stderr, "strata generate: expert GPU disabled: %s\n", reason);
    enabled_ = false;
}

void ExpertGpu::poll_swaps() {
    if (!swapping_ || ch_->swap_ack.load(std::memory_order_acquire) != swap_req_) return;
    for (const auto& [i, slot] : swap_in_) res_[(size_t) i] = slot;
    swapped += (int64_t) swap_in_.size();
    swap_in_.clear();
    swapping_ = false;
}

bool ExpertGpu::launch(int64_t layer, const float* x, const int32_t* ids, int nt, int k, float* out, int32_t* kind,
                       bool pcie) {
    (void) out;
    if (!enabled_ || nt < 1 || nt > max_tokens || k < 1 || nt * k > cap) return false;
    poll_swaps();
    const auto& layout = strata::kernels::cpu::expert_layout();
    EgpuIo& io = ch_->io;
    const int n = nt * k;
    entries_ = 0;
    int g1 = 0, g2 = 0;
    auto take = [&](int i, int32_t item) {   // group every row of ids[i] still owned by the CPU
        ch_->item[g1 + g2] = item;
        io.start[g1 + g2] = entries_;
        for (int j = i; j < n; ++j) {
            if (kind[j] != -1 || ids[j] != ids[i]) continue;
            io.dst[entries_] = j;
            io.tok[entries_++] = j / k;
            kind[j] = -2;
        }
    };
    const size_t row = (size_t) layer * (size_t) n_expert_;
    for (int i = 0; i < n; ++i)
        if (kind[i] == -1 && res_[row + (size_t) ids[i]] >= 0) { take(i, res_[row + (size_t) ids[i]]); ++g1; }
    // Its share of the remaining misses, read over its own PCIe link (the first ones in routing order; GPU0's
    // share is the last ones of what is left), from the layers it pinned.
    if (pcie && pcie_num_ > 0 && layout.layer_offset(layer) + layout.blob_bytes(layer) * (uint64_t) n_expert_ <= pinned_) {
        int misses = 0;
        for (int i = 0; i < n; ++i) {
            if (kind[i] != -1) continue;
            bool first = true;
            for (int j = 0; j < i && first; ++j) first = ids[j] != ids[i];
            misses += first;
        }
        const int want = std::min(kStage, (misses * pcie_num_ + 128) >> 8);
        for (int i = 0; i < n && g2 < want; ++i)
            if (kind[i] == -1) { take(i, ids[i]); ++g2; }
    }
    if (!entries_) return false;
    io.start[g1 + g2] = entries_;
    io.count = g1;
    io.count2 = g2;
    ch_->layer = (int32_t) layer;
    ch_->nt = nt;
    ch_->k = k;
    ch_->groups = g1;
    ch_->groups2 = g2;
    ch_->entries = entries_;
    ch_->lo = *std::min_element(io.dst, io.dst + entries_);
    ch_->hi = *std::max_element(io.dst, io.dst + entries_);
    std::memcpy(io.x, x, (size_t) nt * width * sizeof(float));
    began_ = now_ms();
    streamed += g2;
    pending_ = true;
    ch_->req.store(++req_, std::memory_order_seq_cst);
    if (ch_->sleeping.load(std::memory_order_seq_cst)) SetEvent((HANDLE) wake_);
    return true;
}

bool ExpertGpu::wait(std::string& err) {
    if (!pending_) return true;
    const double spin = now_ms();
    for (uint32_t i = 1; ch_->ack.load(std::memory_order_acquire) != req_; ++i) {
        _mm_pause();
        if ((i & 0xffff) == 0 && !alive()) {
            pending_ = false;
            err = "the worker process exited";
            return false;
        }
    }
    pending_ = false;
    const double t = now_ms();
    ms_wait += t - spin;
    ms += t - began_;
    ms_launch += ch_->submit_ms;
    if (ch_->status == (int32_t) cudaSuccess) {
        ms_gpu += ch_->gpu_ms;
        hits += entries_;
        ++launches;
        return true;
    }
    err = cudaGetErrorString((cudaError_t) ch_->status);
    return false;
}

void ExpertGpu::finish(ExpertDispatch& dispatch, const float* x, const int32_t* ids, int k, float* out) {
    if (!pending_) return;
    std::string err;
    const EgpuIo& io = ch_->io;
    if (wait(err)) {
        for (int i = 0; i < entries_; ++i) {   // only this tier's rows: the rest of `out` is the CPU's
            const size_t offset = (size_t) io.dst[i] * width;
            std::memcpy(out + offset, io.out + offset, width * sizeof(float));
        }
        return;
    }
    disable(err.c_str());
    // A failure must still supply every accepted row; the worker is no longer writing them.
    using namespace strata::kernels::cpu;
    const auto& layout = expert_layout();
    for (int i = 0; i < entries_; ++i) {
        const int row = io.dst[i];
        const uint8_t* blob = dispatch.src->blob(dispatch.layers, ids[row]);
        if (!blob) { dispatch.failed = true; dispatch.fail = "expert GPU fallback has no blob"; return; }
        ActQ act;
        alignas(64) uint8_t native_act[kNativeActBytes];
        ExpertJobMulti job{};
        job.blob = blob; job.nt = 1; job.act[0] = &act;
        job.nact[0] = native_act; job.out[0] = out + (size_t) row * width;
        const float* input = x + (size_t) (row / k) * width;
        if (layout.native) {
            const auto& f = layout.fmt[(size_t) dispatch.layers];
            if (f.gu_type == 42) act_quant_any(input, width, act);
            else native_quant_act(f, input, native_act);
            dispatch.pool->run_split_multi_native(f, &job, 1);
        } else {
            act_quant_q8_1(input, width, act);
            dispatch.pool->run_split_multi(&job, 1);
        }
    }
}

void ExpertGpu::adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary_res,
                      const std::vector<std::pair<int32_t, int32_t>>& primary_pending, int max_swaps) {
    if (!enabled_ || usage.size() != res_.size() || primary_res.size() != res_.size()) return;
    poll_swaps();
    if (swapping_ || max_swaps <= 0) return;
    std::vector<char> taken(res_.size(), 0);
    for (const auto& p : primary_pending) taken[(size_t) p.first] = 1;
    struct Swap { float gain; int32_t layer, in, out; };
    std::vector<Swap> swaps;
    std::vector<std::pair<float, int32_t>> cand, vict;
    for (int64_t l = 0; l < n_layers_; ++l) {
        cand.clear();
        vict.clear();
        const size_t row = (size_t) l * (size_t) n_expert_;
        for (int32_t e = 0; e < (int32_t) n_expert_; ++e) {
            const size_t i = row + (size_t) e;
            if (res_[i] >= 0) {
                // a copy GPU0 has since taken is worth nothing here: it goes first
                vict.emplace_back(primary_res[i] >= 0 || taken[i] ? -1.0f : usage[i], e);
            } else if (primary_res[i] < 0 && !taken[i] && usage[i] >= 2.0f) {
                cand.emplace_back(usage[i], e);
            }
        }
        if (cand.empty() || vict.empty()) continue;
        std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
        const size_t nc = std::min(cand.size(), vict.size());
        std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                          [](auto& a, auto& b) { return a.first < b.first; });
        for (size_t i = 0; i < nc; ++i) {
            if (cand[i].first < vict[i].first + 1.5f) break;
            swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
        }
    }
    if (swaps.empty()) return;
    std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
    swaps.resize(std::min(swaps.size(), (size_t) std::min(max_swaps, kSwapCap)));
    int n = 0;
    for (const Swap& s : swaps) {
        const size_t in = (size_t) s.layer * (size_t) n_expert_ + (size_t) s.in;
        const size_t out = (size_t) s.layer * (size_t) n_expert_ + (size_t) s.out;
        const int32_t slot = res_[out];
        if (slot < 0 || slot_layer_[(size_t) slot] != s.layer) continue;
        res_[out] = kNotResident;   // evicted now: GPU0 or the CPU computes it meanwhile
        ch_->swap_slot[n] = slot;
        ch_->swap_layer[n] = s.layer;
        ch_->swap_expert[n] = s.in;
        swap_in_.emplace_back((int32_t) in, slot);   // resident once the worker says the copy has landed
        ++n;
    }
    if (!n) return;
    ch_->swap_n = n;
    ch_->swap_req.store(++swap_req_, std::memory_order_release);
    swapping_ = true;
}

#else   // the worker process and its section are Windows code; elsewhere the engine runs on one GPU

int expert_gpu_worker_main(int, char**) { return 2; }
ExpertGpu::~ExpertGpu() = default;
bool ExpertGpu::alive() const { return false; }
void ExpertGpu::close() {}
bool ExpertGpu::open(const Options&, const ExpertCache&, const std::string&, uint64_t, uint64_t, const std::string&,
                     int64_t, int64_t, const std::vector<std::pair<int32_t, int32_t>>&, std::string& err) {
    err = "the second-GPU worker is implemented for Windows only";
    return false;
}
bool ExpertGpu::contains(int64_t, int32_t) const { return false; }
bool ExpertGpu::layer_pinned(int64_t) const { return false; }
bool ExpertGpu::prefill_ready(int64_t) const { return false; }
uint16_t* ExpertGpu::prefill_input() { return nullptr; }
float* ExpertGpu::prefill_weights() { return nullptr; }
const float* ExpertGpu::prefill_output() const { return nullptr; }
bool ExpertGpu::prefill_launch(int64_t, int, const int32_t*, const int32_t*) { return false; }
bool ExpertGpu::prefill_wait() { return false; }
void ExpertGpu::set_active(bool) {}
void ExpertGpu::disable(const char*) {}
void ExpertGpu::poll_swaps() {}
bool ExpertGpu::launch(int64_t, const float*, const int32_t*, int, int, float*, int32_t*, bool) { return false; }
bool ExpertGpu::wait(std::string&) { return true; }
void ExpertGpu::finish(ExpertDispatch&, const float*, const int32_t*, int, float*) {}
void ExpertGpu::adapt(const std::vector<float>&, const std::vector<int32_t>&,
                      const std::vector<std::pair<int32_t, int32_t>>&, int) {}
#endif
}  // namespace strata::core
