#pragma once

#include "strata/core/expert_cache.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <thread>

namespace strata::core {
class ExpertSource;
struct ExpertDispatch;

// Fixed second tier. All CUDA resources belong to device_, never to the session's device.
class ExpertGpu {
public:
    ExpertGpu() = default;
    ~ExpertGpu();
    ExpertGpu(const ExpertGpu&) = delete;
    ExpertGpu& operator=(const ExpertGpu&) = delete;
    bool prepare(int device, std::string& err);
    bool open(int device, int reserve_mib, int slots, const ExpertCache& primary,
              ExpertSource& source, const std::vector<std::pair<int32_t, int32_t>>& profile,
              std::string& err);
    bool contains(int64_t layer, int32_t expert) const;
    // kind < 0 belongs to the host; -2 means this tier accepted the row.
    bool launch(int64_t layer, const float* x, const int32_t* ids, int nt, int k,
                float* out, int32_t* kind);
    void finish(ExpertDispatch& dispatch, const float* x, const int32_t* ids, int k, float* out);
    int64_t hits = 0, launches = 0;
    /// launch..done; submitting (on the worker, off the critical path); the host spinning in wait(); on the GPU
    double ms = 0, ms_launch = 0, ms_wait = 0, ms_gpu = 0;

private:
    static constexpr int cap = 128, max_tokens = 8, width = 2560;
    // Pinned host staging.  The inputs come first so ONE H2D copy of a prefix moves the tables and the used rows
    // of x to `dev_`: kernels reading them (and writing 4-byte results) over PCIe made ~40K tiny PCIe transactions
    // a layer, 0.46-0.71 ms on the GPU for a few experts.  The rows come back to `out` with one D2H copy.
    struct Mapped {
        unsigned long long ptr[cap];
        int32_t start[cap + 1], dst[cap], tok[cap], count;
        alignas(256) float x[max_tokens * width];
        alignas(256) float out[cap * width];
    };
    int device_ = -1, entries_ = 0;
    bool enabled_ = false, pending_ = false, ready_ = false;
    std::string failure_, name_;
    ExpertCache cache_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t done_ = nullptr, start_ = nullptr;
    cudaStream_t warm_stream_ = nullptr;   ///< low priority: the keep-warm kernel (src/kernels/cuda/keep_warm.cu)
    cudaEvent_t warm_done_ = nullptr;
    Mapped *host_ = nullptr, *dev_ = nullptr;   ///< dev_: the same struct in VRAM
    void* scratch_ = nullptr;
    uint8_t* quant_ = nullptr;
    float* scales_ = nullptr;
    std::chrono::steady_clock::time_point began_;
    // THE WORKER.  Every CUDA call for a layer runs on this thread, which stays on device_: done on the pool's host
    // thread they cost ~0.2 ms a layer (device switches and seven launches under WDDM) before the CPU workers could
    // start.  The host fills the tables, bumps `req_`, and later spins for `ack_`; the job fields below are written
    // before the bump (host) or before the ack (worker), so the release/acquire pair orders them.
    std::thread worker_;
    std::atomic<uint32_t> req_{0}, ack_{0};
    std::atomic<bool> stop_{false};
    int64_t job_layer_ = 0;
    int job_nt_ = 0, job_groups_ = 0, job_lo_ = 0, job_hi_ = 0;   ///< job_lo_..job_hi_: the output rows written
    cudaError_t job_status_ = cudaSuccess;
    float job_gpu_ms_ = 0;
    double job_submit_ms_ = 0;
    void close();
    void disable(const char* reason);
    bool wait(std::string& err);
    void work();
    cudaError_t submit();
};
}  // namespace strata::core
