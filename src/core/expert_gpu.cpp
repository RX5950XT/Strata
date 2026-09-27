#include "strata/core/expert_gpu.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <immintrin.h>

namespace strata::kernels {
bool keep_warm(uint64_t ns, void* stream);   // src/kernels/cuda/keep_warm.cu
}

namespace strata::core {
namespace {
void force_device(int device) {
    for (int attempt = 0; attempt < 3; ++attempt) {
        cudaGetLastError();
        if (cudaSetDevice(device) != cudaSuccess) continue;
        int now = -1;
        if (cudaGetDevice(&now) == cudaSuccess && now == device) return;
    }
}
struct DeviceScope {
    int previous = 0;
    bool active = false;
    cudaError_t error = cudaSuccess;
    explicit DeviceScope(int device) {
        // A sticky error refuses cudaGetDevice and cudaSetDevice, which is how a fault on this
        // tier used to leave the session on the wrong GPU.
        if (cudaGetDevice(&previous) != cudaSuccess) {
            cudaGetLastError();
            if (cudaGetDevice(&previous) != cudaSuccess) previous = 0;
        }
        if (previous != device) {
            error = cudaSetDevice(device);
            if (error != cudaSuccess) {
                cudaGetLastError();
                error = cudaSetDevice(device);
            }
        }
        active = error == cudaSuccess;
    }
    ~DeviceScope() {
        if (!active) return;
        int now = -1;
        if (cudaGetDevice(&now) != cudaSuccess) {
            cudaGetLastError();
            now = -1;
        }
        if (now == previous) return;
        cudaGetLastError();             // the fault belongs to this tier's device; clear it so the switch works
        if (cudaSetDevice(previous) != cudaSuccess) {
            cudaGetLastError();
            cudaSetDevice(previous);
        }
    }
};
// Destroyed after any DeviceScope in the same function, so a failed switch above still ends on `device`.
struct RestoreDevice {
    int device;
    explicit RestoreDevice(int device) : device(device) {}
    ~RestoreDevice() { force_device(device); }
};
void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace

ExpertGpu::~ExpertGpu() { close(); }

void ExpertGpu::close() {
    if (worker_.joinable()) {
        stop_.store(true, std::memory_order_release);
        req_.fetch_add(1, std::memory_order_release);
        req_.notify_one();
        worker_.join();
    }
    stop_.store(false);
    req_.store(0);
    ack_.store(0);
    if (device_ < 0) return;
    const int dev = device_;
    cudaGetLastError();
    force_device(dev);
    cudaGetLastError();
    if (stream_) cudaStreamSynchronize(stream_);
    cudaGetLastError();
    cache_.close();
    if (scratch_) cudaFree(scratch_);
    if (quant_) cudaFree(quant_);
    if (scales_) cudaFree(scales_);
    if (host_) cudaFreeHost(host_);
    if (dev_) cudaFree(dev_);
    if (done_) cudaEventDestroy(done_);
    if (start_) cudaEventDestroy(start_);
    if (warm_done_) cudaEventDestroy(warm_done_);
    if (warm_stream_) cudaStreamDestroy(warm_stream_);
    warm_done_ = nullptr; warm_stream_ = nullptr;
    if (stream_) cudaStreamDestroy(stream_);
    cudaGetLastError();
    scratch_ = nullptr; quant_ = nullptr; scales_ = nullptr;
    host_ = dev_ = nullptr; done_ = start_ = nullptr; stream_ = nullptr;
    enabled_ = pending_ = ready_ = false;
    device_ = -1;
}

bool ExpertGpu::prepare(int ordinal, std::string& err) {
    using namespace strata::kernels;
    if (device_ == ordinal && stream_) return true;
    int home = 0;
    if (cudaGetDevice(&home) != cudaSuccess) {
        cudaGetLastError();
        home = 0;
    }
    RestoreDevice restore(home);
    const char* stage = "select device";
    try {
        int count = 0, primary_device = -1;
        checked(cudaGetDevice(&primary_device));
        checked(cudaGetDeviceCount(&count));
        if (ordinal < 0 || ordinal >= count) throw std::runtime_error("device is not visible");
        if (ordinal == primary_device) throw std::runtime_error("device is GPU0");
        device_ = ordinal;
        DeviceScope device(device_);
        checked(device.error);
        cudaDeviceProp prop{};
        checked(cudaGetDeviceProperties(&prop, device_));
        if (prop.major < 8) throw std::runtime_error("compute capability must be at least 8.0");
        name_ = prop.name;
        stage = "create stream/event";
        checked(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        checked(cudaEventCreate(&done_));
        checked(cudaEventCreate(&start_));
        int least = 0, greatest = 0;
        checked(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        checked(cudaStreamCreateWithPriority(&warm_stream_, cudaStreamNonBlocking, least));
        checked(cudaEventCreateWithFlags(&warm_done_, cudaEventDisableTiming));
        stage = "allocate staging";
        checked(cudaHostAlloc((void**) &host_, sizeof(Mapped), cudaHostAllocPortable));
        checked(cudaMalloc((void**) &dev_, sizeof(Mapped)));
        stage = "allocate kernel scratch";
        checked(cudaMalloc(&scratch_, std::max(moe_hit_grouped_scratch_bytes(cap, width, cpu::FF),
                                             (uint64_t) native_expert_scratch_bytes(cap, cpu::FF))));
        checked(cudaMalloc(&quant_, max_tokens * (width / 32) * 36));
        checked(cudaMalloc(&scales_, max_tokens * (width / 32) * sizeof(float)));
        std::memset(host_, 0, sizeof(Mapped));
        return true;
    } catch (const std::exception& error) {
        err = std::string(stage) + ": " + error.what();
        cudaGetLastError();
        close();
        force_device(home);
        cudaGetLastError();
        return false;
    }
}

bool ExpertGpu::open(int ordinal, int reserve_mib, int slots, const ExpertCache& primary,
                     ExpertSource& source, const std::vector<std::pair<int32_t, int32_t>>& profile,
                     std::string& err) {
    using namespace strata::kernels;
    int home = 0;
    if (cudaGetDevice(&home) != cudaSuccess) {
        cudaGetLastError();
        home = 0;
    }
    // Failure must leave the caller's device and its error state as they were: a sticky fault here
    // is what made the primary hit path report "could not allocate" after this tier gave up.
    auto bail = [&](const std::string& why) {
        err = why;
        const int dev = device_;
        close();
        if (dev >= 0) {
            force_device(dev);
            cudaGetLastError();
        }
        force_device(home);
        cudaGetLastError();
        return false;
    };
    if (!prepare(ordinal, err)) return bail(err);
    RestoreDevice restore(home);
    try {
        DeviceScope device(device_);
        checked(device.error);
        if (reserve_mib < 0 || slots < -1) throw std::runtime_error("invalid reserve or slot cap");
        if (profile.empty()) throw std::runtime_error("an expert profile is required");
        const auto& layout = cpu::expert_layout();
        for (const auto& f : layout.fmt) {
            const bool gu = f.gu_type == 16 || f.gu_type == 17 || f.gu_type == 18 || f.gu_type == 21 ||
                            f.gu_type == 22 || f.gu_type == 23 || f.gu_type == 29 || f.gu_type == 42;
            const bool down = f.d_type == 20 || f.d_type == 23 || f.d_type == 42;
            if (!gu || !down || f.n_embd != width || f.n_ff != cpu::FF)
                throw std::runtime_error("unsupported native expert geometry or format");
        }
        // The profile's pairs GPU0 did not take, in rank order, as many as fit above the reserve.
        size_t free = 0, total = 0;
        checked(cudaMemGetInfo(&free, &total));
        const uint64_t reserve = (uint64_t) reserve_mib << 20;
        const uint64_t room = free > reserve ? (uint64_t) free - reserve : 0;
        std::vector<int64_t> sizes;
        std::vector<std::pair<int32_t, int32_t>> pairs;
        uint64_t used = 0;
        for (const auto& pair : profile) {
            if (primary.slot_of(pair.first, pair.second) >= 0) continue;
            const uint64_t bytes = layout.blob_bytes(pair.first);
            const uint64_t step = (bytes + 255) & ~255ull;
            if (used + step > room || (slots >= 0 && sizes.size() >= (size_t) slots)) break;
            used += step;
            sizes.push_back((int64_t) bytes);
            pairs.push_back(pair);
        }
        if (pairs.empty()) throw std::runtime_error("no slots fit after the reserve");
        // Under WDDM a cudaMalloc takes its memory when it is first written, so the slots are zeroed and THEN the
        // reserve is checked (the same order as GPU0's cache in generate.cpp); a shortfall - the image encoder
        // sharing the card, the desktop - drops that much from the end of the list and tries again.
        std::string cache_err;
        for (int attempt = 0;; ++attempt) {
            if (!cache_.open_sized(sizes, layout.n_layers, layout.n_expert, cache_err)) throw std::runtime_error(cache_err);
            checked(cudaMemset(cache_.device_slot(0), 0, (size_t) cache_.bytes()));
            checked(cudaDeviceSynchronize());
            checked(cudaMemGetInfo(&free, &total));
            if (free >= reserve) break;
            uint64_t drop = reserve - free + (64ull << 20), dropped = 0;
            while (!sizes.empty() && dropped < drop) {
                dropped += ((uint64_t) sizes.back() + 255) & ~255ull;
                sizes.pop_back();
                pairs.pop_back();
            }
            cache_.close();
            if (sizes.empty() || attempt == 3)
                throw std::runtime_error("only " + std::to_string(free >> 20) + " MiB free once the slots are written (reserve " +
                                         std::to_string(reserve_mib) + " MiB)");
        }
        // A plain cudaMemcpy: the arena may be only partly pinned (see --expert-gpu-pin-gib), and this is startup.
        for (size_t i = 0; i < pairs.size(); ++i) {
            const uint8_t* blob = source.blob(pairs[i].first, pairs[i].second);
            if (!blob) throw std::runtime_error("the expert source has no blob");
            const int slot = cache_.admit(pairs[i].first, pairs[i].second);
            if (!cache_.fill_slot_blocking(slot, blob, cache_err, sizes[i])) throw std::runtime_error(cache_err);
        }
        if (!cache_.verify_slot(0, source.blob(pairs[0].first, pairs[0].second), cache_err, sizes[0]))
            throw std::runtime_error(cache_err);
        // Run every expert format this tier holds once, so a kernel that cannot launch on this card fails here
        // and not at the first doorbell.
        worker_ = std::thread(&ExpertGpu::work, this);
        enabled_ = true;
        std::vector<int> seen;
        for (const auto& pair : pairs) {
            const auto& f = layout.fmt[(size_t) pair.first];
            const int mark = layout.native ? f.gu_type * 64 + f.d_type : 0;
            if (std::find(seen.begin(), seen.end(), mark) != seen.end()) continue;
            seen.push_back(mark);
            int32_t kind = -1, id = pair.second;
            std::string launch_err;
            if (!launch(pair.first, host_->x, &id, 1, 1, host_->out, &kind) || !wait(launch_err))
                throw std::runtime_error("kernel self-test: " + (launch_err.empty() ? failure_ : launch_err));
        }
        hits = 0;
        ms = ms_launch = ms_wait = ms_gpu = 0;
        launches = 0;
        ready_ = true;
        std::fprintf(stderr, "strata generate: expert GPU: %lld slots on %s (device %d), %.2f GiB, %zu MiB free of %zu MiB; slot 0 verified\n",
                     (long long) cache_.slots(), name_.c_str(), device_, cache_.gib(), free >> 20, total >> 20);
        return true;
    } catch (const std::exception& error) {
        return bail(error.what());
    }
}

bool ExpertGpu::contains(int64_t layer, int32_t expert) const {
    return enabled_ && cache_.slot_of(layer, expert) >= 0;
}

void ExpertGpu::disable(const char* reason) {
    failure_ = reason;
    if (enabled_ && ready_) std::fprintf(stderr, "strata generate: expert GPU %d disabled: %s\n", device_, reason);
    enabled_ = false;
}

bool ExpertGpu::launch(int64_t layer, const float* x, const int32_t* ids, int nt, int k,
                       float* out, int32_t* kind) {
    if (!enabled_ || nt < 1 || nt > max_tokens || k < 1 || nt * k > cap) return false;
    entries_ = 0;
    int groups = 0;
    for (int i = 0; i < nt * k; ++i) {
        if (kind[i] != -1 || !contains(layer, ids[i])) continue;
        host_->ptr[groups] = (unsigned long long) cache_.device_slot(cache_.slot_of(layer, ids[i]));
        host_->start[groups++] = entries_;
        for (int j = i; j < nt * k; ++j) {
            if (kind[j] != -1 || ids[j] != ids[i]) continue;
            host_->dst[entries_] = j;
            host_->tok[entries_++] = j / k;
            kind[j] = -2;
        }
    }
    if (!entries_) return false;
    host_->start[groups] = entries_;
    host_->count = groups;
    job_lo_ = *std::min_element(host_->dst, host_->dst + entries_);
    job_hi_ = *std::max_element(host_->dst, host_->dst + entries_);
    began_ = std::chrono::steady_clock::now();
    if (x != host_->x) std::memcpy(host_->x, x, (size_t) nt * width * sizeof(float));
    job_layer_ = layer;
    job_nt_ = nt;
    job_groups_ = groups;
    pending_ = true;
    req_.fetch_add(1, std::memory_order_release);
    req_.notify_one();
    return true;
}

// On the worker, with device_ current: the layer's kernels, and its answer once the GPU has finished.
cudaError_t ExpertGpu::submit() {
    using namespace strata::kernels;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        const auto& layout = cpu::expert_layout();
        // Keep the card at its working clocks while the bursts last (keep_warm.cu).  1 ms kernels, relaunched here:
        // under WDDM a 200 ms one held the expert kernels behind it (p90 193 ms), 1 ms ones cost nothing measurable,
        // and an idle server lets the card idle again 1 ms after the last layer.
        const cudaError_t warm = cudaEventQuery(warm_done_);
        if (warm == cudaSuccess && keep_warm(1000000ull, warm_stream_)) checked(cudaEventRecord(warm_done_, warm_stream_));
        else if (warm != cudaErrorNotReady) cudaGetLastError();
        checked(cudaEventRecord(start_, stream_));
        checked(cudaMemcpyAsync(dev_, host_, offsetof(Mapped, x) + (size_t) job_nt_ * width * sizeof(float),
                                cudaMemcpyHostToDevice, stream_));
        if (layout.native) {
            const auto& f = layout.fmt[(size_t) job_layer_];
            const auto L = native_expert_layout(f.gu_type, f.d_type, f.n_embd, f.n_ff);
            quantize_q8_1_rows(dev_->x, job_nt_, width, quant_, stream_, true);
            native_expert_grouped(L, dev_->ptr, dev_->start, &dev_->count, dev_->dst, dev_->tok,
                                  job_groups_, entries_, quant_, scratch_, dev_->out, stream_, true);
        } else {
            quantize_q8_0_scaled(dev_->x, quant_, scales_, job_nt_ * width, stream_, true);
            moe_grouped_s2(dev_->ptr, dev_->start, &dev_->count, dev_->dst, dev_->tok,
                           job_groups_, entries_, quant_, scales_, scratch_, dev_->out, stream_, true);
        }
        const size_t lo = (size_t) job_lo_ * width, rows = (size_t) (job_hi_ - job_lo_ + 1) * width;
        checked(cudaMemcpyAsync(host_->out + lo, dev_->out + lo, rows * sizeof(float), cudaMemcpyDeviceToHost, stream_));
        checked(cudaEventRecord(done_, stream_));
    } catch (const std::exception&) {
        const cudaError_t e = cudaGetLastError();
        cudaStreamSynchronize(stream_);
        return e != cudaSuccess ? e : cudaErrorLaunchFailure;
    }
    job_submit_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    cudaError_t status;
    do { status = cudaEventQuery(done_); } while (status == cudaErrorNotReady);
    job_gpu_ms_ = 0;
    if (status == cudaSuccess) cudaEventElapsedTime(&job_gpu_ms_, start_, done_);
    else cudaStreamSynchronize(stream_);
    return status;
}

void ExpertGpu::work() {
    cudaSetDevice(device_);
    uint32_t served = ack_.load(std::memory_order_relaxed);
    auto last = std::chrono::steady_clock::now();
    while (true) {
        const uint32_t want = req_.load(std::memory_order_acquire);
        if (stop_.load(std::memory_order_acquire)) return;
        if (want == served) {
            // spin while layers keep coming (one every ~1 ms), sleep once they stop
            if (std::chrono::steady_clock::now() - last < std::chrono::milliseconds(2)) _mm_pause();
            else req_.wait(served, std::memory_order_acquire);
            continue;
        }
        job_status_ = submit();
        served = want;
        last = std::chrono::steady_clock::now();
        ack_.store(served, std::memory_order_release);
    }
}

bool ExpertGpu::wait(std::string& err) {
    if (!pending_) return true;
    const auto spin = std::chrono::steady_clock::now();
    const uint32_t want = req_.load(std::memory_order_relaxed);
    while (ack_.load(std::memory_order_acquire) != want) _mm_pause();
    pending_ = false;
    const auto now = std::chrono::steady_clock::now();
    ms_wait += std::chrono::duration<double, std::milli>(now - spin).count();
    ms += std::chrono::duration<double, std::milli>(now - began_).count();
    ms_launch += job_submit_ms_;
    if (job_status_ == cudaSuccess) {
        ms_gpu += job_gpu_ms_;
        hits += entries_;
        ++launches;
        return true;
    }
    err = cudaGetErrorString(job_status_);
    return false;
}

void ExpertGpu::finish(ExpertDispatch& dispatch, const float* x, const int32_t* ids, int k, float* out) {
    if (!pending_) return;
    std::string err;
    if (wait(err)) {
        for (int i = 0; i < entries_; ++i) {   // only this tier's rows: the rest of `out` is the CPU's
            const size_t offset = (size_t) host_->dst[i] * width;
            std::memcpy(out + offset, host_->out + offset, width * sizeof(float));
        }
        return;
    }
    disable(err.c_str());
    // A runtime failure must still supply every accepted row, after GPU writes have stopped.
    using namespace strata::kernels::cpu;
    const auto& layout = expert_layout();
    for (int i = 0; i < entries_; ++i) {
        const int row = host_->dst[i];
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
}  // namespace strata::core
