#pragma once

#include "strata/core/expert_cache.hpp"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {
struct ExpertDispatch;
struct EgpuChannel;

// The second GPU as a second expert tier, driven by a WORKER PROCESS (`strata --expert-gpu-worker ...`) that sees
// only that GPU and maps the expert arena (a named section) itself.
//
// Why a process and not a second context in this one: Windows lets the machine page-lock about 78 GiB of host
// memory in total, counted once per GPU the memory is mapped for, and a pin in a process with two contexts is
// mapped for both - so that process could pin only ~39 GiB of the arena.  Here the engine pins the whole arena for
// GPU0, exactly as with one GPU, and the worker pins what is left of the budget for its own PCIe reads
// (docs/DUAL-GPU.md).
//
// The engine decides everything (which experts the worker holds, which misses it reads over its own PCIe link,
// which slots it swaps); the worker only copies and computes.  All of this object's state lives in this process.
class ExpertGpu {
public:
    struct Options {
        std::string device;       ///< the worker's CUDA_VISIBLE_DEVICES
        int reserve_mib = 512;    ///< VRAM the worker leaves free
        int slots = -1;           ///< cap on its slots (-1 = as many as fit)
        double pin_gib = -1;      ///< arena the worker pins for its PCIe reads (< 0 = what the machine's budget leaves)
        int pcie_num = 0;         ///< share of a verify window's misses (x/256) the worker reads over its PCIe link
    };
    ExpertGpu() = default;
    ~ExpertGpu();
    ExpertGpu(const ExpertGpu&) = delete;
    ExpertGpu& operator=(const ExpertGpu&) = delete;
    /// `arena_name` / `arena_bytes`: the section the expert arena lives in; `main_pinned`: what this process pinned.
    bool open(const Options& opt, const ExpertCache& primary, const std::string& arena_name, uint64_t arena_bytes,
              uint64_t main_pinned, const std::string& pack_dir, int64_t n_layers, int64_t n_expert,
              const std::vector<std::pair<int32_t, int32_t>>& profile, std::string& err);
    bool contains(int64_t layer, int32_t expert) const;
    /// Claims this tier's rows (kind -1 -> -2): its resident experts, and with `pcie` a share of the other misses,
    /// which it reads over its own PCIe link.  kind 0 rows (GPU0's) are never taken.
    bool launch(int64_t layer, const float* x, const int32_t* ids, int nt, int k, float* out, int32_t* kind,
                bool pcie = false);
    void finish(ExpertDispatch& dispatch, const float* x, const int32_t* ids, int k, float* out);
    /// Between rounds: swap the most-routed experts that neither GPU holds into this tier's least-routed slots of the
    /// same layer.  `primary_res` / `primary_pending`: GPU0's residency and the swaps it has in flight.
    void adapt(const std::vector<float>& usage, const std::vector<int32_t>& primary_res,
               const std::vector<std::pair<int32_t, int32_t>>& primary_pending, int max_swaps);
    /// A request is running: the worker keeps its GPU at working clocks (and its PCIe link at full speed) from the
    /// prompt on, instead of spending the first second of the reply climbing out of P8.
    void set_active(bool on);
    int64_t hits = 0, streamed = 0, launches = 0, swapped = 0;
    /// launch..done; submitting (on the worker, off the critical path); this process spinning in wait(); on the GPU
    double ms = 0, ms_launch = 0, ms_wait = 0, ms_gpu = 0;

private:
    static constexpr int cap = 128, max_tokens = 8, width = 2560;
    EgpuChannel* ch_ = nullptr;
    void *map_ = nullptr, *proc_ = nullptr, *job_ = nullptr, *wake_ = nullptr;
    bool enabled_ = false, pending_ = false, swapping_ = false;
    int entries_ = 0, pcie_num_ = 0;
    int64_t n_layers_ = 0, n_expert_ = 0;
    uint64_t pinned_ = 0;                    ///< the worker's pinned arena prefix
    std::vector<int32_t> res_;               ///< (layer, expert) -> worker slot, or kNotResident
    std::vector<int32_t> slot_layer_;        ///< the layer each slot was sized for (swaps stay in that layer)
    std::vector<std::pair<int32_t, int32_t>> swap_in_;   ///< (residency index, slot) once the worker confirms
    std::string failure_, name_;
    double began_ = 0;
    uint32_t req_ = 0, swap_req_ = 0;
    void close();
    void disable(const char* reason);
    bool wait(std::string& err);
    void poll_swaps();
    bool alive() const;
};

/// The worker process's main (`strata --expert-gpu-worker <channel> <arena> <arena bytes> <pack> <layers> <experts>
/// <reserve MiB> <pin bytes>`).
int expert_gpu_worker_main(int argc, char** argv);
}  // namespace strata::core
