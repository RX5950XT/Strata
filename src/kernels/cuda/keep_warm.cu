// src/kernels/cuda/keep_warm.cu - keeps a second GPU out of its idle clocks while it serves experts.
//
// The expert GPU (docs/DUAL-GPU.md) gets ~1 ms of work per layer in short bursts, which the driver reads as idle:
// measured on an RTX 3060 Ti it sat in P8 (210 MHz core, 405 MHz memory) and each launch took 0.71 ms on the GPU
// instead of tens of microseconds.  One thread that sleeps until a deadline keeps a kernel resident, so the driver
// holds the working clocks; it uses one SM slot on a low-priority stream.  Keep it short (the caller uses 1 ms):
// under WDDM a long one delays the real kernels on the other stream.
#include <cuda_runtime.h>

#include <cstdint>

namespace strata::kernels {
namespace {

__device__ __forceinline__ uint64_t global_ns() {
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__global__ void keep_warm_kernel(uint64_t ns) {
    const uint64_t end = global_ns() + ns;
    while (global_ns() < end) __nanosleep(100000);
}

}  // namespace

bool keep_warm(uint64_t ns, void* stream) {
    keep_warm_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(ns);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace strata::kernels
