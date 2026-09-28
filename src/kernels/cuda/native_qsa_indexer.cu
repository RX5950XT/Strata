// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp; ggml/src/ggml-cuda/{set-rows.cu,norm.cu,rope.cu}.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {
std::atomic<bool> enabled{false};
constexpr int D = 128, R = 4, ROT = 64, THREADS = 256;
__device__ float warp_sum(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset);
    return x;
}
__device__ float pooled_value(float incoming, const float* previous, int pos, int pos_base,
                              const float* gamma, float epsilon, float theta_scale, const int32_t* mtab) {
    const int d = threadIdx.x;
    __shared__ float values[D];
    __shared__ float partials[32];
    float mean = 0.0f;
    if (d < D) {
        // The spare's four gather indices all name cell zero. Completed blocks
        // use chronological slices; each graph ADD materializes an F32 sum.
        float sum = pos == 0 ? incoming : previous[0];
#pragma unroll
        for (int j = 1; j < R; ++j)
            sum = __fadd_rn(sum, pos == 0 || j == R - 1 ? incoming : previous[j]);
        mean = __fmaf_rn(0.25f, sum, 0.0f); // SCALE includes a zero bias.
    }
    float square_sum = 0.0f;
    if (d < D) square_sum += mean * mean;
    square_sum = warp_sum(square_sum);
    const int lane = d % 32;
    if (lane == 0) partials[d / 32] = square_sum;
    __syncthreads();
    square_sum = lane < THREADS / 32 ? partials[lane] : 0.0f;
    square_sum = warp_sum(square_sum);
    const float scale = rsqrtf(square_sum / D + epsilon);
    if (d < D) values[d] = scale * mean * gamma[d];
    __syncthreads();
    if (d >= D) return 0.0f;
    const int b = pos / R;
    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
    float y = values[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const float theta = (pos == 0 ? 0 : mrope_pos(mtab, rope_pos, pair)) * powf(theta_scale, float(pair));
        const float c = cosf(theta), s = sinf(theta);
        const float a = values[pair], z = values[pair + ROT / 2];
        y = d < ROT / 2 ? a * c - z * s : a * s + z * c;
    }
    return y;
}
__global__ void append(const float* __restrict__ raw, const int32_t* __restrict__ pos_dev,
                        int pos_base, const float* __restrict__ gamma, float epsilon,
                        float* __restrict__ tail, float* __restrict__ dead,
                        float* __restrict__ pooled, int32_t* __restrict__ block_pos,
                        int max_cells, float theta_scale, const int32_t* __restrict__ mtab) {
    const int pos = *pos_dev, d = threadIdx.x;
    if (pos < 0 || pos >= max_cells) return;
    const int slot = pos % R;
    float incoming = 0.0f;
    if (d < D) {
        // SET_ROWS stores F16; GET_ROWS expands those exact values to F32.
        incoming = __half2float(__float2half_rn(raw[d]));
        if (slot < R - 1) tail[slot * D + d] = incoming;
    }
    if (pos != 0 && slot != R - 1) return;
    float previous[R - 1] = {};
    if (d < D && pos != 0)
        for (int j = 0; j < R - 1; ++j) previous[j] = tail[j * D + d];
    const float y = pooled_value(incoming, previous, pos, pos_base, gamma, epsilon, theta_scale, mtab);
    if (d >= D) return;
    const int b = pos / R;
    const int rope_pos = pos == 0 ? 0 : pos_base + R * b;
    pooled[std::size_t(b) * D + d] = y;
    if (pos == 0) dead[d] = y;
    else pooled[std::size_t(b + 1) * D + d] = dead[d];
    if (d == 0 && pos != 0) *block_pos = rope_pos;
}
// Only completed rows and cell zero's dead key are written here. In particular,
// spare rows and tail writes must wait until all blocks have read the old tail.
__global__ void append_batch_rows(const float* raw, int64_t T, const int32_t* positions,
                                   int64_t stride, int pos_base, const float* gamma, float epsilon,
                                   QsaIndexerBuffers buffers, int max_cells, float theta_scale,
                                   const int32_t* mtab) {
    const int64_t t = blockIdx.x;
    const int pos = positions[t * stride], d = threadIdx.x;
    if (pos < 0 || pos >= max_cells || (pos != 0 && pos % R != R - 1)) return;
    float incoming = 0.0f, previous[R - 1] = {};
    if (d < D) {
        incoming = __half2float(__float2half_rn(raw[t * D + d]));
        if (pos != 0) for (int j = 0; j < R - 1; ++j) {
            const int64_t source = t - (R - 1) + j;
            previous[j] = source >= 0
                ? __half2float(__float2half_rn(raw[source * D + d])) : buffers.tail[j * D + d];
        }
    }
    const float y = pooled_value(incoming, previous, pos, pos_base, gamma, epsilon, theta_scale, mtab);
    if (d < D) {
        if (pos == 0) buffers.dead[d] = y;
        else buffers.pooled[std::size_t(pos / R) * D + d] = y;
    }
}
__global__ void append_batch_finish(const float* raw, int64_t T, const int32_t* positions,
                                     int64_t stride, int pos_base, QsaIndexerBuffers buffers, int max_cells) {
    const int d = threadIdx.x;
    // the last in-range cell: a verify commit pads the rejected tokens with -1
    __shared__ int64_t last_t;
    if (d == 0) {
        last_t = -1;
        for (int64_t t = T - 1; t >= 0; --t) {
            const int p = positions[t * stride];
            if (p >= 0 && p < max_cells) { last_t = t; break; }
        }
    }
    __syncthreads();
    if (last_t < 0) return;
    const int64_t last = positions[last_t * stride], first = last - last_t;
    // Each tail slot retains its last write, even after a block completes.
    for (int j = 0; j < R - 1; ++j) {
        const int64_t pos = last - (last - j + R) % R, t = pos - first;
        if (d < D && pos >= 0 && t >= 0 && t < T && positions[t * stride] == pos)
            buffers.tail[j * D + d] = __half2float(__float2half_rn(raw[t * D + d]));
    }
    const int64_t complete = last - (last + 1) % R, t = complete - first;
    if (complete >= 0 && t >= 0 && t < T && positions[t * stride] == complete) {
        if (d < D) buffers.pooled[std::size_t(complete / R + 1) * D + d] = buffers.dead[d];
        if (d == 0) *buffers.block_pos = pos_base + R * int(complete / R);
    } else if (first <= 0 && -first < T && positions[-first * stride] == 0) {
        if (d < D) buffers.pooled[d] = buffers.dead[d];
    }
}
struct Span { const void* p; std::size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<std::uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p)
        throw std::invalid_argument("native QSA indexer requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a.p), y = reinterpret_cast<std::uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
} // namespace

void native_qsa_indexer_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_indexer_enabled() { return enabled.load(std::memory_order_relaxed); }
void native_qsa_indexer_append(const float* raw, const int32_t* relative_pos_device, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, float freq_base, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const Span spans[] = {{raw,D*4},{relative_pos_device,4},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    append<<<1,THREADS,0,static_cast<cudaStream_t>(stream)>>>(raw,relative_pos_device,pos_base,gamma,epsilon,
        b.tail,b.dead,b.pooled,b.block_pos,int(max_cells),theta_scale,mrope_table());
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_qsa_indexer_append_batch(const float* raw, int64_t T, const int32_t* relative_pos_device,
                                     int64_t pos_stride, int32_t pos_base,
                               const float* gamma, float epsilon, const QsaIndexerBuffers& b,
                               const QsaShapes& s, int64_t max_cells, float freq_base, void* stream) {
    if (T < 1 || T > INT32_MAX || pos_stride < 1 ||
        uint64_t(T) > SIZE_MAX / (D * sizeof(float)) ||
        uint64_t(pos_stride) > (SIZE_MAX / sizeof(int32_t) - 1) / uint64_t(T > 1 ? T - 1 : 1))
        throw std::invalid_argument("native QSA indexer requires bounded positive batch size and position stride");
    if (!stream || s.idx_dim != D || s.idx_block != R || s.n_rot != ROT ||
        max_cells < 1 || max_cells > INT32_MAX || pos_base < 0 || pos_base % R ||
        int64_t(pos_base) + max_cells > INT32_MAX || !std::isfinite(epsilon) || epsilon <= 0.0f ||
        !std::isfinite(freq_base) || freq_base <= 1.0f)
        throw std::invalid_argument("native QSA indexer requires fixed geometry, aligned position base, positive capacity/epsilon, valid frequency and explicit stream");
    const Span spans[] = {{raw,std::size_t(T)*D*4},{relative_pos_device,(std::size_t(T-1)*pos_stride+1)*4},{gamma,D*4},{b.tail,(R-1)*D*4},
        {b.dead,D*4},{b.pooled,std::size_t(max_cells/R+1)*D*4},{b.block_pos,4}};
    for (const auto& span : spans) validate(span);
    for (int i = 0; i < 7; ++i) for (int j = i + 1; j < 7; ++j)
        if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA indexer buffers overlap");
    const float theta_scale = powf(freq_base, -2.0f / ROT);
    append_batch_rows<<<unsigned(T),THREADS,0,static_cast<cudaStream_t>(stream)>>>(raw,T,relative_pos_device,
        pos_stride,pos_base,gamma,epsilon,b,int(max_cells),theta_scale,mrope_table());
    auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    append_batch_finish<<<1,THREADS,0,static_cast<cudaStream_t>(stream)>>>(raw,T,relative_pos_device,
        pos_stride,pos_base,b,int(max_cells));
    error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::kernels
