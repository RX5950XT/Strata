#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/mrope.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strata::kernels;
#ifdef ORIGINAL_CHECK
namespace strata::kernels {
void native_qsa_indexer_append_original(const float*, const int32_t*, int32_t, const float*, float,
    const QsaIndexerBuffers&, const QsaShapes&, int64_t, float, void*);
}
#endif
void check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    ~Device() { cudaFree(p); }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
};
struct State {
    Device<float> tail{384}, dead{128}, pooled;
    Device<int32_t> pos{1};
    explicit State(int cap) : pooled(size_t(cap / 4 + 1) * 128) {
        check(cudaMemset(tail.p, 0, 384 * 4));
        check(cudaMemset(dead.p, 0, 128 * 4));
        check(cudaMemset(pooled.p, 0, size_t(cap / 4 + 1) * 128 * 4));
        check(cudaMemset(pos.p, 0xff, 4));
    }
    QsaIndexerBuffers buffers() { return {tail.p, dead.p, pooled.p, pos.p}; }
};
void equal(const void* a, const void* b, size_t bytes, const char* name) {
    std::vector<unsigned char> x(bytes), y(bytes);
    check(cudaMemcpy(x.data(), a, bytes, cudaMemcpyDeviceToHost));
    check(cudaMemcpy(y.data(), b, bytes, cudaMemcpyDeviceToHost));
    if (std::memcmp(x.data(), y.data(), bytes)) {
        size_t i = 0;
        while (x[i] == y[i]) ++i;
        std::printf("FAIL %s byte=%zu sequential=%u batch=%u\n", name, i, x[i], y[i]);
        throw std::runtime_error("bitwise mismatch");
    }
}
void compare(State& a, State& b, int cap) {
    equal(a.tail.p, b.tail.p, 384 * 4, "tail");
    equal(a.dead.p, b.dead.p, 128 * 4, "dead");
    equal(a.pooled.p, b.pooled.p, size_t(cap / 4 + 1) * 128 * 4, "pooled");
    equal(a.pos.p, b.pos.p, 4, "block_pos");
}
// keep: per chunk, how many leading tokens are real; the rest are -1, as a verify commit pads rejected tokens
void run(int start, const std::vector<int>& chunks, int cap, int base, int stride,
         bool table, cudaStream_t stream, const std::vector<int>& keep = {}) {
    int total = std::max(start, 0);
    for (int n : chunks) total += n;
    std::mt19937 rng(1234 + total + base + stride);
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<float> raw(size_t(total) * 128), gamma(128);
    for (float& v : raw) v = dist(rng);
    for (float& v : gamma) v = dist(rng);
    for (size_t i = 0; i < raw.size(); i += 257) raw[i] = -0.0f;
    for (size_t i = 1; i < raw.size(); i += 263) raw[i] = 1.00048828125f;
    std::vector<int32_t> positions(size_t(total) * stride, -12345);
    for (int i = 0; i < total; ++i) positions[size_t(i) * stride] = i + std::min(start, 0);
    if (!keep.empty()) {
        int i = std::max(start, 0), p = i;
        for (size_t c = 0; c < chunks.size(); ++c)
            for (int k = 0; k < chunks[c]; ++k, ++i) positions[size_t(i) * stride] = k < keep[c] ? p++ : -1;
    }
    Device<float> dr(raw.size()), dg(128);
    Device<int32_t> dp(positions.size()), mt(size_t(base + cap) * 3);
    check(cudaMemcpy(dr.p, raw.data(), raw.size() * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dg.p, gamma.data(), 128 * 4, cudaMemcpyHostToDevice));
    check(cudaMemcpy(dp.p, positions.data(), positions.size() * 4, cudaMemcpyHostToDevice));
    if (table) {
        std::vector<int32_t> host(size_t(base + cap) * 3);
        for (size_t i = 0; i < host.size(); ++i) host[i] = int(i / 3 + i % 3 * 7);
        check(cudaMemcpy(mt.p, host.data(), host.size() * 4, cudaMemcpyHostToDevice));
    }
    mrope_table_set(table ? mt.p : nullptr);
    State single(cap), batch(cap);
    auto one = [&](State& state, int i) {
        native_qsa_indexer_append(dr.p + size_t(i) * 128, dp.p + size_t(i) * stride, base,
            dg.p, 1e-6f, state.buffers(), qsa_real_shapes(), cap, 1e7f, stream);
    };
    for (int i = 0; i < std::max(start, 0); ++i) { one(single, i); one(batch, i); }
    int offset = std::max(start, 0);
    for (int n : chunks) {
        for (int i = offset; i < offset + n; ++i) one(single, i);
        native_qsa_indexer_append_batch(dr.p + size_t(offset) * 128, n, dp.p + size_t(offset) * stride,
            stride, base, dg.p, 1e-6f, batch.buffers(), qsa_real_shapes(), cap, 1e7f, stream);
        check(cudaStreamSynchronize(stream));
        compare(single, batch, cap);
        offset += n;
    }
#ifdef ORIGINAL_CHECK
    State original(cap);
    for (int i = 0; i < total; ++i)
        native_qsa_indexer_append_original(dr.p + size_t(i) * 128, dp.p + size_t(i) * stride, base,
            dg.p, 1e-6f, original.buffers(), qsa_real_shapes(), cap, 1e7f, stream);
    check(cudaStreamSynchronize(stream));
    compare(original, single, cap);
#endif
    mrope_table_set(nullptr);
    std::printf("PASS start=%d T=%d calls=%zu capacity=%d base=%d stride=%d mrope=%d\n",
        start, total - std::max(start, 0), chunks.size(), cap, base, stride, int(table));
}
int main(int argc, char** argv) {
    try {
        int device = argc > 1 ? std::stoi(argv[1]) : 0;
        if (argc == 1) {
            int count = 0, best = -1;
            check(cudaGetDeviceCount(&count));
            for (int i = 0; i < count; ++i) {
                cudaDeviceProp candidate{};
                check(cudaGetDeviceProperties(&candidate, i));
                const int capability = candidate.major * 10 + candidate.minor;
                if (capability > best) { best = capability; device = i; }
            }
        }
        check(cudaSetDevice(device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, device));
        std::printf("GPU %d: %s\n", device, prop.name);
        cudaStream_t stream;
        check(cudaStreamCreate(&stream));
        for (int n : {1, 2, 3, 4, 5, 8191, 8192}) run(0, {n}, 8200, 0, 1, false, stream);
        for (int start : {1, 2, 3, 5})
            for (int n : {1, 2, 3, 4, 5, 17}) run(start, {n}, 40, 0, 19, false, stream);
        run(0, {1, 2, 1, 5, 3, 16, 2}, 40, 128, 19, false, stream);
        run(3, {2, 7, 1, 9}, 40, 128, 3, true, stream);
        for (int cap : {1, 2, 3, 4, 5, 7, 8, 9}) run(0, {16, 4}, cap, 16, 3, false, stream);
        run(-5, {2, 2, 1, 1, 2, 9}, 9, 16, 3, false, stream);
        run(-5, {20}, 9, 16, 3, true, stream);
        for (int start : {0, 1, 2, 3, 4, 5, 6, 7})
            for (int k : {1, 2, 3, 4, 5}) run(start, {5, 5, 5}, 40, 0, 7, false, stream, {k, 5 - k / 2, k});
        check(cudaStreamDestroy(stream));
#ifdef ORIGINAL_CHECK
        std::puts("PASS original single-append behavior unchanged");
#endif
        std::puts("PASS all 83 cases: tail, dead, pooled, block_pos (memcmp)");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
