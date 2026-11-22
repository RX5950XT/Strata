// Windows diagnostic: identical real expert bytes, the installed build's kernels, cache versus RAM.
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <omp.h>
#include "strata/kernels/cpu/pool.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;
using Clock = std::chrono::steady_clock;
constexpr int jobs_per_batch = 4, banks = 64;

void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

void load(std::ifstream& file, uint64_t offset, void* dst, size_t bytes) {
    file.seekg(offset);
    file.read(static_cast<char*>(dst), bytes);
    require(bool(file), "GGUF read failed");
}

uint64_t read_bytes(const uint8_t* base, size_t bytes) {
    _ReadWriteBarrier();
    __m256i a = _mm256_setzero_si256(), b = a, c = a, d = a;
    for (size_t i = 0; i < bytes; i += 128) {
        a = _mm256_add_epi64(a, _mm256_loadu_si256((const __m256i*)(base+i)));
        b = _mm256_add_epi64(b, _mm256_loadu_si256((const __m256i*)(base+i+32)));
        c = _mm256_add_epi64(c, _mm256_loadu_si256((const __m256i*)(base+i+64)));
        d = _mm256_add_epi64(d, _mm256_loadu_si256((const __m256i*)(base+i+96)));
    }
    alignas(32) uint64_t v[4];
    _mm256_store_si256((__m256i*)v, _mm256_add_epi64(_mm256_add_epi64(a,b),_mm256_add_epi64(c,d)));
    return v[0]+v[1]+v[2]+v[3];
}

int main(int argc, char** argv) try {
    setvbuf(stdout, nullptr, _IONBF, 0);
    require(argc == 8, "usage: bench gguf mode threads tokens seconds gate_off up_off (down_off from layer0)");
    const std::string mode = argv[2];
    const int threads = std::stoi(argv[3]), nt = std::stoi(argv[4]);
    const double duration = std::stod(argv[5]);
    require((mode == "hot" || mode == "ram" || mode == "read" || mode == "paired") && threads>=1 && threads<=8 && nt>=1 && nt<=3 && duration>=0.1,
            "invalid benchmark parameters");
    require(SetPriorityClass(GetCurrentProcess(), ABOVE_NORMAL_PRIORITY_CLASS), "priority failed");
    require(SetThreadAffinityMask(GetCurrentThread(), 1) != 0, "host affinity failed");
    cpu::NativeFmt f;
    std::string error;
    if (!cpu::native_fmt(18, 20, cpu::H, cpu::FF, f, error)) throw std::runtime_error(error);
    require(f.bytes == 2176000, "pack geometry mismatch");
    const size_t batch_bytes = f.bytes * jobs_per_batch, total_bytes = batch_bytes * banks;
    auto storage = std::unique_ptr<uint8_t, decltype(&_aligned_free)>((uint8_t*)_aligned_malloc(total_bytes, 64), &_aligned_free);
    require(bool(storage), "allocation failed");
    std::ifstream file(argv[1], std::ios::binary);
    require(bool(file), "GGUF open failed");
    const uint64_t gu = std::stoull(argv[6]), up = std::stoull(argv[7]), down = 29548278272ull;
    for (int e=0; e<jobs_per_batch; ++e) {
        uint8_t* dst = storage.get()+e*f.bytes;
        load(file, gu+e*f.up_off, dst, f.up_off);
        load(file, up+e*f.up_off, dst+f.up_off, f.up_off);
        load(file, down+e*(f.bytes-f.down_off), dst+f.down_off, f.bytes-f.down_off);
    }
    for (int bank=1; bank<banks; ++bank) std::memcpy(storage.get()+bank*batch_bytes, storage.get(), batch_bytes);
    std::vector<uint8_t> act(nt * cpu::kNativeActBytes);
    std::vector<float> x(cpu::H), out(jobs_per_batch*nt*cpu::H), ff(nt*cpu::FF);
    std::vector<uint8_t> hq(nt*cpu::kNativeHBytes);
    for (int t=0; t<nt; ++t) {
        for (int i=0; i<cpu::H; ++i) x[i]=0.1f*std::sin(float(i+17*t));
        cpu::native_quant_act(f, x.data(), act.data()+t*cpu::kNativeActBytes);
    }
    std::vector<cpu::ExpertJobMulti> jobs(jobs_per_batch);
    for (int e=0; e<jobs_per_batch; ++e) {
        jobs[e].nt=nt;
        for (int t=0; t<nt; ++t) {
            jobs[e].nact[t]=act.data()+t*cpu::kNativeActBytes;
            jobs[e].out[t]=out.data()+(e*nt+t)*cpu::H;
        }
    }
    std::unique_ptr<cpu::ExpertPool> pool;
    if (threads>1 && mode!="read") pool=std::make_unique<cpu::ExpertPool>(threads-1, true, true);
    auto compute = [&](int bank) {
        for (int e=0; e<jobs_per_batch; ++e) jobs[e].blob=storage.get()+bank*batch_bytes+e*f.bytes;
        if (pool) { pool->run_split_multi_native(f,jobs.data(),jobs_per_batch); return; }
        const void* hp[3]; float* fp[3];
        for (int t=0;t<nt;++t) { hp[t]=hq.data()+t*cpu::kNativeHBytes; fp[t]=ff.data()+t*cpu::FF; }
        for (auto& job:jobs) {
            cpu::native_gu_rows(f,job.blob,job.nact,nt,fp,0,cpu::FF);
            for (int t=0;t<nt;++t) cpu::native_quant_h(f,fp[t],hq.data()+t*cpu::kNativeHBytes);
            cpu::native_down_rows(f,job.blob,hp,nt,job.out,0,cpu::H);
        }
    };
    uint64_t iterations=0, checksum=0;
    double seconds=0, gu_ms=0, q_ms=0, down_ms=0;
    if (mode=="read") {
        omp_set_num_threads(threads);
        std::vector<uint64_t> sums(threads);
        const size_t part=total_bytes/threads;
        require(part%128==0,"read alignment mismatch");
        const auto start=Clock::now();
        #pragma omp parallel
        {
            const int tid=omp_get_thread_num();
            require(SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1)<<(2*tid))!=0,"reader affinity failed");
            uint64_t count=0, sum=0;
            do { sum+=read_bytes(storage.get()+tid*part,part); ++count; }
            while (std::chrono::duration<double>(Clock::now()-start).count()<duration);
            sums[tid]=sum;
            #pragma omp atomic
            iterations+=count;
        }
        seconds=std::chrono::duration<double>(Clock::now()-start).count();
        for (auto sum:sums) checksum+=sum;
        std::printf("{\"mode\":\"read\",\"threads\":%d,\"bytes\":%zu,\"seconds\":%.9f,\"GB_s\":%.6f,\"checksum\":%llu}\n",
                    threads,total_bytes,seconds,(double)(iterations*part)/seconds/1e9,(unsigned long long)checksum);
        return 0;
    }
    compute(0); const auto expected=out;
    compute(banks-1);
    require(std::memcmp(expected.data(),out.data(),out.size()*sizeof(float))==0,"hot/RAM outputs differ");
    for (float value:out) require(std::isfinite(value),"nonfinite output");
    for (int warm=0; warm<banks; ++warm) compute(mode=="hot" ? 0 : warm);
    if (mode=="paired") {
        for (int pair=0; pair<8; ++pair) for (int arm=0; arm<2; ++arm) {
            const bool hot=(pair+arm)%2==0;
            for (int warm=0; warm<32; ++warm) compute(hot ? 0 : warm);
            const double gu0=pool?pool->ms_multi_gu:0, down0=pool?pool->ms_multi_down:0;
            const auto begin=Clock::now(); uint64_t n=0;
            do { compute(hot ? 0 : int(n%banks)); ++n; }
            while (std::chrono::duration<double>(Clock::now()-begin).count()<duration);
            const double elapsed=std::chrono::duration<double>(Clock::now()-begin).count();
            require(std::memcmp(expected.data(),out.data(),out.size()*sizeof(float))==0,"paired outputs differ");
            std::printf("{\"pair\":%d,\"mode\":\"%s\",\"threads\":%d,\"tokens\":%d,\"iterations\":%llu,\"batch_ms\":%.9f,\"gu_ms_per_batch\":%.9f,\"down_ms_per_batch\":%.9f,\"bitwise_equal\":true}\n",
                        pair,hot?"hot":"ram",threads,nt,(unsigned long long)n,elapsed*1000/n,
                        pool?(pool->ms_multi_gu-gu0)/n:0,pool?(pool->ms_multi_down-down0)/n:0);
        }
        return 0;
    }
    if (pool) { gu_ms=pool->ms_multi_gu; q_ms=pool->ms_multi_q; down_ms=pool->ms_multi_down; }
    const auto start=Clock::now();
    do { compute(mode=="hot" ? 0 : int(iterations%banks)); ++iterations; }
    while (std::chrono::duration<double>(Clock::now()-start).count()<duration);
    seconds=std::chrono::duration<double>(Clock::now()-start).count();
    if (pool) { gu_ms=pool->ms_multi_gu-gu_ms; q_ms=pool->ms_multi_q-q_ms; down_ms=pool->ms_multi_down-down_ms; }
    double result=0; for (float value:out) result+=value;
    std::printf("{\"mode\":\"%s\",\"threads\":%d,\"tokens\":%d,\"working_set_bytes\":%zu,\"batch_bytes\":%zu,\"iterations\":%llu,\"seconds\":%.9f,\"batch_ms\":%.9f,\"effective_GB_s\":%.6f,\"gu_ms_per_batch\":%.9f,\"quant_ms_per_batch\":%.9f,\"down_ms_per_batch\":%.9f,\"output_sum\":%.9f,\"bitwise_equal\":true}\n",
                mode.c_str(),threads,nt,mode=="hot"?batch_bytes:total_bytes,batch_bytes,(unsigned long long)iterations,seconds,
                seconds*1000/iterations,batch_bytes*iterations/seconds/1e9,gu_ms/iterations,q_ms/iterations,down_ms/iterations,result);
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr,"benchmark error: %s\n",error.what()); return 1; }
