// A GPU copy engine consumes host RAM bandwidth while CPU arithmetic stays unchanged.
#define NOMINMAX
#include <windows.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char** argv) {
    setvbuf(stdout,nullptr,_IONBF,0);
    if (argc!=3) return 2;
    const double seconds=std::atof(argv[1]);
    const int device=std::atoi(argv[2]);
    if (seconds<=0 || device<0) return 2;
    if (!SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<1)) return 1;
    if (cudaSetDevice(device)!=cudaSuccess || cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync)!=cudaSuccess) return 1;
    const size_t bytes=512ull<<20;
    void *host=nullptr,*gpu=nullptr;
    if (cudaHostAlloc(&host,bytes,cudaHostAllocDefault)!=cudaSuccess) return 1;
    if (cudaMalloc(&gpu,bytes)!=cudaSuccess) { cudaFreeHost(host); return 1; }
    std::memset(host,0x3f,bytes);
    std::printf("READY\n");
    const auto start=std::chrono::steady_clock::now();
    unsigned long long loops=0;
    do {
        if (cudaMemcpy(gpu,host,bytes,cudaMemcpyHostToDevice)!=cudaSuccess) return 1;
        ++loops;
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<seconds);
    const double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    std::printf("{\"seconds\":%.6f,\"host_read_GB_s\":%.6f,\"bytes\":%zu,\"copies\":%llu}\n",elapsed,loops*bytes/elapsed/1e9,bytes,loops);
    cudaFree(gpu); cudaFreeHost(host);
    return 0;
}
