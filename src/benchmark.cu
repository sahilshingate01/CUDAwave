/**
 * @file benchmark.cu
 * @brief Timing harness, CSV output.
 */
#include "common.h"
#include "fdtd2d_cpu.h"
#include "fdtd2d.cuh"
#include "dpw.cuh"
#include "tfsf.cuh"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>

namespace fdtd {

void run_gpu_simulation(FDTDParams p, float* h_ez_out, float& time_ms) {
    size_t pitch;
    float *d_hx, *d_hy, *d_ez, *d_ez_prev;
    CHECK_CUDA(cudaMallocPitch(&d_hx, &pitch, p.ny * sizeof(float), p.nx));
    CHECK_CUDA(cudaMallocPitch(&d_hy, &pitch, p.ny * sizeof(float), p.nx));
    CHECK_CUDA(cudaMallocPitch(&d_ez, &pitch, p.ny * sizeof(float), p.nx));
    CHECK_CUDA(cudaMemset2D(d_hx, pitch, 0, p.ny * sizeof(float), p.nx));
    CHECK_CUDA(cudaMemset2D(d_hy, pitch, 0, p.ny * sizeof(float), p.nx));
    CHECK_CUDA(cudaMemset2D(d_ez, pitch, 0, p.ny * sizeof(float), p.nx));
    
    CHECK_CUDA(cudaMalloc(&d_ez_prev, 2 * (p.nx + p.ny) * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_ez_prev, 0, 2 * (p.nx + p.ny) * sizeof(float)));

    int naux = p.ny + 2 * p.tfsf_low + 10;
    float *d_ez1d, *d_hx1d, *d_ez1d_prev;
    CHECK_CUDA(cudaMalloc(&d_ez1d, naux * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_hx1d, naux * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_ez1d_prev, 2 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_ez1d, 0, naux * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_hx1d, 0, naux * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_ez1d_prev, 0, 2 * sizeof(float)));

    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_ez;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
    resDesc.res.pitch2D.width = p.ny;
    resDesc.res.pitch2D.height = p.nx;
    resDesc.res.pitch2D.pitchInBytes = pitch;

    cudaTextureDesc texDesc = {};
    texDesc.readMode = cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex_ez = 0;
    CHECK_CUDA(cudaCreateTextureObject(&tex_ez, &resDesc, &texDesc));

    cudaStream_t stream_main, stream_dpw;
    CHECK_CUDA(cudaStreamCreate(&stream_main));
    CHECK_CUDA(cudaStreamCreate(&stream_dpw));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Warm-up
    for (int n = 0; n < 10; ++n) {
        gpu::launch_update_h(p, d_hx, d_hy, d_ez, pitch, stream_main);
        dpw::launch_dpw_update(p, d_ez1d, d_hx1d, d_ez1d_prev, naux, n, stream_dpw);
        CHECK_CUDA(cudaDeviceSynchronize());
        gpu::launch_tfsf_h(p, d_hx, d_hy, d_ez1d, pitch, tex_ez, stream_main);
        
        gpu::launch_save_edges(p, d_ez, d_ez_prev, pitch, stream_main);
        gpu::launch_update_e(p, d_ez, d_hx, d_hy, pitch, stream_main);
        gpu::launch_tfsf_e(p, d_ez, d_hx1d, pitch, stream_main);
        gpu::launch_apply_source(p, d_ez, pitch, n, stream_main);
        gpu::launch_apply_abc(p, d_ez, d_ez_prev, pitch, stream_main);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Main run
    CHECK_CUDA(cudaEventRecord(start, stream_main));
    for (int n = 0; n < p.steps; ++n) {
        gpu::launch_update_h(p, d_hx, d_hy, d_ez, pitch, stream_main);
        dpw::launch_dpw_update(p, d_ez1d, d_hx1d, d_ez1d_prev, naux, n, stream_dpw);
        CHECK_CUDA(cudaDeviceSynchronize()); // wait for DPW to finish before TF/SF
        gpu::launch_tfsf_h(p, d_hx, d_hy, d_ez1d, pitch, tex_ez, stream_main);
        
        gpu::launch_save_edges(p, d_ez, d_ez_prev, pitch, stream_main);
        gpu::launch_update_e(p, d_ez, d_hx, d_hy, pitch, stream_main);
        gpu::launch_tfsf_e(p, d_ez, d_hx1d, pitch, stream_main);
        gpu::launch_apply_source(p, d_ez, pitch, n, stream_main);
        gpu::launch_apply_abc(p, d_ez, d_ez_prev, pitch, stream_main);
    }
    CHECK_CUDA(cudaEventRecord(stop, stream_main));
    CHECK_CUDA(cudaEventSynchronize(stop));

    CHECK_CUDA(cudaEventElapsedTime(&time_ms, start, stop));

    if (h_ez_out) {
        CHECK_CUDA(cudaMemcpy2D(h_ez_out, p.ny * sizeof(float), d_ez, pitch, p.ny * sizeof(float), p.nx, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaDestroyTextureObject(tex_ez));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaStreamDestroy(stream_main));
    CHECK_CUDA(cudaStreamDestroy(stream_dpw));
    CHECK_CUDA(cudaFree(d_hx));
    CHECK_CUDA(cudaFree(d_hy));
    CHECK_CUDA(cudaFree(d_ez));
    CHECK_CUDA(cudaFree(d_ez_prev));
    CHECK_CUDA(cudaFree(d_ez1d));
    CHECK_CUDA(cudaFree(d_hx1d));
    CHECK_CUDA(cudaFree(d_ez1d_prev));
}

void run_benchmark(FDTDParams default_p) {
    std::vector<int> sizes = {256, 512, 1024};
    
    std::ofstream csv("data/benchmark.csv");
    csv << "grid,steps,gpu_ms,cpu_ms,speedup\n";

    std::cout << std::left << std::setw(12) << "Grid" 
              << std::setw(10) << "Steps"
              << std::setw(12) << "GPU(ms)"
              << std::setw(12) << "CPU(ms)"
              << "Speedup" << std::endl;

    for (int size : sizes) {
        FDTDParams p = default_p;
        p.nx = size;
        p.ny = size;
        p.tfsf_low = 50;
        p.tfsf_high = size - 50;

        float gpu_ms = 0.0f;
        std::vector<float> ez_out(p.nx * p.ny);
        run_gpu_simulation(p, ez_out.data(), gpu_ms);

        if (size == 512) {
            std::ofstream bin("data/ez_final.bin", std::ios::binary);
            bin.write(reinterpret_cast<char*>(ez_out.data()), ez_out.size() * sizeof(float));
        }

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<float> cpu_ez_out(p.nx * p.ny);
        cpu::run_cpu_fdtd(p, cpu_ez_out.data());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> cpu_duration = end - start;
        float cpu_ms = cpu_duration.count();

        float speedup = cpu_ms / gpu_ms;

        std::cout << std::left << std::setw(12) << (std::to_string(size) + "x" + std::to_string(size))
                  << std::setw(10) << p.steps
                  << std::setw(12) << std::fixed << std::setprecision(1) << gpu_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << cpu_ms
                  << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

        csv << size << "x" << size << "," << p.steps << "," << gpu_ms << "," << cpu_ms << "," << speedup << "\n";
    }
}

} // namespace fdtd