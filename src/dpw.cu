/**
 * @file dpw.cu
 * @brief 1D DPW auxiliary grid kernel.
 */
#include "dpw.cuh"
#include <nvtx3/nvToolsExt.h>

namespace fdtd {
namespace dpw {

__global__ void k_dpw_update_h(int naux, float dt_mu0_dy, float* __restrict__ d_hx1d, const float* __restrict__ d_ez1d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < naux - 1) {
        d_hx1d[j] -= dt_mu0_dy * (d_ez1d[j + 1] - d_ez1d[j]);
    }
}

__global__ void k_dpw_update_e(int naux, float dt_eps0_dy, float* __restrict__ d_ez1d, const float* __restrict__ d_hx1d) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j > 0 && j < naux - 1) {
        d_ez1d[j] -= dt_eps0_dy * (d_hx1d[j] - d_hx1d[j - 1]);
    }
}

__global__ void k_dpw_abc_and_source(int naux, float cb, float dt, float f0, int step, float* __restrict__ d_ez1d, float* __restrict__ d_ez1d_prev) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float ez_0 = d_ez1d[0];
        float ez_1 = d_ez1d[1];
        float ez_n1 = d_ez1d[naux - 1];
        float ez_n2 = d_ez1d[naux - 2];
        
        d_ez1d[0] = d_ez1d_prev[0] + cb * (ez_1 - ez_0);
        d_ez1d[naux - 1] = d_ez1d_prev[1] + cb * (ez_n2 - ez_n1);
        
        d_ez1d_prev[0] = ez_1;
        d_ez1d_prev[1] = ez_n2;
        
        // Soft Source at index 5
        d_ez1d[5] += sinf(2.0f * (float)M_PI * f0 * step * dt);
    }
}

void launch_dpw_update(FDTDParams p, float* d_ez1d, float* d_hx1d, float* d_ez1d_prev, int naux, int step, cudaStream_t stream) {
    nvtxRangePushA("DPW_Update");
    int threads = 256;
    int blocks = (naux + threads - 1) / threads;
    float dt_mu0_dy = p.dt / (p.mu0 * p.dy);
    float dt_eps0_dy = p.dt / (p.eps0 * p.dy);
    float cb = (p.c0 * p.dt - p.dy) / (p.c0 * p.dt + p.dy);
    
    k_dpw_update_h<<<blocks, threads, 0, stream>>>(naux, dt_mu0_dy, d_hx1d, d_ez1d);
    k_dpw_update_e<<<blocks, threads, 0, stream>>>(naux, dt_eps0_dy, d_ez1d, d_hx1d);
    k_dpw_abc_and_source<<<1, 1, 0, stream>>>(naux, cb, p.dt, p.f0, step, d_ez1d, d_ez1d_prev);
    nvtxRangePop();
}

} // namespace dpw
} // namespace fdtd