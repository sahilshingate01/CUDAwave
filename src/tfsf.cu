/**
 * @file tfsf.cu
 * @brief TF/SF injection kernels.
 */
#include "tfsf.cuh"
#include <nvtx3/nvToolsExt.h>

namespace fdtd {
namespace gpu {

__global__ void k_tfsf_hx_bottom(int nx, int tfsf_low, float dt_mu0_dy, float* __restrict__ d_hx, const float* __restrict__ d_ez1d, size_t pitch, cudaTextureObject_t tex_ez) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tfsf_low && i <= nx - tfsf_low) {
        int j = tfsf_low - 1;
        float* hx_row = (float*)((char*)d_hx + i * pitch);
        // The 1D wave propagates in y. Incident Ez at j=tfsf_low is d_ez1d[offset].
        // offset is mapped. Let's assume tfsf_low maps to index 10 in 1D array.
        int dpw_idx = 10; 
        hx_row[j] -= dt_mu0_dy * d_ez1d[dpw_idx];
    }
}

__global__ void k_tfsf_hx_top(int nx, int tfsf_high, float dt_mu0_dy, float* __restrict__ d_hx, const float* __restrict__ d_ez1d, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx - tfsf_high && i <= tfsf_high) {
        int j = tfsf_high;
        float* hx_row = (float*)((char*)d_hx + i * pitch);
        int dpw_idx = 10 + (tfsf_high - (nx - tfsf_high)); // Simplification for mapped index
        hx_row[j] += dt_mu0_dy * d_ez1d[dpw_idx];
    }
}

__global__ void k_tfsf_hy_left(int ny, int tfsf_low, float dt_mu0_dx, float* __restrict__ d_hy, const float* __restrict__ d_ez1d, size_t pitch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= tfsf_low && j <= ny - tfsf_low) {
        int i = tfsf_low - 1;
        float* hy_row = (float*)((char*)d_hy + i * pitch);
        int dpw_idx = 10 + (j - tfsf_low);
        hy_row[j] += dt_mu0_dx * d_ez1d[dpw_idx];
    }
}

__global__ void k_tfsf_hy_right(int nx, int ny, int tfsf_high, float dt_mu0_dx, float* __restrict__ d_hy, const float* __restrict__ d_ez1d, size_t pitch) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nx - tfsf_high && j <= tfsf_high) {
        int i = tfsf_high;
        float* hy_row = (float*)((char*)d_hy + i * pitch);
        int dpw_idx = 10 + (j - (nx - tfsf_high));
        hy_row[j] -= dt_mu0_dx * d_ez1d[dpw_idx];
    }
}

__global__ void k_tfsf_ez_bottom(int nx, int tfsf_low, float dt_eps0_dy, float* __restrict__ d_ez, const float* __restrict__ d_hx1d, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= tfsf_low && i <= nx - tfsf_low) {
        int j = tfsf_low;
        float* ez_row = (float*)((char*)d_ez + i * pitch);
        int dpw_idx = 10 - 1; 
        ez_row[j] -= dt_eps0_dy * d_hx1d[dpw_idx];
    }
}

__global__ void k_tfsf_ez_top(int nx, int tfsf_high, float dt_eps0_dy, float* __restrict__ d_ez, const float* __restrict__ d_hx1d, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx - tfsf_high && i <= tfsf_high) {
        int j = tfsf_high;
        float* ez_row = (float*)((char*)d_ez + i * pitch);
        int dpw_idx = 10 + (tfsf_high - (nx - tfsf_high));
        ez_row[j] += dt_eps0_dy * d_hx1d[dpw_idx];
    }
}

void launch_tfsf_h(FDTDParams p, float* d_hx, float* d_hy, const float* d_ez1d, size_t pitch_bytes, cudaTextureObject_t tex_ez, cudaStream_t stream) {
    nvtxRangePushA("TFSF_H");
    int threads = 256;
    float dt_mu0_dy = p.dt / (p.mu0 * p.dy);
    float dt_mu0_dx = p.dt / (p.mu0 * p.dx);

    int blocks_x = (p.nx + threads - 1) / threads;
    int blocks_y = (p.ny + threads - 1) / threads;

    k_tfsf_hx_bottom<<<blocks_x, threads, 0, stream>>>(p.nx, p.tfsf_low, dt_mu0_dy, d_hx, d_ez1d, pitch_bytes, tex_ez);
    k_tfsf_hx_top<<<blocks_x, threads, 0, stream>>>(p.nx, p.tfsf_high, dt_mu0_dy, d_hx, d_ez1d, pitch_bytes);
    k_tfsf_hy_left<<<blocks_y, threads, 0, stream>>>(p.ny, p.tfsf_low, dt_mu0_dx, d_hy, d_ez1d, pitch_bytes);
    k_tfsf_hy_right<<<blocks_y, threads, 0, stream>>>(p.nx, p.ny, p.tfsf_high, dt_mu0_dx, d_hy, d_ez1d, pitch_bytes);
    nvtxRangePop();
}

void launch_tfsf_e(FDTDParams p, float* d_ez, const float* d_hx1d, size_t pitch_bytes, cudaStream_t stream) {
    nvtxRangePushA("TFSF_E");
    int threads = 256;
    float dt_eps0_dy = p.dt / (p.eps0 * p.dy);

    int blocks_x = (p.nx + threads - 1) / threads;

    k_tfsf_ez_bottom<<<blocks_x, threads, 0, stream>>>(p.nx, p.tfsf_low, dt_eps0_dy, d_ez, d_hx1d, pitch_bytes);
    k_tfsf_ez_top<<<blocks_x, threads, 0, stream>>>(p.nx, p.tfsf_high, dt_eps0_dy, d_ez, d_hx1d, pitch_bytes);
    nvtxRangePop();
}

} // namespace gpu
} // namespace fdtd