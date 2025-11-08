/**
 * @file fdtd2d.cu
 * @brief GPU FDTD update kernels.
 */
#include "fdtd2d.cuh"
#include <nvtx3/nvToolsExt.h>
#include <algorithm>

namespace fdtd {
namespace gpu {

__global__ void k_update_h(int nx, int ny, float dt_mu0_dy, float dt_mu0_dx, float* __restrict__ d_hx, float* __restrict__ d_hy, const float* __restrict__ d_ez, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < nx && j < ny) {
        const float* ez_row = (const float*)((const char*)d_ez + i * pitch);
        const float* ez_row_next = (const float*)((const char*)d_ez + (i + 1) * pitch);
        
        if (j < ny - 1) {
            float* hx_row = (float*)((char*)d_hx + i * pitch);
            hx_row[j] -= dt_mu0_dy * (ez_row[j + 1] - ez_row[j]);
        }
        
        if (i < nx - 1) {
            float* hy_row = (float*)((char*)d_hy + i * pitch);
            hy_row[j] += dt_mu0_dx * (ez_row_next[j] - ez_row[j]);
        }
    }
}

__global__ void k_update_e(int nx, int ny, float dt_eps0_dx, float dt_eps0_dy, float* __restrict__ d_ez, const float* __restrict__ d_hx, const float* __restrict__ d_hy, size_t pitch) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        float* ez_row = (float*)((char*)d_ez + i * pitch);
        const float* hx_row = (const float*)((const char*)d_hx + i * pitch);
        const float* hy_row = (const float*)((const char*)d_hy + i * pitch);
        const float* hy_row_prev = (const float*)((const char*)d_hy + (i - 1) * pitch);
        
        ez_row[j] += dt_eps0_dx * (hy_row[j] - hy_row_prev[j])
                   - dt_eps0_dy * (hx_row[j] - hx_row[j - 1]);
    }
}

__global__ void k_save_edges(int nx, int ny, const float* __restrict__ d_ez, float* __restrict__ d_ez_prev, size_t pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ny) {
        d_ez_prev[idx] = *((const float*)((const char*)d_ez + 1 * pitch) + idx);
        d_ez_prev[ny + idx] = *((const float*)((const char*)d_ez + (nx - 2) * pitch) + idx);
    }
    if (idx < nx) {
        d_ez_prev[2 * ny + idx] = *((const float*)((const char*)d_ez + idx * pitch) + 1);
        d_ez_prev[2 * ny + nx + idx] = *((const float*)((const char*)d_ez + idx * pitch) + ny - 2);
    }
}

__global__ void k_apply_abc(int nx, int ny, float cb, float* __restrict__ d_ez, const float* __restrict__ d_ez_prev, size_t pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ny) {
        float* ez_left = (float*)((char*)d_ez + 0 * pitch);
        float* ez_left_1 = (float*)((char*)d_ez + 1 * pitch);
        float* ez_right = (float*)((char*)d_ez + (nx - 1) * pitch);
        float* ez_right_1 = (float*)((char*)d_ez + (nx - 2) * pitch);
        
        ez_left[idx] = d_ez_prev[idx] + cb * (ez_left_1[idx] - ez_left[idx]);
        ez_right[idx] = d_ez_prev[ny + idx] + cb * (ez_right_1[idx] - ez_right[idx]);
    }
    if (idx < nx) {
        float* ez_row = (float*)((char*)d_ez + idx * pitch);
        ez_row[0] = d_ez_prev[2 * ny + idx] + cb * (ez_row[1] - ez_row[0]);
        ez_row[ny - 1] = d_ez_prev[2 * ny + nx + idx] + cb * (ez_row[ny - 2] - ez_row[ny - 1]);
    }
}

__global__ void k_apply_source(int cx, int cy, float dt, float f0, int step, float* __restrict__ d_ez, size_t pitch) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float* ez_row = (float*)((char*)d_ez + cx * pitch);
        ez_row[cy] += sinf(2.0f * (float)M_PI * f0 * step * dt);
    }
}

void launch_update_h(FDTDParams p, float* d_hx, float* d_hy, const float* d_ez, size_t pitch_bytes, cudaStream_t stream) {
    nvtxRangePushA("Update_H");
    dim3 block(16, 16);
    dim3 grid((p.nx + block.x - 1) / block.x, (p.ny + block.y - 1) / block.y);
    float dt_mu0_dy = p.dt / (p.mu0 * p.dy);
    float dt_mu0_dx = p.dt / (p.mu0 * p.dx);
    k_update_h<<<grid, block, 0, stream>>>(p.nx, p.ny, dt_mu0_dy, dt_mu0_dx, d_hx, d_hy, d_ez, pitch_bytes);
    nvtxRangePop();
}

void launch_update_e(FDTDParams p, float* d_ez, const float* d_hx, const float* d_hy, size_t pitch_bytes, cudaStream_t stream) {
    nvtxRangePushA("Update_E");
    dim3 block(16, 16);
    dim3 grid((p.nx + block.x - 1) / block.x, (p.ny + block.y - 1) / block.y);
    float dt_eps0_dx = p.dt / (p.eps0 * p.dx);
    float dt_eps0_dy = p.dt / (p.eps0 * p.dy);
    k_update_e<<<grid, block, 0, stream>>>(p.nx, p.ny, dt_eps0_dx, dt_eps0_dy, d_ez, d_hx, d_hy, pitch_bytes);
    nvtxRangePop();
}

void launch_save_edges(FDTDParams p, const float* d_ez, float* d_ez_prev, size_t pitch_bytes, cudaStream_t stream) {
    nvtxRangePushA("Save_Edges");
    int max_dim = std::max(p.nx, p.ny);
    int threads = 256;
    int blocks = (max_dim + threads - 1) / threads;
    k_save_edges<<<blocks, threads, 0, stream>>>(p.nx, p.ny, d_ez, d_ez_prev, pitch_bytes);
    nvtxRangePop();
}

void launch_apply_abc(FDTDParams p, float* d_ez, const float* d_ez_prev, size_t pitch_bytes, cudaStream_t stream) {
    nvtxRangePushA("Apply_ABC");
    int max_dim = std::max(p.nx, p.ny);
    int threads = 256;
    int blocks = (max_dim + threads - 1) / threads;
    float cb = (p.c0 * p.dt - p.dx) / (p.c0 * p.dt + p.dx);
    k_apply_abc<<<blocks, threads, 0, stream>>>(p.nx, p.ny, cb, d_ez, d_ez_prev, pitch_bytes);
    nvtxRangePop();
}

void launch_apply_source(FDTDParams p, float* d_ez, size_t pitch_bytes, int step, cudaStream_t stream) {
    nvtxRangePushA("Apply_Source");
    k_apply_source<<<1, 1, 0, stream>>>(p.nx / 2, p.ny / 2, p.dt, p.f0, step, d_ez, pitch_bytes);
    nvtxRangePop();
}

} // namespace gpu
} // namespace fdtd