/**
 * @file tfsf.cuh
 * @brief TF/SF boundary injection declarations.
 */
#pragma once
#include "common.h"

namespace fdtd {
namespace gpu {

/**
 * @brief Launches kernels to apply TF/SF corrections.
 */
void launch_tfsf_h(FDTDParams p, float* d_hx, float* d_hy, const float* d_ez1d, size_t pitch_bytes, cudaTextureObject_t tex_ez, cudaStream_t stream = 0);
void launch_tfsf_e(FDTDParams p, float* d_ez, const float* d_hx1d, size_t pitch_bytes, cudaStream_t stream = 0);

} // namespace gpu
} // namespace fdtd