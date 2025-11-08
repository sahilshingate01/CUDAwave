/**
 * @file fdtd2d.cuh
 * @brief GPU FDTD declarations.
 */
#pragma once
#include "common.h"

namespace fdtd {
namespace gpu {

/**
 * @brief Launches kernel to update Hx and Hy fields.
 */
void launch_update_h(FDTDParams p, float* d_hx, float* d_hy, const float* d_ez, size_t pitch_bytes, cudaStream_t stream = 0);

/**
 * @brief Launches kernel to update Ez field.
 */
void launch_update_e(FDTDParams p, float* d_ez, const float* d_hx, const float* d_hy, size_t pitch_bytes, cudaStream_t stream = 0);

/**
 * @brief Launches kernel to save edges for ABC.
 */
void launch_save_edges(FDTDParams p, const float* d_ez, float* d_ez_prev, size_t pitch_bytes, cudaStream_t stream = 0);

/**
 * @brief Launches kernel to apply Mur ABC.
 */
void launch_apply_abc(FDTDParams p, float* d_ez, const float* d_ez_prev, size_t pitch_bytes, cudaStream_t stream = 0);

/**
 * @brief Launches kernel to apply soft source.
 */
void launch_apply_source(FDTDParams p, float* d_ez, size_t pitch_bytes, int step, cudaStream_t stream = 0);

} // namespace gpu
} // namespace fdtd
