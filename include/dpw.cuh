/**
 * @file dpw.cuh
 * @brief 1D DPW auxiliary grid declarations.
 */
#pragma once
#include "common.h"

namespace fdtd {
namespace dpw {

/**
 * @brief Launches kernel to update 1D DPW grid.
 */
void launch_dpw_update(FDTDParams p, float* d_ez1d, float* d_hx1d, float* d_ez1d_prev, int naux, int step, cudaStream_t stream = 0);

} // namespace dpw
} // namespace fdtd