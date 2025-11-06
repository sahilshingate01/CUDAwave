/**
 * @file fdtd2d_cpu.h
 * @brief CPU FDTD declarations.
 */
#pragma once
#include "common.h"

namespace fdtd {
namespace cpu {

/**
 * @brief Runs the CPU FDTD reference implementation.
 * @param p FDTD simulation parameters.
 * @param Ez_out Pointer to pre-allocated host array (size nx * ny) for final Ez snapshot.
 */
void run_cpu_fdtd(FDTDParams p, float* Ez_out);

} // namespace cpu
} // namespace fdtd
