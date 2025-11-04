/**
 * @file common.h
 * @brief Shared constants, structs, and macros for the 2D FDTD simulation.
 */
#pragma once

#include <iostream>
#include <cmath>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \""                        \
                      << cudaGetErrorString(err) << "\"" << std::endl;   \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

namespace fdtd {

/**
 * @brief Parameters for the FDTD simulation.
 */
struct FDTDParams {
    int nx;             ///< Number of grid cells in x
    int ny;             ///< Number of grid cells in y
    int steps;          ///< Total number of time steps
    float dx;           ///< Spatial step size in x (meters)
    float dy;           ///< Spatial step size in y (meters)
    float dt;           ///< Time step size (seconds)
    float f0;           ///< Source frequency (Hz)
    int tfsf_low;       ///< Lower bound index for TF/SF contour
    int tfsf_high;      ///< Upper bound index for TF/SF contour
    
    // Derived values
    float eps0;
    float mu0;
    float c0;

    FDTDParams(int _nx = 512, int _ny = 512, int _steps = 2000) 
        : nx(_nx), ny(_ny), steps(_steps) {
        dx = 1e-3f;
        dy = 1e-3f;
        c0 = 299792458.0f; // Speed of light
        dt = dx / (std::sqrt(2.0f) * c0);
        f0 = 2.4e9f;
        tfsf_low = 50;
        tfsf_high = nx - 50;
        
        eps0 = 8.8541878128e-12f;
        mu0 = 4.0f * M_PI * 1e-7f;
    }
};

} // namespace fdtd
