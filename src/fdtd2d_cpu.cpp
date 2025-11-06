/**
 * @file fdtd2d_cpu.cpp
 * @brief CPU FDTD reference implementation (single-threaded).
 */
#include "fdtd2d_cpu.h"
#include <vector>
#include <cmath>
#include <iostream>

namespace fdtd {
namespace cpu {

void run_cpu_fdtd(FDTDParams p, float* Ez_out) {
    int nx = p.nx;
    int ny = p.ny;
    
    std::vector<float> Hx(nx * ny, 0.0f);
    std::vector<float> Hy(nx * ny, 0.0f);
    std::vector<float> Ez(nx * ny, 0.0f);
    
    // Mur ABC variables
    std::vector<float> ez_left_prev(ny, 0.0f);
    std::vector<float> ez_right_prev(ny, 0.0f);
    std::vector<float> ez_top_prev(nx, 0.0f);
    std::vector<float> ez_bottom_prev(nx, 0.0f);
    
    float cb = (p.c0 * p.dt - p.dx) / (p.c0 * p.dt + p.dx);

    float dt_mu0_dy = p.dt / (p.mu0 * p.dy);
    float dt_mu0_dx = p.dt / (p.mu0 * p.dx);
    float dt_eps0_dx = p.dt / (p.eps0 * p.dx);
    float dt_eps0_dy = p.dt / (p.eps0 * p.dy);

    for (int n = 0; n < p.steps; ++n) {
        // Update Hx
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j < ny - 1; ++j) {
                Hx[i * ny + j] -= dt_mu0_dy * (Ez[i * ny + j + 1] - Ez[i * ny + j]);
            }
        }

        // Update Hy
        for (int i = 0; i < nx - 1; ++i) {
            for (int j = 0; j < ny; ++j) {
                Hy[i * ny + j] += dt_mu0_dx * (Ez[(i + 1) * ny + j] - Ez[i * ny + j]);
            }
        }

        // Mur ABC store previous boundary E fields
        for (int j = 0; j < ny; ++j) {
            ez_left_prev[j] = Ez[1 * ny + j];
            ez_right_prev[j] = Ez[(nx - 2) * ny + j];
        }
        for (int i = 0; i < nx; ++i) {
            ez_bottom_prev[i] = Ez[i * ny + 1];
            ez_top_prev[i] = Ez[i * ny + ny - 2];
        }

        // Update Ez
        for (int i = 1; i < nx - 1; ++i) {
            for (int j = 1; j < ny - 1; ++j) {
                Ez[i * ny + j] += dt_eps0_dx * (Hy[i * ny + j] - Hy[(i - 1) * ny + j])
                                - dt_eps0_dy * (Hx[i * ny + j] - Hx[i * ny + j - 1]);
            }
        }

        // Soft Source (center)
        Ez[(nx / 2) * ny + (ny / 2)] += std::sin(2.0f * M_PI * p.f0 * n * p.dt);

        // Mur ABC 1st order
        for (int j = 0; j < ny; ++j) {
            Ez[0 * ny + j] = ez_left_prev[j] + cb * (Ez[1 * ny + j] - Ez[0 * ny + j]);
            Ez[(nx - 1) * ny + j] = ez_right_prev[j] + cb * (Ez[(nx - 2) * ny + j] - Ez[(nx - 1) * ny + j]);
        }
        for (int i = 0; i < nx; ++i) {
            Ez[i * ny + 0] = ez_bottom_prev[i] + cb * (Ez[i * ny + 1] - Ez[i * ny + 0]);
            Ez[i * ny + ny - 1] = ez_top_prev[i] + cb * (Ez[i * ny + ny - 2] - Ez[i * ny + ny - 1]);
        }
    }

    // Copy to output
    for (int i = 0; i < nx * ny; ++i) {
        Ez_out[i] = Ez[i];
    }
}

} // namespace cpu
} // namespace fdtd
