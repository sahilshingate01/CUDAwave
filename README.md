# fdtd_gpu: CUDA-Accelerated 2D FDTD Simulation

## Overview
This project implements a 2D Finite-Difference Time-Domain (FDTD) electromagnetic simulation accelerated via CUDA. It serves as a proof-of-work for advanced GPU computational electromagnetics techniques, including TF/SF boundaries and a fully on-device 1D auxiliary grid. It is intended for researchers and students looking for a high-performance baseline.

## Physics Background
- **FDTD**: Solves Maxwell's equations in the time domain using the Yee scheme. Here, we simulate the 2D TE mode (Ez, Hx, Hy).
- **TF/SF (Total-Field / Scattered-Field)**: A boundary condition that injects an incident wave (e.g., plane wave) into a localized region without affecting the scattered fields outside.
- **1D DPW**: A Dispersive Plane Wave auxiliary grid is simulated alongside the 2D domain. It computes the exact incident fields for the TF/SF boundary to avoid numerical dispersion mismatches.

## Build Instructions
Prerequisites: CMake >= 3.18, CUDA Toolkit >= 11.0, GCC/Clang.
```bash
mkdir build && cd build
cmake ..
make
```

## Usage
Run the main executable with different modes:
```bash
# Run the full benchmark suite
./fdtd_gpu --mode bench

# Run GPU simulation with custom grid size
./fdtd_gpu --mode gpu --nx 512 --ny 512 --steps 2000

# Run CPU baseline
./fdtd_gpu --mode cpu
```

## Sample Output
- `data/benchmark.csv`: Raw timing data.
- `data/ez_field.png`: Heatmap of the Ez field snapshot, showing the propagating waves and the TF/SF boundary (white dashed line).
- `data/benchmark.png`: Bar chart comparing GPU vs. CPU execution times alongside a speedup line.

## Implementation Notes
- **Memory Layout**: Uses `cudaMallocPitch` for optimal coalesced memory access of 2D grids, avoiding bank conflicts.
- **Texture Memory**: The `Ez` field is bound to a 2D texture object (`cudaTextureObject_t`) during the TF/SF update to leverage the GPU's L2 texture cache for spatial locality.
- **Concurrency**: The 1D DPW update and the 2D H-field update are overlapped using distinct CUDA streams.
- **1D DPW On-Device**: Running the 1D auxiliary simulation entirely on the device avoids slow CPU-GPU memory transfers each time step.

## Performance Results
| Grid       | Steps | GPU(ms) | CPU(ms) | Speedup |
|------------|-------|---------|---------|---------|
| 256x256    | 2000  | XX.X    | XXX.X   | XX.Xx   |
| 512x512    | 2000  | XX.X    | XXX.X   | XX.Xx   |
| 1024x1024  | 2000  | XX.X    | XXX.X   | XX.Xx   |

*(Run `./fdtd_gpu --mode bench` to generate actual timings for your system)*

## Roadmap / Next Steps
- Implement PML (Perfectly Matched Layers) for superior boundary absorption.
- Extend the simulation to a full 3D domain.
- Add support for complex permittivity and frequency-dependent materials.
