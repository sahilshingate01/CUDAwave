cmake_minimum_required(VERSION 3.18)
project(fdtd_gpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA configurations
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")

set_target_properties(fdtd_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(fdtd_gpu
    src/main.cu
    src/fdtd2d.cu
    src/fdtd2d_cpu.cpp
    src/dpw.cu
    src/tfsf.cu
    src/benchmark.cu
)

target_include_directories(fdtd_gpu PRIVATE include)
target_link_libraries(fdtd_gpu PRIVATE cudart)

# Ensure set_target_properties is actually applied correctly to the target
set_target_properties(fdtd_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_custom_target(run_bench
    COMMAND ./fdtd_gpu --mode bench
    DEPENDS fdtd_gpu
    COMMENT "Running GPU vs CPU benchmark"
)