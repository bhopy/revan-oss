cmake_minimum_required(VERSION 3.18)
project(Revan LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA architecture: Turing (RTX 2060 Super) only
set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(CUDAToolkit REQUIRED)

add_executable(revan src/revan.cpp)

# Headers from bin_lib/include
target_include_directories(revan PRIVATE ${CMAKE_SOURCE_DIR}/bin_lib/include)

# Static libraries from bin_lib/lib
target_link_directories(revan PRIVATE ${CMAKE_SOURCE_DIR}/bin_lib/lib)

# Link order matters: llama depends on ggml, ggml depends on backends
target_link_libraries(revan PRIVATE
    llama
    common
    ggml
    ggml-base
    ggml-cpu
    ggml-cuda
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cuda_driver
)

# Windows-specific: need ws2_32 for some ggml internals, pathcch for path handling
if(WIN32)
    target_link_libraries(revan PRIVATE ws2_32 pathcch)
endif()
