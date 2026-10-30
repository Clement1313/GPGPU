cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(project_irgpua LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

find_package(OpenMP REQUIRED)

# This example uses the CMake Package Manager (CPM) to simplify fetching CCCL from GitHub
# For more information, see https://github.com/cpm-cmake/CPM.cmake
include(cmake/CPM.cmake)

CPMAddPackage(
        NAME CCCL
        GITHUB_REPOSITORY "nvidia/cccl"
        GIT_TAG "v3.5.0.dev"
        )

CPMAddPackage(
    NAME NVTX
    GITHUB_REPOSITORY NVIDIA/NVTX
    GIT_TAG v3.3.0-c-cpp
    GIT_SHALLOW TRUE)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
# Set the C++ standard
set(CXX_STANDARD 17)
# Add fconcept for lambdas
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
# Generate cubin for this specific GPU
set(CMAKE_CUDA_ARCHITECTURES "native")

include_directories(src)

add_executable(main src/main.cu src/fix_cpu.cu)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main PRIVATE OpenMP::OpenMP_CXX CCCL::CCCL nvtx3-cpp)
