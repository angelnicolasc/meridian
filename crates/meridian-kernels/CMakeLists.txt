cmake_minimum_required(VERSION 3.28)

project(meridian_kernels LANGUAGES C CXX CUDA)

# Conservative compute capability set. June-2026 fleet: Ada (8.9), Hopper (9.0),
# Blackwell (10.0). Older targets dropped because Meridian assumes bf16 +
# cooperative groups widely available.
set(CMAKE_CUDA_ARCHITECTURES "89;90;100")
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MERIDIAN_BUILD_CUDA "Build real CUDA kernels (otherwise stub)" OFF)

if(MERIDIAN_BUILD_CUDA)
    add_library(meridian_kernels_cuda STATIC
        cuda/entropy_probe.cu
        cuda/eat_kernel.cu
    )
    target_include_directories(meridian_kernels_cuda PUBLIC cuda)
    set_target_properties(meridian_kernels_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(meridian_kernels_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -Xptxas=-v -lineinfo>
    )

    install(TARGETS meridian_kernels_cuda
            ARCHIVE DESTINATION lib)
endif()
