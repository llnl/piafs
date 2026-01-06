# GPU support library

# Check for GPU support
option(ENABLE_GPU "Enable GPU support (CUDA/HIP)" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_HIP "Enable HIP support" OFF)

if(ENABLE_GPU)
  if(ENABLE_CUDA)
    # CUDA support - language should already be enabled at project level
    if(NOT CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
    endif()
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA support enabled")
    add_compile_definitions(GPU_CUDA)
    set(GPU_COMPILER ${CMAKE_CUDA_COMPILER})
    set(GPU_LANGUAGE CUDA)
  elseif(ENABLE_HIP)
    # HIP support
    find_package(hip REQUIRED)
    message(STATUS "HIP support enabled")
    add_compile_definitions(GPU_HIP)
    set(GPU_COMPILER ${HIP_HIPCC_EXECUTABLE})
    set(GPU_LANGUAGE CXX)
  else()
    message(FATAL_ERROR "ENABLE_GPU requires either ENABLE_CUDA or ENABLE_HIP")
  endif()
else()
  message(STATUS "GPU support disabled")
  add_compile_definitions(GPU_NONE)
endif()

# GPU-aware MPI support (requires CUDA-aware or ROCm-aware MPI)
option(ENABLE_GPU_AWARE_MPI "Enable GPU-aware MPI" OFF)
if(ENABLE_GPU AND ENABLE_GPU_AWARE_MPI)
  message(STATUS "GPU-aware MPI enabled - MPI will use device pointers directly")
  add_compile_definitions(GPU_AWARE_MPI)
endif()

# GPU infrastructure source files (common utilities)
set(GPU_SOURCES
  gpu.c
  gpu_arrayfunctions.c
  gpu_config.c
  gpu_initialize.c
  gpu_io.c
  gpu_mpi.c
  gpu_time_integration.c
  gpu_runtime.c
)

# GPU kernel files (compiled with GPU compiler)
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    set(GPU_KERNEL_SOURCES
      gpu_kernels.cu
      gpu_kernels_optimized.cu
      gpu_launch.cu
      gpu_mpi_kernels.cu
    )
    set_source_files_properties(${GPU_KERNEL_SOURCES} PROPERTIES
      LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
    )
    list(APPEND GPU_SOURCES ${GPU_KERNEL_SOURCES})
  elseif(ENABLE_HIP)
    set(GPU_KERNEL_SOURCES
      gpu_kernels.cu
      gpu_kernels_optimized.cu
      gpu_launch.cu
      gpu_mpi_kernels.cu
    )
    # HIP: ensure .cu files are compiled as CXX with HIP flags
    if(CMAKE_HIP_ARCHITECTURES)
      set(_hip_arch_flags "-x;hip;--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(_hip_arch_flags "-x;hip")
    endif()
    set_source_files_properties(${GPU_KERNEL_SOURCES} PROPERTIES
      LANGUAGE CXX
      COMPILE_OPTIONS "${_hip_arch_flags}"
    )
    list(APPEND GPU_SOURCES ${GPU_KERNEL_SOURCES})
  endif()
else()
  # CPU fallback - build launch wrappers as C files (they have CPU fallback code)
  list(APPEND GPU_SOURCES 
    gpu_launch.c
    gpu_mpi_launch.c
  )
endif()

# Create GPU library
add_library(GPU STATIC ${GPU_SOURCES})

# Include directories
target_include_directories(GPU PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Link libraries
if(ENABLE_CUDA)
  target_link_libraries(GPU PUBLIC CUDA::cudart)
elseif(ENABLE_HIP)
  # HIP runtime target name varies across ROCm installs.
  if(TARGET hip::device)
    target_link_libraries(GPU PUBLIC hip::device)
  elseif(TARGET hip::amdhip64)
    target_link_libraries(GPU PUBLIC hip::amdhip64)
  endif()
endif()

# Compiler flags
# Note: CUDA architecture is set at project level in top-level CMakeLists.txt

