set(EULER1D_SOURCES
  Euler1DCleanup.c
  Euler1DComputeCFL.c
  Euler1DEigen.c
  Euler1DFlux.c
  Euler1DFunctions.c
  Euler1DInitialize.c
  Euler1DPreStep.c
  Euler1DSource.c
  Euler1DUpwind.c
  Euler1DWriteChem.c
  gpu_flux_function.c
  gpu_source_function.c
  gpu_upwind_function.c
  gpu_cfl_function.c
)

# GPU kernel files (compiled with GPU compiler)
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    list(APPEND EULER1D_SOURCES
      gpu_flux.cu
      gpu_source.cu
      gpu_source_launch.cu
      gpu_upwind.cu
      gpu_upwind_launch.cu
      gpu_cfl.cu
      gpu_cfl_launch.cu
    )
    set_source_files_properties(
      gpu_flux.cu
      gpu_source.cu
      gpu_source_launch.cu
      gpu_upwind.cu
      gpu_upwind_launch.cu
      gpu_cfl.cu
      gpu_cfl_launch.cu
      PROPERTIES
      LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
    )
  elseif(ENABLE_HIP)
    list(APPEND EULER1D_SOURCES
      gpu_flux.cu
      gpu_source.cu
      gpu_source_launch.cu
      gpu_upwind.cu
      gpu_upwind_launch.cu
      gpu_cfl.cu
      gpu_cfl_launch.cu
    )
    # HIP: ensure .cu files are compiled as CXX with HIP flags
    if(CMAKE_HIP_ARCHITECTURES)
      set(_hip_arch_flags "-x;hip;--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(_hip_arch_flags "-x;hip")
    endif()
    set_source_files_properties(
      gpu_flux.cu
      gpu_source.cu
      gpu_source_launch.cu
      gpu_upwind.cu
      gpu_upwind_launch.cu
      gpu_cfl.cu
      gpu_cfl_launch.cu
      PROPERTIES
      LANGUAGE CXX
      COMPILE_OPTIONS "${_hip_arch_flags}"
    )
  endif()
else()
  # CPU fallback - use .c version
  list(APPEND EULER1D_SOURCES
    gpu_flux_launch.c
    gpu_source_launch.c
    gpu_upwind_launch.c
    gpu_cfl_launch.c
  )
endif()

add_library(Euler1D STATIC ${EULER1D_SOURCES})

target_include_directories(Euler1D PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/physicalmodels
)

# Enable device linking for CUDA separable compilation
if(ENABLE_GPU AND ENABLE_CUDA)
  set_property(TARGET Euler1D PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Add GPU compile definitions if GPU is enabled
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_compile_definitions(Euler1D PRIVATE GPU_CUDA)
    if(CUDAToolkit_INCLUDE_DIRS)
      target_include_directories(Euler1D PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()
    target_link_libraries(Euler1D PUBLIC CUDA::cudart)
  elseif(ENABLE_HIP)
    target_compile_definitions(Euler1D PRIVATE GPU_HIP)
  endif()
endif()
