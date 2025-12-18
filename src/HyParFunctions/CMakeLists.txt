set(HYPARFUNCTIONS_SOURCES
  ApplyBoundaryConditions.c
  BoundaryIntegral.c
  CalculateError.c
  CalculateConservationError.c
  ExactSolution.c
  HyperbolicFunction.c
  IncrementFilename.c
  NonLinearInterpolation.c
  SourceFunction.c
  VolumeIntegral.c
  gpu_hyperbolic_function.c
  gpu_first_derivative.c
)

# GPU kernel files (compiled with GPU compiler)
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    list(APPEND HYPARFUNCTIONS_SOURCES
      gpu_hyperbolic.cu
      gpu_derivative.cu
      gpu_hyperbolic_launch.cu
      gpu_derivative_launch.cu
    )
    set_source_files_properties(
      gpu_hyperbolic.cu
      gpu_derivative.cu
      gpu_hyperbolic_launch.cu
      gpu_derivative_launch.cu
      PROPERTIES
      LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
    )
  elseif(ENABLE_HIP)
    list(APPEND HYPARFUNCTIONS_SOURCES
      gpu_hyperbolic.cu
      gpu_derivative.cu
      gpu_hyperbolic_launch.cu
      gpu_derivative_launch.cu
    )
    # HIP: ensure .cu files are compiled as CXX with HIP flags
    if(CMAKE_HIP_ARCHITECTURES)
      set(_hip_arch_flags "-x;hip;--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(_hip_arch_flags "-x;hip")
    endif()
    set_source_files_properties(
      gpu_hyperbolic.cu
      gpu_derivative.cu
      gpu_hyperbolic_launch.cu
      gpu_derivative_launch.cu
      PROPERTIES
      LANGUAGE CXX
      COMPILE_OPTIONS "${_hip_arch_flags}"
    )
  endif()
else()
  # CPU fallback - use .c versions
  list(APPEND HYPARFUNCTIONS_SOURCES
    gpu_hyperbolic_launch.c
    gpu_derivative_launch.c
  )
endif()

add_library(HyParFunctions STATIC ${HYPARFUNCTIONS_SOURCES})

target_include_directories(HyParFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Add GPU compile definitions if GPU is enabled
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_compile_definitions(HyParFunctions PRIVATE GPU_CUDA)
    # Add CUDA include directories so cuda_runtime.h can be found
    target_include_directories(HyParFunctions PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    # Link CUDA runtime library for CUDA object files
    target_link_libraries(HyParFunctions PUBLIC CUDA::cudart)
  elseif(ENABLE_HIP)
    target_compile_definitions(HyParFunctions PRIVATE GPU_HIP)
    # HIP runtime target name varies across ROCm installs.
    # When building with hipcc as CXX, linking may also be handled by the compiler driver.
    if(TARGET hip::device)
      target_link_libraries(HyParFunctions PUBLIC hip::device)
    elseif(TARGET hip::amdhip64)
      target_link_libraries(HyParFunctions PUBLIC hip::amdhip64)
    endif()
  endif()
endif()
