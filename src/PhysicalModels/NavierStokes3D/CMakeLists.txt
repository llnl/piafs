set(NAVIERSTOKES3D_SOURCES
  NavierStokes3DCleanup.c
  NavierStokes3DComputeCFL.c
  NavierStokes3DEigen.c
  NavierStokes3DFlux.c
  NavierStokes3DFunctions.c
  NavierStokes3DInitialize.c
  NavierStokes3DParabolicFunction.c
  NavierStokes3DPreStep.c
  NavierStokes3DSource.c
  NavierStokes3DUpwind.c
  NavierStokes3DWriteChem.c
  gpu_parabolic_function.c
  gpu_flux_function.c
  gpu_source_function.c
  gpu_upwind_function.c
  gpu_cfl_function.c
)

# GPU kernel files (compiled with GPU compiler)
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    list(APPEND NAVIERSTOKES3D_SOURCES
      gpu_parabolic.cu
      gpu_parabolic_launch.cu
      gpu_flux.cu
      gpu_flux_launch.cu
      gpu_source.cu
      gpu_source_launch.cu
      gpu_upwind.cu
      gpu_upwind_launch.cu
      gpu_cfl.cu
      gpu_cfl_launch.cu
    )
    set_source_files_properties(
      gpu_parabolic.cu
      gpu_parabolic_launch.cu
      gpu_flux.cu
      gpu_flux_launch.cu
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
    list(APPEND NAVIERSTOKES3D_SOURCES
      gpu_parabolic.cu
      gpu_parabolic_launch.cu
      gpu_flux.cu
      gpu_flux_launch.cu
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
      gpu_parabolic.cu
      gpu_parabolic_launch.cu
      gpu_flux.cu
      gpu_flux_launch.cu
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
  list(APPEND NAVIERSTOKES3D_SOURCES
    gpu_parabolic_launch.c
    gpu_flux_launch.c
    gpu_source_launch.c
    gpu_upwind_launch.c
    gpu_cfl_launch.c
  )
endif()

add_library(NavierStokes3D STATIC ${NAVIERSTOKES3D_SOURCES})

target_include_directories(NavierStokes3D PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/physicalmodels
)

# Enable device linking for CUDA separable compilation
if(ENABLE_GPU AND ENABLE_CUDA)
  set_property(TARGET NavierStokes3D PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

# Add GPU compile definitions if GPU is enabled
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_compile_definitions(NavierStokes3D PRIVATE GPU_CUDA)
    # Add CUDA include directories so cuda_runtime.h can be found
    target_include_directories(NavierStokes3D PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    # Link CUDA runtime library for CUDA object files
    target_link_libraries(NavierStokes3D PUBLIC CUDA::cudart)
  elseif(ENABLE_HIP)
    target_compile_definitions(NavierStokes3D PRIVATE GPU_HIP)
    # HIP runtime target name varies across ROCm installs.
    if(TARGET hip::device)
      target_link_libraries(NavierStokes3D PUBLIC hip::device)
    elseif(TARGET hip::amdhip64)
      target_link_libraries(NavierStokes3D PUBLIC hip::amdhip64)
    endif()
  endif()
endif()
