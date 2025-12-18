set(INTERPOLATIONFUNCTIONS_SOURCES
  Interp1PrimFifthOrderCRWENO.c
  Interp1PrimFifthOrderCRWENOChar.c
  Interp1PrimFifthOrderCompactUpwind.c
  Interp1PrimFifthOrderCompactUpwindChar.c
  Interp1PrimFifthOrderUpwind.c
  Interp1PrimFifthOrderUpwindChar.c
  Interp1PrimFifthOrderWENO.c
  Interp1PrimFifthOrderWENOChar.c
  Interp1PrimFirstOrderUpwind.c
  Interp1PrimFirstOrderUpwindChar.c
  Interp1PrimFourthOrderCentral.c
  Interp1PrimFourthOrderCentralChar.c
  Interp1PrimSecondOrderMUSCL.c
  Interp1PrimSecondOrderMUSCLChar.c
  Interp1PrimThirdOrderMUSCL.c
  Interp1PrimThirdOrderMUSCLChar.c
  Interp1PrimSecondOrderCentral.c
  Interp1PrimSecondOrderCentralChar.c
  CompactSchemeCleanup.c
  CompactSchemeInitialize.c
  MUSCLInitialize.c
  WENOCleanup.c
  WENOFifthOrderCalculateWeights.c
  WENOFifthOrderInitializeWeights.c
  WENOInitialize.c
  gpu_interpolation_function.c
  gpu_weno_weights_function.c
)

# GPU kernel files (compiled with GPU compiler)
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    list(APPEND INTERPOLATIONFUNCTIONS_SOURCES
      gpu_interpolation_launch.cu
      gpu_interpolation.cu
      gpu_weno_weights.cu
    )
    set_source_files_properties(
      gpu_interpolation_launch.cu
      gpu_interpolation.cu
      gpu_weno_weights.cu
      PROPERTIES
      LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
    )
  elseif(ENABLE_HIP)
    list(APPEND INTERPOLATIONFUNCTIONS_SOURCES
      gpu_interpolation_launch.cu
      gpu_interpolation.cu
      gpu_weno_weights.cu
    )
    # HIP: ensure .cu files are compiled as CXX with HIP flags
    # Use CMAKE_HIP_ARCHITECTURES if set, otherwise default to gfx90a
    if(CMAKE_HIP_ARCHITECTURES)
      set(_hip_arch_flags "-x;hip;--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(_hip_arch_flags "-x;hip")
    endif()
    set_source_files_properties(
      gpu_interpolation_launch.cu
      gpu_interpolation.cu
      gpu_weno_weights.cu
      PROPERTIES
      LANGUAGE CXX
      COMPILE_OPTIONS "${_hip_arch_flags}"
    )
  endif()
else()
  # CPU-only build: compile C fallback launch wrappers
  list(APPEND INTERPOLATIONFUNCTIONS_SOURCES
    gpu_interpolation_launch.c
  )
endif()

add_library(InterpolationFunctions STATIC ${INTERPOLATIONFUNCTIONS_SOURCES})

target_include_directories(InterpolationFunctions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Ensure GPU compile definitions propagate to this target (needed by C/C++ wrappers)
if(ENABLE_GPU AND ENABLE_CUDA)
  target_compile_definitions(InterpolationFunctions PRIVATE GPU_CUDA)
elseif(ENABLE_GPU AND ENABLE_HIP)
  target_compile_definitions(InterpolationFunctions PRIVATE GPU_HIP)
endif()

# CUDA headers are needed by C/C++ sources that include gpu.h (cuda_runtime.h).
if(ENABLE_GPU AND ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_include_directories(InterpolationFunctions PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(InterpolationFunctions PUBLIC CUDA::cudart)
endif()

# Enable device linking for CUDA separable compilation
if(ENABLE_GPU AND ENABLE_CUDA)
  set_property(TARGET InterpolationFunctions PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
