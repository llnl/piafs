add_library(TimeIntegration STATIC
  TimeCleanup.c
  TimeInitialize.c
  TimeExplicitRKCleanup.c
  TimeExplicitRKInitialize.c
  TimePreStep.c
  TimePostStep.c
  TimePrintStep.c
  TimeRHSFunctionExplicit.c
  TimeStep.c
  TimeForwardEuler.c
  TimeRK.c
)

target_include_directories(TimeIntegration PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Add GPU compile definitions if GPU is enabled
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_compile_definitions(TimeIntegration PRIVATE GPU_CUDA)
    # Add CUDA include directories so cuda_runtime.h can be found
    target_include_directories(TimeIntegration PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  elseif(ENABLE_HIP)
    target_compile_definitions(TimeIntegration PRIVATE GPU_HIP)
    # HIP includes should be available via hip package
  endif()
endif()
