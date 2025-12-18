set(BC_SOURCES
  BCCleanup.c
  BCDirichlet.c
  BCExtrapolate.c
  BCInitialize.c
  BCNoslipWall.c
  BCPeriodic.c
  BCReflect.c
  BCSlipWall.c
  BCSponge.c
  BCSubsonicAmbivalent.c
  BCSubsonicInflow.c
  BCSubsonicOutflow.c
  BCSupersonicInflow.c
  BCSupersonicOutflow.c
)

if(ENABLE_GPU)
  if(ENABLE_CUDA)
    list(APPEND BC_SOURCES gpu_bc_kernels.cu)
    set_source_files_properties(gpu_bc_kernels.cu PROPERTIES
      LANGUAGE CUDA
      CUDA_SEPARABLE_COMPILATION ON
    )
  elseif(ENABLE_HIP)
    list(APPEND BC_SOURCES gpu_bc_kernels.cu)
    # Set -x hip flag to force HIP compilation (avoids CUDA library lookup)
    if(CMAKE_HIP_ARCHITECTURES)
      set(_bc_hip_flags "-x;hip;--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(_bc_hip_flags "-x;hip")
    endif()
    set_source_files_properties(gpu_bc_kernels.cu PROPERTIES
      LANGUAGE CXX
      COMPILE_OPTIONS "${_bc_hip_flags}"
    )
  endif()
endif()

add_library(BoundaryConditions STATIC ${BC_SOURCES})

target_include_directories(BoundaryConditions PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_include_directories(BoundaryConditions PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(BoundaryConditions PRIVATE CUDA::cudart)
    set_property(TARGET BoundaryConditions PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(BoundaryConditions PRIVATE GPU_CUDA)
  elseif(ENABLE_HIP)
    if(TARGET hip::device)
      target_link_libraries(BoundaryConditions PRIVATE hip::device)
    else()
      target_link_libraries(BoundaryConditions PRIVATE hip::amdhip64)
    endif()
    target_compile_definitions(BoundaryConditions PRIVATE GPU_HIP)
  endif()
endif()
