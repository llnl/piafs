set(CHEMISTRY_SOURCES
  ChemistryCleanup.c
  ChemistryInitialize.c
  ChemistrySolve.c
  ChemistryWriteSpecies.c
)

# Add GPU sources if GPU is enabled
if(ENABLE_GPU)
  list(APPEND CHEMISTRY_SOURCES
    gpu_chemistry.c
  )
  
  if(ENABLE_CUDA)
    list(APPEND CHEMISTRY_SOURCES
      gpu_chemistry.cu
    )
    set_source_files_properties(
      gpu_chemistry.cu
      PROPERTIES LANGUAGE CUDA
    )
  elseif(ENABLE_HIP)
    # HIP: ensure .cu files are compiled as CXX with HIP flags
    list(APPEND CHEMISTRY_SOURCES
      gpu_chemistry.cu
    )
    # Set -x hip flag to force HIP compilation (avoids CUDA library lookup)
    if(CMAKE_HIP_ARCHITECTURES)
      set(_chemistry_hip_flags "-x;hip;--offload-arch=${CMAKE_HIP_ARCHITECTURES}")
    else()
      set(_chemistry_hip_flags "-x;hip")
    endif()
    set_source_files_properties(
      gpu_chemistry.cu
      PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS "${_chemistry_hip_flags}"
    )
  endif()
endif()

add_library(Chemistry STATIC ${CHEMISTRY_SOURCES})

target_include_directories(Chemistry PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/physicalmodels
)

# Add GPU compile definitions and link libraries
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_include_directories(Chemistry PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(Chemistry PRIVATE CUDA::cudart)
    set_property(TARGET Chemistry PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_compile_definitions(Chemistry PRIVATE GPU_CUDA)
  elseif(ENABLE_HIP)
    if(TARGET hip::device)
      target_link_libraries(Chemistry PRIVATE hip::device)
    else()
      target_link_libraries(Chemistry PRIVATE hip::amdhip64)
    endif()
    target_compile_definitions(Chemistry PRIVATE GPU_HIP)
  endif()
endif()
