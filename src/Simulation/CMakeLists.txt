add_library(Simulation STATIC
  Cleanup.c
  EnsembleSimulationsDefine.cpp
  Initialize.c
  InitializeBoundaries.c
  InitializePhysics.c
  InitializePhysicsData.c
  InitializeSolvers.c
  InitialSolution.c
  OutputSolution.cpp
  ReadInputs.c
  SimulationWriteErrors.c
  SingleSimulationDefine.cpp
  Solve.cpp
  WriteInputs.c
)

# Ensure .cpp files are compiled as C++, not HIP
# hipcc should handle .cpp files as regular C++, but we explicitly set language
if(ENABLE_GPU AND ENABLE_HIP)
  set_source_files_properties(
    EnsembleSimulationsDefine.cpp
    OutputSolution.cpp
    SingleSimulationDefine.cpp
    Solve.cpp
    PROPERTIES
    LANGUAGE CXX
    COMPILE_FLAGS "-x c++"
  )
endif()

target_include_directories(Simulation PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# Add GPU compile definitions if GPU is enabled
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_compile_definitions(Simulation PRIVATE GPU_CUDA)
    # Add CUDA include directories so cuda_runtime.h can be found
    target_include_directories(Simulation PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  elseif(ENABLE_HIP)
    target_compile_definitions(Simulation PRIVATE GPU_HIP)
    # HIP includes should be available via hip package
  endif()
endif()

# Add MPI include directories to Simulation target
# The root CMakeLists.txt already added MPI includes globally, but we also add them
# explicitly to this target to ensure they're in the compile command
if(NOT ENABLE_SERIAL)
  # Use CMake's detected MPI include directories (already found by find_package(MPI))
  set(MPI_INCLUDE_DIRS "")
  
  if(MPI_C_FOUND AND MPI_C_INCLUDE_DIRS)
    list(APPEND MPI_INCLUDE_DIRS ${MPI_C_INCLUDE_DIRS})
  endif()
  if(MPI_CXX_FOUND AND MPI_CXX_INCLUDE_DIRS)
    list(APPEND MPI_INCLUDE_DIRS ${MPI_CXX_INCLUDE_DIRS})
  endif()
  
  # Remove duplicates and add to target
  if(MPI_INCLUDE_DIRS)
    list(REMOVE_DUPLICATES MPI_INCLUDE_DIRS)
    foreach(inc_dir ${MPI_INCLUDE_DIRS})
      target_compile_options(Simulation PUBLIC -I${inc_dir})
    endforeach()
  endif()
endif()
