# Add all subdirectories
add_subdirectory(ArrayFunctions)
add_subdirectory(BoundaryConditions)
add_subdirectory(CommonFunctions)
add_subdirectory(FirstDerivative)
add_subdirectory(HyParFunctions)
add_subdirectory(InterpolationFunctions)
add_subdirectory(IOFunctions)
add_subdirectory(LimiterFunctions)
add_subdirectory(MathFunctions)
add_subdirectory(MPIFunctions)
add_subdirectory(PhysicalModels)
add_subdirectory(Simulation)
add_subdirectory(TimeIntegration)
add_subdirectory(TridiagLU)

# Build binary name suffix from build configuration
set(PIAFS_NAME_SUFFIX "")

# Add compiler info
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-gcc")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-intel")
else()
  string(TOLOWER "${CMAKE_CXX_COMPILER_ID}" COMPILER_LOWER)
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-${COMPILER_LOWER}")
endif()

# Add MPI/serial info
if(ENABLE_SERIAL)
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-serial")
else()
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-mpi")
endif()

# Add OpenMP info
if(ENABLE_OMP)
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-omp")
endif()

# Add build type (if not Release)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
  string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
  set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-${BUILD_TYPE_LOWER}")
endif()

# Set the executable name
set(PIAFS_EXECUTABLE_NAME "PIAFS${PIAFS_NAME_SUFFIX}")
message(STATUS "Binary name: ${PIAFS_EXECUTABLE_NAME}")

# Main executable
add_executable(${PIAFS_EXECUTABLE_NAME} main.cpp)

# Include build_info.h
target_include_directories(${PIAFS_EXECUTABLE_NAME} PRIVATE ${CMAKE_BINARY_DIR})

# Link all libraries to the main executable
# Order matters for static libraries - dependencies should come after
target_link_libraries(${PIAFS_EXECUTABLE_NAME}
  Simulation
  HyParFunctions
  TimeIntegration
  BoundaryConditions
  Euler1D
  NavierStokes2D
  NavierStokes3D
  Chemistry
  InterpolationFunctions
  LimiterFunctions
  FirstDerivative
  TridiagLU
  IOFunctions
  MPIFunctions
  ArrayFunctions
  MathFunctions
  CommonFunctions
)

# If MPI is enabled, link MPI libraries
if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(${PIAFS_EXECUTABLE_NAME} MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(${PIAFS_EXECUTABLE_NAME} MPI::MPI_C)
  endif()
endif()

# If OpenMP is enabled, link OpenMP library
if(ENABLE_OMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${PIAFS_EXECUTABLE_NAME} OpenMP::OpenMP_CXX)
  endif()
endif()

# Install the executable to piafs/bin (matching autotools behavior)
install(TARGETS ${PIAFS_EXECUTABLE_NAME} DESTINATION bin)
