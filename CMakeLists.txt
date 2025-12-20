cmake_minimum_required(VERSION 3.10)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PIAFS_VERSION)
string(STRIP "${PIAFS_VERSION}" PIAFS_VERSION)

# Options - define before project() so we can check them
option(ENABLE_SERIAL "Enable serial mode (no MPI)" OFF)
option(ENABLE_OMP "Enable OpenMP support" OFF)
option(ENABLE_GPU "Enable GPU support (CUDA/HIP)" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_HIP "Enable HIP support" OFF)

# Determine languages - add CUDA if enabled
# Note: HIP uses CXX compiler (hipcc) so no separate language needed
set(PROJECT_LANGUAGES C CXX)
if(ENABLE_GPU AND ENABLE_CUDA)
  list(APPEND PROJECT_LANGUAGES CUDA)
endif()
project(piafs VERSION ${PIAFS_VERSION} LANGUAGES ${PROJECT_LANGUAGES})

# Make version available to subdirectories
set_property(GLOBAL PROPERTY PIAFS_VERSION ${PIAFS_VERSION})

# Set default install prefix to source directory (like autotools)
# This matches autotools behavior: AC_PREFIX_DEFAULT('$(top_srcdir)')
# Override CMake's default (/usr/local) with source directory
# Check if user hasn't explicitly set a different prefix
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install path prefix" FORCE)
  message(STATUS "Setting default install prefix to source directory: ${CMAKE_INSTALL_PREFIX}")
elseif(CMAKE_INSTALL_PREFIX STREQUAL "/usr/local")
  # Also override if it's still the default /usr/local (in case the flag above didn't work)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}" CACHE PATH "Install path prefix" FORCE)
  message(STATUS "Overriding /usr/local default, setting install prefix to source directory: ${CMAKE_INSTALL_PREFIX}")
endif()

# Set C and C++ standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_SERIAL "Enable serial mode (no MPI)" OFF)
option(ENABLE_OMP "Enable OpenMP support" OFF)
option(ENABLE_GPU "Enable GPU support (CUDA/HIP)" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(ENABLE_HIP "Enable HIP support" OFF)
option(ENABLE_GPU_AWARE_MPI "Enable GPU-aware MPI (requires CUDA-aware or ROCm-aware MPI)" OFF)

# Enable CUDA language early if needed (before any subdirectories process CUDA files)
# Note: CUDA language is already enabled in project() if ENABLE_GPU and ENABLE_CUDA are set
if(ENABLE_GPU AND ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  message(STATUS "CUDA support enabled at project level")
  # Set CUDA architecture - default to sm_75 (Turing) or higher
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75" CACHE STRING "CUDA architectures to build for")
    message(STATUS "CUDA architecture not specified, defaulting to sm_75")
  else()
    message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()
endif()

# Enable HIP early if needed so headers (hip/hip_runtime.h) are discoverable for C/C++ sources
# that include gpu.h (many non-.cu files include it when GPU_HIP is defined).
if(ENABLE_GPU AND ENABLE_HIP)
  # For HIP builds, CMAKE_CXX_COMPILER should be hipcc or an MPI wrapper configured to use hipcc/rocmcc
  get_filename_component(CXX_COMPILER_NAME "${CMAKE_CXX_COMPILER}" NAME)
  
  # Check if it's hipcc, rocmcc, or an MPI wrapper
  if(CXX_COMPILER_NAME MATCHES "hipcc|rocmcc|mpicxx|mpiCC|mpic\\+\\+")
    message(STATUS "HIP-compatible C++ compiler detected: ${CMAKE_CXX_COMPILER}")
    # For MPI wrappers, verify they're using a HIP-compatible compiler
    if(CXX_COMPILER_NAME MATCHES "mpi")
      execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} --version
        OUTPUT_VARIABLE CXX_VERSION_OUTPUT
        ERROR_QUIET
      )
      if(CXX_VERSION_OUTPUT MATCHES "hipcc|rocmcc|clang.*HIP|amdclang")
        message(STATUS "MPI wrapper is using HIP-compatible compiler")
      else()
        message(WARNING "MPI C++ compiler wrapper may not be using hipcc/rocmcc as underlying compiler")
        message(WARNING "Compiler version output: ${CXX_VERSION_OUTPUT}")
        message(WARNING "If build fails, ensure MPI wrappers are configured with HIP-compatible compilers")
      endif()
    endif()
  else()
    message(WARNING "For HIP builds, CMAKE_CXX_COMPILER should be hipcc/rocmcc or MPI wrapper configured with HIP compiler")
    message(WARNING "Found: ${CMAKE_CXX_COMPILER}")
    message(WARNING "For non-MPI builds: cmake -DCMAKE_CXX_COMPILER=hipcc ...")
    message(WARNING "For MPI builds: Use MPI compiler wrapper configured with hipcc/rocmcc")
    message(FATAL_ERROR "HIP build requires HIP-compatible C++ compiler")
  endif()
  
  find_package(hip REQUIRED)
  message(STATUS "HIP support enabled at project level")
  message(STATUS "HIP compiler: ${CMAKE_CXX_COMPILER}")
  
  # HIP headers require exactly one platform macro.
  # Prefer HIP_PLATFORM from environment/CMake, otherwise default to AMD (ROCm).
  set(_piafs_hip_platform "amd")
  if(DEFINED HIP_PLATFORM)
    string(TOLOWER "${HIP_PLATFORM}" _piafs_hip_platform)
  elseif(DEFINED ENV{HIP_PLATFORM})
    string(TOLOWER "$ENV{HIP_PLATFORM}" _piafs_hip_platform)
  endif()
  if(_piafs_hip_platform MATCHES "nvidia")
    add_compile_definitions(__HIP_PLATFORM_NVIDIA__)
    message(STATUS "HIP platform: NVIDIA")
  else()
    add_compile_definitions(__HIP_PLATFORM_AMD__)
    message(STATUS "HIP platform: AMD")
  endif()
  
  # HIP include directories may be exposed under different variable names depending on ROCm/CMake.
  if(DEFINED HIP_INCLUDE_DIRS)
    include_directories(SYSTEM ${HIP_INCLUDE_DIRS})
    message(STATUS "HIP include dirs: ${HIP_INCLUDE_DIRS}")
  elseif(DEFINED hip_INCLUDE_DIRS)
    include_directories(SYSTEM ${hip_INCLUDE_DIRS})
    message(STATUS "HIP include dirs: ${hip_INCLUDE_DIRS}")
  elseif(DEFINED ENV{ROCM_PATH})
    include_directories(SYSTEM $ENV{ROCM_PATH}/include)
    message(STATUS "HIP include dirs (from ROCM_PATH): $ENV{ROCM_PATH}/include")
  endif()
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Configure MPI or serial mode
if(NOT ENABLE_SERIAL)
  find_package(MPI REQUIRED)
  if(MPI_C_FOUND)
    message(STATUS "Found MPI C compiler: ${MPI_C_COMPILER}")
    # Add MPI include directories globally for all C code
    include_directories(${MPI_C_INCLUDE_DIRS})
  else()
    message(WARNING "MPI C compiler not found, building in serial mode")
    add_compile_definitions(serial)
    set(ENABLE_SERIAL ON)
  endif()

  if(MPI_CXX_FOUND)
    message(STATUS "Found MPI C++ compiler: ${MPI_CXX_COMPILER}")
    # Add MPI include directories globally for all C++ code
    include_directories(${MPI_CXX_INCLUDE_DIRS})
    # For HIP, we need to use hipcc, not the MPI C++ compiler
    if(ENABLE_GPU AND ENABLE_HIP)
      # Try to find hipcc
      find_program(HIPCC_EXECUTABLE hipcc
        PATHS
        /opt/rocm/bin
        $ENV{ROCM_PATH}/bin
        $ENV{HIP_PATH}/bin
        PATH_SUFFIXES bin
      )
      if(HIPCC_EXECUTABLE)
        message(STATUS "HIP enabled: Using hipcc instead of MPI C++ compiler")
        message(STATUS "Found hipcc: ${HIPCC_EXECUTABLE}")
        # set(CMAKE_CXX_COMPILER ${HIPCC_EXECUTABLE} CACHE STRING "C++ compiler" FORCE)
        # When using hipcc, MPI_CXX_INCLUDE_DIRS might be empty
        # Use MPI_C_INCLUDE_DIRS as fallback, or query mpicxx for includes
        if(NOT MPI_CXX_INCLUDE_DIRS AND MPI_CXX_COMPILER)
          # Try to get include dirs from mpicxx -showme:compile
          execute_process(
            COMMAND ${MPI_CXX_COMPILER} -showme:compile
            OUTPUT_VARIABLE MPI_CXX_COMPILE_FLAGS
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
          )
          if(MPI_CXX_COMPILE_FLAGS)
            # Extract -I flags from compile flags
            string(REGEX MATCHALL "-I[^ ]+" MPI_CXX_INCLUDE_DIRS "${MPI_CXX_COMPILE_FLAGS}")
            string(REPLACE "-I" "" MPI_CXX_INCLUDE_DIRS "${MPI_CXX_INCLUDE_DIRS}")
            message(STATUS "Extracted MPI include dirs from mpicxx: ${MPI_CXX_INCLUDE_DIRS}")
          endif()
        endif()
      else()
        message(WARNING "hipcc not found, using MPI C++ compiler (HIP may not work)")
        # set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
      endif()
    else()
      # set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
    endif()
  else()
    message(WARNING "MPI C++ compiler not found, building in serial mode")
    add_compile_definitions(serial)
    set(ENABLE_SERIAL ON)
  endif()

  # Set MPIEXEC for tests - use user-specified value or detected executable
  # Only set if not already defined by user via -DMPIEXEC=...
  if(NOT MPIEXEC)
    if(MPIEXEC_EXECUTABLE)
      set(MPIEXEC "${MPIEXEC_EXECUTABLE}" CACHE STRING "MPI run command for tests")
    else()
      set(MPIEXEC "mpiexec" CACHE STRING "MPI run command for tests")
    endif()
  else()
    # User specified MPIEXEC, ensure it's in cache
    set(MPIEXEC "${MPIEXEC}" CACHE STRING "MPI run command for tests")
  endif()
else()
  message(STATUS "Building in serial mode (MPI disabled)")
  add_compile_definitions(serial)
  # For serial builds, default to mpiexec if not specified
  if(NOT MPIEXEC)
    set(MPIEXEC "mpiexec" CACHE STRING "MPI run command for tests")
  else()
    set(MPIEXEC "${MPIEXEC}" CACHE STRING "MPI run command for tests")
  endif()
endif()

# OpenMP support
if(ENABLE_OMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_C_FOUND)
    message(STATUS "OpenMP C support enabled")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  endif()
  if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP C++ support enabled")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_compile_definitions(with_omp)
  endif()
endif()

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-alloc-size-larger-than")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-alloc-size-larger-than")
endif()

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/physicalmodels)

# Add MPI include directories if MPI is enabled
# Try multiple methods to get MPI include directories
if(NOT ENABLE_SERIAL)
  set(MPI_INCLUDE_DIRS_GLOBAL "")
  
  # Method 1: Try different MPI compiler flags (OpenMPI, MPICH, Intel MPI, etc.)
  if(MPI_C_COMPILER)
    set(MPI_FLAGS_TO_TRY "-showme:compile" "-show" "-compile-info" "-showme")
    foreach(flag ${MPI_FLAGS_TO_TRY})
      execute_process(
        COMMAND ${MPI_C_COMPILER} ${flag}
        OUTPUT_VARIABLE MPI_C_COMPILE_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(MPI_C_COMPILE_FLAGS)
        string(REGEX MATCHALL "-I[^ ]+" MPI_C_INCLUDE_FLAGS "${MPI_C_COMPILE_FLAGS}")
        foreach(inc_flag ${MPI_C_INCLUDE_FLAGS})
          string(REPLACE "-I" "" inc_dir "${inc_flag}")
          list(APPEND MPI_INCLUDE_DIRS_GLOBAL "${inc_dir}")
        endforeach()
        break()  # Found working flag, stop trying
      endif()
    endforeach()
  endif()
  
  # Method 2: Use CMake's detected include dirs (most reliable)
  if(MPI_C_FOUND AND MPI_C_INCLUDE_DIRS)
    list(APPEND MPI_INCLUDE_DIRS_GLOBAL ${MPI_C_INCLUDE_DIRS})
  endif()
  if(MPI_CXX_FOUND AND MPI_CXX_INCLUDE_DIRS)
    list(APPEND MPI_INCLUDE_DIRS_GLOBAL ${MPI_CXX_INCLUDE_DIRS})
  endif()
  
  # Remove duplicates
  if(MPI_INCLUDE_DIRS_GLOBAL)
    list(REMOVE_DUPLICATES MPI_INCLUDE_DIRS_GLOBAL)
    # Add to global include directories
    include_directories(${MPI_INCLUDE_DIRS_GLOBAL})
    # Also add to compiler flags explicitly (critical for hipcc)
    foreach(inc_dir ${MPI_INCLUDE_DIRS_GLOBAL})
      set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -I${inc_dir}")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${inc_dir}")
    endforeach()
    message(STATUS "MPI include directories: ${MPI_INCLUDE_DIRS_GLOBAL}")
  else()
    message(WARNING "Could not find MPI include directories!")
    message(WARNING "MPI_C_INCLUDE_DIRS: ${MPI_C_INCLUDE_DIRS}")
    message(WARNING "MPI_CXX_INCLUDE_DIRS: ${MPI_CXX_INCLUDE_DIRS}")
  endif()
endif()

# Link with math library
link_libraries(m)

# Add subdirectories
add_subdirectory(src)

# Installation
# Only install the executable to bin/ (matching autotools minimal install)
# Headers, Examples, Extras, and docs are not installed by default

# Get git information
find_package(Git QUIET)
set(PIAFS_GIT_HASH "unknown")
set(PIAFS_GIT_BRANCH "unknown")
set(PIAFS_GIT_DIRTY "unknown")
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PIAFS_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PIAFS_GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  execute_process(
    COMMAND ${GIT_EXECUTABLE} status --porcelain
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE PIAFS_GIT_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(PIAFS_GIT_STATUS)
    set(PIAFS_GIT_DIRTY "yes")
  else()
    set(PIAFS_GIT_DIRTY "no")
  endif()
endif()

# Get build date
string(TIMESTAMP PIAFS_BUILD_DATE "%Y-%m-%d %H:%M:%S" UTC)

# Get compiler version
set(PIAFS_COMPILER_VERSION "unknown")
if(CMAKE_CXX_COMPILER_VERSION)
  set(PIAFS_COMPILER_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
endif()

# Get MPI version
set(PIAFS_MPI_VERSION "unknown")
if(NOT ENABLE_SERIAL)
  if(MPI_FOUND AND MPI_VERSION)
    set(PIAFS_MPI_VERSION "${MPI_VERSION}")
  elseif(MPI_FOUND AND MPI_C_COMPILER)
    # Try to extract MPI version from mpicc --version
    execute_process(
      COMMAND ${MPI_C_COMPILER} --version
      OUTPUT_VARIABLE MPI_VERSION_OUTPUT
      ERROR_VARIABLE MPI_VERSION_ERROR
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(MPI_VERSION_OUTPUT)
      # Look for version pattern like "version X.Y.Z" or "X.Y.Z"
      # For "mpicc for MVAPICH2 version 2.3.7", we want to match "version 2.3"
      # Try to match version after "version" keyword first (e.g., "version 2.3.7")
      # Use case-insensitive match and allow for various formats
      string(REGEX MATCH "[Vv]ersion[ ]+([0-9]+)\\.([0-9]+)" MPI_VERSION_MATCH "${MPI_VERSION_OUTPUT}")
      if(MPI_VERSION_MATCH)
        # Validate: major version should be reasonable (1-9 for common MPI versions)
        # Most MPI versions are 1.x, 2.x, 3.x, 4.x - allow up to 9
        if(CMAKE_MATCH_1 GREATER 0 AND CMAKE_MATCH_1 LESS 10)
          set(PIAFS_MPI_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
          set(MPI_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}" CACHE STRING "MPI version" FORCE)
        endif()
      else()
        # Fallback: match first X.Y pattern, but validate it looks like a version
        # Look for patterns that are likely version numbers (1-9.x)
        string(REGEX MATCH "([1-9])\\.([0-9]+)" MPI_VERSION_MATCH "${MPI_VERSION_OUTPUT}")
        if(MPI_VERSION_MATCH)
          set(PIAFS_MPI_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
          set(MPI_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}" CACHE STRING "MPI version" FORCE)
        endif()
      endif()
    endif()
  endif()
endif()

# Set build configuration strings
if(ENABLE_SERIAL)
  set(PIAFS_MPI_MODE "serial")
else()
  set(PIAFS_MPI_MODE "mpi")
endif()

if(ENABLE_OMP)
  set(PIAFS_OPENMP "enabled")
else()
  set(PIAFS_OPENMP "disabled")
endif()

# Get compiler name
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(PIAFS_COMPILER "gcc")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(PIAFS_COMPILER "clang")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  set(PIAFS_COMPILER "intel")
else()
  string(TOLOWER "${CMAKE_CXX_COMPILER_ID}" PIAFS_COMPILER)
endif()

# Generate build_info.h
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/build_info.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/build_info.h
  @ONLY
)

# Generate config.h for compatibility
# configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Testing support
enable_testing()

# Add unit tests
add_subdirectory(Tests/unit)

# Add regression tests (CMake-based, replaces test_mpi.sh)
add_subdirectory(Tests)

# Add a fixture to ensure binary is prepared before testing
add_test(
  NAME prepare_piafs_binary
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target prepare_test_binary
)
set_tests_properties(prepare_piafs_binary PROPERTIES
  FIXTURES_SETUP prepare_binary
)

# Display configuration summary
message(STATUS "")
message(STATUS "PIAFS Configuration Summary:")
message(STATUS "  Version: ${PIAFS_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Serial mode: ${ENABLE_SERIAL}")
message(STATUS "  OpenMP support: ${ENABLE_OMP}")
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    message(STATUS "  GPU support: CUDA")
  elseif(ENABLE_HIP)
    message(STATUS "  GPU support: HIP")
  endif()
  message(STATUS "  GPU-aware MPI: ${ENABLE_GPU_AWARE_MPI}")
else()
  message(STATUS "  GPU support: disabled")
endif()
message(STATUS "  C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "  CXX Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  Testing: Enabled (use 'make test' or 'ctest')")
message(STATUS "  MPI executor for tests: ${MPIEXEC}")
message(STATUS "")

# Create a custom target to copy PIAFS binary to bin/ before running tests
# This must be after src/CMakeLists.txt is processed so PIAFS_EXECUTABLE_NAME is available
if(EXISTS "${CMAKE_SOURCE_DIR}/src/CMakeLists.txt")
  # Get the executable name from src/CMakeLists.txt
  # We need to read it or set it here - let's set it here to match src/CMakeLists.txt logic
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
  
  # Add GPU info
  if(ENABLE_GPU)
    if(ENABLE_CUDA)
      set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-cuda")
    elseif(ENABLE_HIP)
      set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-hip")
    endif()
  endif()
  
  # Add build type (if not Release)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    string(TOLOWER "${CMAKE_BUILD_TYPE}" BUILD_TYPE_LOWER)
    set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-${BUILD_TYPE_LOWER}")
  endif()
  
  set(PIAFS_EXECUTABLE_NAME "PIAFS${PIAFS_NAME_SUFFIX}")
  
  add_custom_target(prepare_test_binary
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:${PIAFS_EXECUTABLE_NAME}> ${CMAKE_BINARY_DIR}/bin/${PIAFS_EXECUTABLE_NAME}
    DEPENDS ${PIAFS_EXECUTABLE_NAME}
    COMMENT "Copying ${PIAFS_EXECUTABLE_NAME} binary to bin/ for testing"
  )
endif()
