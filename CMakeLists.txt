cmake_minimum_required(VERSION 3.10)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PIAFS_VERSION)
string(STRIP "${PIAFS_VERSION}" PIAFS_VERSION)

project(piafs VERSION ${PIAFS_VERSION} LANGUAGES C CXX)

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
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
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
