cmake_minimum_required(VERSION 3.10)

# Read version from VERSION file
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" PIAFS_VERSION)
string(STRIP "${PIAFS_VERSION}" PIAFS_VERSION)

project(piafs VERSION ${PIAFS_VERSION} LANGUAGES C CXX)

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
    set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
  else()
    message(WARNING "MPI C compiler not found, building in serial mode")
    add_compile_definitions(serial)
    set(ENABLE_SERIAL ON)
  endif()

  if(MPI_CXX_FOUND)
    message(STATUS "Found MPI C++ compiler: ${MPI_CXX_COMPILER}")
    set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})
  else()
    message(WARNING "MPI C++ compiler not found, building in serial mode")
    add_compile_definitions(serial)
    set(ENABLE_SERIAL ON)
  endif()
else()
  message(STATUS "Building in serial mode (MPI disabled)")
  add_compile_definitions(serial)
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
install(DIRECTORY include/ DESTINATION include/piafs)
install(DIRECTORY Examples DESTINATION share/piafs)
install(DIRECTORY Extras DESTINATION share/piafs)
install(FILES README.md LICENSE DESTINATION share/doc/piafs)

# Generate config.h for compatibility
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

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
message(STATUS "")
