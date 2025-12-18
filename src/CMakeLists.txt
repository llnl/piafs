# Add all subdirectories
add_subdirectory(ArrayFunctions)
add_subdirectory(BoundaryConditions)
add_subdirectory(CommonFunctions)
add_subdirectory(FirstDerivative)
add_subdirectory(GPU)
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

# Set the executable name
set(PIAFS_EXECUTABLE_NAME "PIAFS${PIAFS_NAME_SUFFIX}")
message(STATUS "Binary name: ${PIAFS_EXECUTABLE_NAME}")

# Main executable
add_executable(${PIAFS_EXECUTABLE_NAME} main.cpp)

# Include build_info.h
target_include_directories(${PIAFS_EXECUTABLE_NAME} PRIVATE ${CMAKE_BINARY_DIR})

# Add GPU compile definitions to main executable if GPU is enabled
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    target_compile_definitions(${PIAFS_EXECUTABLE_NAME} PRIVATE GPU_CUDA)
  elseif(ENABLE_HIP)
    target_compile_definitions(${PIAFS_EXECUTABLE_NAME} PRIVATE GPU_HIP)
  endif()
endif()

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
  GPU
)

# Explicitly link CUDA runtime if CUDA is enabled
# This ensures cudaGetDeviceCount() and other CUDA runtime functions work
if(ENABLE_GPU AND ENABLE_CUDA)
  target_link_libraries(${PIAFS_EXECUTABLE_NAME} CUDA::cudart)
  # Enable device linking for CUDA separable compilation
  # This ensures device symbols from static libraries are properly linked
  set_property(TARGET ${PIAFS_EXECUTABLE_NAME} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  message(STATUS "Linking CUDA runtime library to ${PIAFS_EXECUTABLE_NAME}")
  message(STATUS "CUDA device linking enabled for ${PIAFS_EXECUTABLE_NAME}")
endif()

# Link MPI libraries if MPI is enabled
if(NOT ENABLE_SERIAL)
  # Link MPI libraries - need both C and CXX libraries
  if(MPI_CXX_FOUND AND MPI_CXX_LIBRARIES)
    target_link_libraries(${PIAFS_EXECUTABLE_NAME} ${MPI_CXX_LIBRARIES})
    message(STATUS "Linking MPI CXX libraries: ${MPI_CXX_LIBRARIES}")
  endif()
  if(MPI_C_FOUND AND MPI_C_LIBRARIES)
    target_link_libraries(${PIAFS_EXECUTABLE_NAME} ${MPI_C_LIBRARIES})
    message(STATUS "Linking MPI C libraries: ${MPI_C_LIBRARIES}")
  endif()
  # Also try to use MPI::MPI_CXX and MPI::MPI_C targets if available (modern CMake)
  if(TARGET MPI::MPI_CXX)
    target_link_libraries(${PIAFS_EXECUTABLE_NAME} MPI::MPI_CXX)
    message(STATUS "Linking MPI::MPI_CXX target")
  endif()
  if(TARGET MPI::MPI_C)
    target_link_libraries(${PIAFS_EXECUTABLE_NAME} MPI::MPI_C)
    message(STATUS "Linking MPI::MPI_C target")
  endif()
  # If MPI libraries are still empty, try to get them from mpicc/mpicxx
  if((NOT MPI_CXX_LIBRARIES AND NOT MPI_C_LIBRARIES) OR (NOT TARGET MPI::MPI_CXX AND NOT TARGET MPI::MPI_C))
    if(MPI_CXX_COMPILER)
      # Try -show first (works with Cray MPICH)
      execute_process(
        COMMAND ${MPI_CXX_COMPILER} -show
        OUTPUT_VARIABLE MPI_CXX_SHOW_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if(MPI_CXX_SHOW_OUTPUT)
        message(STATUS "Parsing mpicxx -show output: ${MPI_CXX_SHOW_OUTPUT}")
        # Split the output into words to parse properly
        string(REPLACE " " ";" MPI_WORDS "${MPI_CXX_SHOW_OUTPUT}")
        
        # Process each word
        set(WHOLE_ARCHIVE_PROCESSED FALSE)
        foreach(word ${MPI_WORDS})
          # Handle -L flags (library directories) - use target_link_options to ensure they're in linker command
          if(word MATCHES "^-L(.+)$")
            target_link_options(${PIAFS_EXECUTABLE_NAME} PRIVATE ${word})
            string(REGEX REPLACE "^-L" "" lib_dir "${word}")
            message(STATUS "Added MPI library directory: ${word} (${lib_dir})")
          # Handle -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive (comma-separated)
          elseif(word MATCHES "^-Wl,--whole-archive")
            # Split on commas to get individual parts
            string(REPLACE "," ";" WL_PARTS "${word}")
            set(WHOLE_ARCHIVE_LIB "")
            foreach(part ${WL_PARTS})
              if(part MATCHES "^-l(.+)$")
                string(REGEX REPLACE "^-l" "" lib_name "${part}")
                set(WHOLE_ARCHIVE_LIB "${lib_name}")
              endif()
            endforeach()
            if(WHOLE_ARCHIVE_LIB)
              target_link_options(${PIAFS_EXECUTABLE_NAME} PRIVATE "-Wl,--whole-archive" "-l${WHOLE_ARCHIVE_LIB}" "-Wl,--no-whole-archive")
              message(STATUS "Added MPI whole-archive library: -l${WHOLE_ARCHIVE_LIB}")
              set(WHOLE_ARCHIVE_PROCESSED TRUE)
            endif()
          # Handle -Wl,-rpath,path (may contain colons) - add as RPATH
          elseif(word MATCHES "^-Wl,-rpath,")
            string(REGEX REPLACE "^-Wl,-rpath," "" rpath_paths "${word}")
            # Split by colons if multiple paths
            string(REPLACE ":" ";" rpath_list "${rpath_paths}")
            foreach(rpath_dir ${rpath_list})
              # Add as RPATH using target_link_options
              target_link_options(${PIAFS_EXECUTABLE_NAME} PRIVATE "-Wl,-rpath,${rpath_dir}")
              message(STATUS "Added MPI RPATH: ${rpath_dir}")
            endforeach()
          # Handle other -Wl flags (pass through as-is)
          elseif(word MATCHES "^-Wl,")
            target_link_options(${PIAFS_EXECUTABLE_NAME} PRIVATE ${word})
          # Handle -l flags (libraries) - skip hugetlbfs if already processed in whole-archive
          elseif(word MATCHES "^-l(.+)$")
            string(REGEX REPLACE "^-l" "" lib_name "${word}")
            # Skip hugetlbfs if we already processed it in whole-archive
            if(NOT (WHOLE_ARCHIVE_PROCESSED AND lib_name STREQUAL "hugetlbfs"))
              target_link_libraries(${PIAFS_EXECUTABLE_NAME} ${word})
              message(STATUS "Added MPI library: ${word}")
            endif()
          endif()
        endforeach()
        
        message(STATUS "Finished parsing mpicxx -show output")
      else()
        # Fallback to -showme:link (OpenMPI)
        execute_process(
          COMMAND ${MPI_CXX_COMPILER} -showme:link
          OUTPUT_VARIABLE MPI_CXX_LINK_FLAGS
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
        )
        if(MPI_CXX_LINK_FLAGS)
          string(REGEX MATCHALL "-l[^ ]+" MPI_LIBS "${MPI_CXX_LINK_FLAGS}")
          string(REGEX MATCHALL "-L[^ ]+" MPI_LIB_DIRS "${MPI_CXX_LINK_FLAGS}")
          foreach(lib ${MPI_LIBS})
            target_link_libraries(${PIAFS_EXECUTABLE_NAME} ${lib})
          endforeach()
          foreach(lib_dir ${MPI_LIB_DIRS})
            string(REPLACE "-L" "" lib_dir_clean "${lib_dir}")
            target_link_directories(${PIAFS_EXECUTABLE_NAME} PRIVATE ${lib_dir_clean})
          endforeach()
          message(STATUS "Linking MPI libraries from mpicxx -showme:link: ${MPI_LIBS}")
        endif()
      endif()
    endif()
  endif()
endif()
