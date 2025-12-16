# Unit tests using Google Test
cmake_minimum_required(VERSION 3.10)

# Enable testing
enable_testing()

# Download and configure Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.14.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/physicalmodels)
include_directories(${CMAKE_BINARY_DIR})

# Test executables
add_executable(test_array_functions ArrayFunctions/test_array_functions.cpp)
target_link_libraries(test_array_functions
  gtest_main
  ArrayFunctions
  MathFunctions
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_array_functions MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_array_functions MPI::MPI_C)
  endif()
endif()

add_executable(test_math_functions MathFunctions/test_math_functions.cpp)
target_link_libraries(test_math_functions
  gtest_main
  MathFunctions
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_math_functions MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_math_functions MPI::MPI_C)
  endif()
endif()

add_executable(test_common_functions CommonFunctions/test_common_functions.cpp)
target_link_libraries(test_common_functions
  gtest_main
  CommonFunctions
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_common_functions MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_common_functions MPI::MPI_C)
  endif()
endif()

# Register tests with CTest and add labels for grouping
include(GoogleTest)

gtest_discover_tests(test_array_functions
  PROPERTIES
    LABELS "unit;array"
)

gtest_discover_tests(test_math_functions
  PROPERTIES
    LABELS "unit;math"
)

gtest_discover_tests(test_common_functions
  PROPERTIES
    LABELS "unit;common"
)

add_executable(test_boundary_conditions BoundaryConditions/test_boundary_conditions.cpp)
target_link_libraries(test_boundary_conditions
  gtest_main
  BoundaryConditions
  ArrayFunctions
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_boundary_conditions MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_boundary_conditions MPI::MPI_C)
  endif()
endif()

gtest_discover_tests(test_boundary_conditions
  PROPERTIES
    LABELS "unit;boundary"
)

add_executable(test_first_derivative FirstDerivative/test_first_derivative.cpp)
target_link_libraries(test_first_derivative
  gtest_main
  FirstDerivative
  ArrayFunctions
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_first_derivative MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_first_derivative MPI::MPI_C)
  endif()
endif()

gtest_discover_tests(test_first_derivative
  PROPERTIES
    LABELS "unit;firstderivative"
)

add_executable(test_interpolation_functions InterpolationFunctions/test_interpolation_functions.cpp)
target_link_libraries(test_interpolation_functions
  gtest_main
  InterpolationFunctions
  ArrayFunctions
  TridiagLU
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_interpolation_functions MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_interpolation_functions MPI::MPI_C)
  endif()
endif()

gtest_discover_tests(test_interpolation_functions
  PROPERTIES
    LABELS "unit;interpolation"
)

add_executable(test_limiter_functions LimiterFunctions/test_limiter_functions.cpp)
target_link_libraries(test_limiter_functions
  gtest_main
  LimiterFunctions
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_limiter_functions MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_limiter_functions MPI::MPI_C)
  endif()
endif()

gtest_discover_tests(test_limiter_functions
  PROPERTIES
    LABELS "unit;limiter"
)

add_executable(test_io_functions IOFunctions/test_io_functions.cpp)
target_link_libraries(test_io_functions
  gtest_main
  IOFunctions
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_io_functions MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_io_functions MPI::MPI_C)
  endif()
endif()

gtest_discover_tests(test_io_functions
  PROPERTIES
    LABELS "unit;io"
)

add_executable(test_time_integration TimeIntegration/test_time_integration.cpp)
target_link_libraries(test_time_integration
  gtest_main
  TimeIntegration
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_time_integration MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_time_integration MPI::MPI_C)
  endif()
endif()

gtest_discover_tests(test_time_integration
  PROPERTIES
    LABELS "unit;timeintegration"
)

add_executable(test_tridiag_lu TridiagLU/test_tridiag_lu.cpp)
target_link_libraries(test_tridiag_lu
  gtest_main
  TridiagLU
  m
)

if(NOT ENABLE_SERIAL)
  if(MPI_CXX_FOUND)
    target_link_libraries(test_tridiag_lu MPI::MPI_CXX)
  endif()
  if(MPI_C_FOUND)
    target_link_libraries(test_tridiag_lu MPI::MPI_C)
  endif()
endif()

gtest_discover_tests(test_tridiag_lu
  PROPERTIES
    LABELS "unit;tridiaglu"
)
