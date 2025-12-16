# PIAFS Regression Tests Configuration

cmake_minimum_required(VERSION 3.10)

# Build the PIAFS diff utility as a standalone executable
add_executable(PIAFS_DIFF ${CMAKE_SOURCE_DIR}/Extras/piafsDiff_RegTests.c)
target_link_libraries(PIAFS_DIFF m)
set_target_properties(PIAFS_DIFF PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test_run_temp"
)

# Determine PIAFS executable name/path
# This should match the logic in src/CMakeLists.txt
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

set(PIAFS_EXECUTABLE_NAME "PIAFS${PIAFS_NAME_SUFFIX}")
set(PIAFS_EXECUTABLE_PATH "${CMAKE_BINARY_DIR}/bin/${PIAFS_EXECUTABLE_NAME}")

# Add a fixture to build PIAFS_DIFF before benchmarks setup
add_test(
  NAME build_piafs_diff
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target PIAFS_DIFF
)
set_tests_properties(build_piafs_diff PROPERTIES
  FIXTURES_SETUP diff_tool
  LABELS "regression;setup"
)

# Create benchmark setup fixture
# Pass the list of expected benchmarks for verification
string(REPLACE ";" "," BENCHMARKS_LIST "${REGRESSION_BENCHMARKS}")
add_test(
  NAME setup_benchmarks
  COMMAND ${CMAKE_COMMAND}
    -DPIAFS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
    -DPIAFS_BINARY_DIR=${CMAKE_BINARY_DIR}
    -DPIAFS_EXEC=${PIAFS_EXECUTABLE_PATH}
    -DMPIEXEC=${MPIEXEC}
    -DC_COMPILER=${CMAKE_C_COMPILER}
    -DEXPECTED_BENCHMARKS=${BENCHMARKS_LIST}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SetupBenchmarks.cmake
)

set_tests_properties(setup_benchmarks PROPERTIES
  FIXTURES_SETUP regression_benchmarks
  FIXTURES_REQUIRED "prepare_binary;diff_tool"
  LABELS "regression;setup"
  TIMEOUT 300
)

# Define all known regression test benchmarks
# This list should be kept in sync with the piafs_benchmarks repository
set(REGRESSION_BENCHMARKS
  1d_beam_grating
  1d_euler_densitysinewave
  1d_euler_firstorder
  1d_euler_forwardeuler
  1d_euler_laxshocktube
  1d_euler_muscl2
  1d_euler_rk22
  1d_euler_rk33
  1d_euler_secondorder_central
  2d_beam_grating
  2d_ensemble_navstok_vortconv
  2d_navstok_densitysinewave
  2d_navstok_flatplatelaminar
  2d_navstok_lidcav
  2d_navstok_riemann4
  2d_navstok_vortconv
  3d_beam_grating
  3d_navstok_riemann4_xy
  3d_navstok_riemann4_xz
  3d_navstok_riemann4_yz
)

# Function to add a regression test
function(add_regression_test TEST_NAME)
  add_test(
    NAME "regression_${TEST_NAME}"
    COMMAND ${CMAKE_COMMAND}
      -DTEST_NAME=${TEST_NAME}
      -DPIAFS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
      -DPIAFS_BINARY_DIR=${CMAKE_BINARY_DIR}
      -DPIAFS_EXEC=${PIAFS_EXECUTABLE_PATH}
      -DMPIEXEC=${MPIEXEC}
      -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunRegressionTest.cmake
  )

  set_tests_properties("regression_${TEST_NAME}" PROPERTIES
    FIXTURES_REQUIRED "regression_benchmarks;prepare_binary"
    LABELS "regression"
    TIMEOUT 600
  )
endfunction()

# Register all known regression tests
foreach(TEST_NAME ${REGRESSION_BENCHMARKS})
  add_regression_test("${TEST_NAME}")
  message(STATUS "Registered regression test: ${TEST_NAME}")
endforeach()

# Add a convenience target to run all regression tests
add_custom_target(test_regression
  COMMAND ${CMAKE_CTEST_COMMAND} -L regression --output-on-failure
  COMMENT "Running all regression tests"
  USES_TERMINAL
)
