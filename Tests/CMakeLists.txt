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

# Add GPU info
if(ENABLE_GPU)
  if(ENABLE_CUDA)
    set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-cuda")
  elseif(ENABLE_HIP)
    set(PIAFS_NAME_SUFFIX "${PIAFS_NAME_SUFFIX}-hip")
  endif()
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

# Detect if we're on Matrix, Tuolumne, or Dane for HPC testing
execute_process(
  COMMAND hostname
  OUTPUT_VARIABLE HOSTNAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Check if we're on Matrix, Tuolumne, or Dane
string(REGEX MATCH "^matrix" IS_MATRIX "${HOSTNAME}")
string(REGEX MATCH "^tuolumne" IS_TUOLUMNE "${HOSTNAME}")
string(REGEX MATCH "^dane" IS_DANE "${HOSTNAME}")

if(IS_MATRIX OR IS_TUOLUMNE)
  set(USE_HPC_TEST_SCRIPTS TRUE)
  set(USE_GPU_TESTS TRUE)
  if(IS_MATRIX)
    set(HPC_TEST_SCRIPTS_DIR "/g/g92/ghosh5/Runs/piafs/regtests/matrix/_test_Fri_Dec_12_12.30.49_PST_2025")
    message(STATUS "Detected Matrix system - using GPU test scripts from ${HPC_TEST_SCRIPTS_DIR}")
  else()
    set(HPC_TEST_SCRIPTS_DIR "/g/g92/ghosh5/Runs/piafs/regtests/tuolumne/_test_Fri_Dec_12_13.43.38_PST_2025")
    message(STATUS "Detected Tuolumne system - using GPU test scripts from ${HPC_TEST_SCRIPTS_DIR}")
  endif()
elseif(IS_DANE)
  set(USE_HPC_TEST_SCRIPTS TRUE)
  set(USE_GPU_TESTS FALSE)
  message(STATUS "Detected Dane system - using HPC test scripts with CPU-only configuration")
else()
  set(USE_HPC_TEST_SCRIPTS FALSE)
  set(USE_GPU_TESTS FALSE)
  message(STATUS "Not on Matrix/Tuolumne/Dane - using standard test configuration")
endif()

# Categorize tests by GPU count assignment
# Mixed dimensions for better test coverage of GPU scaling
# Tests with 1 GPU (includes 1D, 2D, and 3D tests)
set(TESTS_1GPU
  # 1D tests with 1 GPU
  1d_beam_grating
  1d_euler_densitysinewave
  1d_euler_firstorder
  1d_euler_muscl2
  # 2D tests with 1 GPU
  2d_navstok_flatplatelaminar
  2d_navstok_lidcav
  # 3D test with 1 GPU
  3d_navstok_riemann4_yz
)

# Tests with 2 GPUs (includes 1D, 2D, and 3D tests)
set(TESTS_2GPU
  # 1D tests with 2 GPUs
  1d_euler_laxshocktube
  1d_euler_rk22
  # 2D tests with 2 GPUs
  2d_beam_grating
  2d_ensemble_navstok_vortconv
  2d_navstok_densitysinewave
  2d_navstok_vortconv
  # 3D test with 2 GPUs
  3d_beam_grating
)

# Tests with 4 GPUs (includes 1D, 2D, and 3D tests)
set(TESTS_4GPU
  # 1D tests with 4 GPUs
  1d_euler_rk33
  1d_euler_secondorder_central
  # 2D test with 4 GPUs
  2d_navstok_riemann4
  # 3D tests with 4 GPUs
  3d_navstok_riemann4_xy
  3d_navstok_riemann4_xz
)

# Tests that must run with GPU disabled (interpolation schemes not yet implemented on GPU)
# These tests use crweno5 or cupw5 interpolation schemes which aren't GPU-ready
set(GPU_DISABLED_TESTS
  1d_euler_laxshocktube
  2d_navstok_vortconv
)

# Function to add a regression test
function(add_regression_test TEST_NAME)
  # Check if this test should have GPU disabled
  set(DISABLE_GPU_FOR_TEST OFF)
  if(TEST_NAME IN_LIST GPU_DISABLED_TESTS)
    set(DISABLE_GPU_FOR_TEST ON)
  endif()
  
  # Skip GPU-disabled tests entirely when running GPU regression tests
  if(USE_GPU_TESTS AND ENABLE_GPU AND DISABLE_GPU_FOR_TEST)
    message(STATUS "Skipping regression test ${TEST_NAME} (requires GPU-unsupported interpolation: crweno5/cupw5)")
    return()
  endif()
  
  # Determine MPI rank count for this test (matches GPU count for consistency)
  set(MPI_RANKS 1)
  if(TEST_NAME IN_LIST TESTS_2GPU)
    set(MPI_RANKS 2)
  elseif(TEST_NAME IN_LIST TESTS_4GPU)
    set(MPI_RANKS 4)
  endif()

  if(USE_HPC_TEST_SCRIPTS)
    if(USE_GPU_TESTS AND ENABLE_GPU)
      # Use GPU-specific test script on Matrix/Tuolumne
      add_test(
        NAME "regression_${TEST_NAME}"
        COMMAND ${CMAKE_COMMAND}
          -DTEST_NAME=${TEST_NAME}
          -DPIAFS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DPIAFS_BINARY_DIR=${CMAKE_BINARY_DIR}
          -DPIAFS_EXEC=${PIAFS_EXECUTABLE_PATH}
          -DMPIEXEC=${MPIEXEC}
          -DENABLE_GPU=${ENABLE_GPU}
          -DHPC_TEST_SCRIPTS_DIR=${HPC_TEST_SCRIPTS_DIR}
          -DMPI_RANKS=${MPI_RANKS}
          -DHOSTNAME=${HOSTNAME}
          -DDISABLE_GPU_FOR_TEST=${DISABLE_GPU_FOR_TEST}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunGPURegressionTest.cmake
      )
    else()
      # Use HPC test script for Dane (CPU-only with srun)
      add_test(
        NAME "regression_${TEST_NAME}"
        COMMAND ${CMAKE_COMMAND}
          -DTEST_NAME=${TEST_NAME}
          -DPIAFS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DPIAFS_BINARY_DIR=${CMAKE_BINARY_DIR}
          -DPIAFS_EXEC=${PIAFS_EXECUTABLE_PATH}
          -DMPIEXEC=${MPIEXEC}
          -DENABLE_GPU=${ENABLE_GPU}
          -DMPI_RANKS=${MPI_RANKS}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunHPCRegressionTest.cmake
      )
    endif()
  else()
    # Use standard test script for desktop systems
    add_test(
      NAME "regression_${TEST_NAME}"
      COMMAND ${CMAKE_COMMAND}
        -DTEST_NAME=${TEST_NAME}
        -DPIAFS_SOURCE_DIR=${CMAKE_SOURCE_DIR}
        -DPIAFS_BINARY_DIR=${CMAKE_BINARY_DIR}
        -DPIAFS_EXEC=${PIAFS_EXECUTABLE_PATH}
        -DMPIEXEC=${MPIEXEC}
        -DENABLE_GPU=${ENABLE_GPU}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunRegressionTest.cmake
    )
  endif()

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
