# CMake script to set up PIAFS benchmarks for regression testing
#
# This script:
# 1. Clones/updates the benchmarks repository
# 2. Compiles the PIAFS diff utility
# 3. Prepares the test run directory

cmake_minimum_required(VERSION 3.10)

# Required variables that must be passed:
# - PIAFS_SOURCE_DIR: Source directory of PIAFS
# - PIAFS_BINARY_DIR: Build directory of PIAFS
# - PIAFS_EXEC: Path to the PIAFS executable
# - MPIEXEC: MPI executor command
# - C_COMPILER: C compiler to use for building diff utility

message(STATUS "Setting up PIAFS regression test benchmarks...")

# Configuration
set(BENCHMARKS_REPO "https://github.com/debog/piafs_benchmarks.git")
set(BENCHMARKS_BRANCH "master")
set(TEST_RUN_DIR "${PIAFS_BINARY_DIR}/test_run_temp")
set(BENCHMARKS_DIR "${TEST_RUN_DIR}/benchmarks")
set(DIFF_TOOL_SRC "${PIAFS_SOURCE_DIR}/Extras/piafsDiff_RegTests.c")
set(DIFF_TOOL_BIN "${TEST_RUN_DIR}/PIAFS_DIFF")

# Create test run directory
file(MAKE_DIRECTORY "${TEST_RUN_DIR}")

# Clone or update benchmarks repository
if(EXISTS "${BENCHMARKS_DIR}/.git")
  message(STATUS "Updating existing benchmarks repository...")
  execute_process(
    COMMAND git reset --hard HEAD
    WORKING_DIRECTORY "${BENCHMARKS_DIR}"
    RESULT_VARIABLE GIT_RESET_RESULT
    OUTPUT_QUIET ERROR_QUIET
  )
  execute_process(
    COMMAND git checkout ${BENCHMARKS_BRANCH}
    WORKING_DIRECTORY "${BENCHMARKS_DIR}"
    RESULT_VARIABLE GIT_CHECKOUT_RESULT
    OUTPUT_QUIET ERROR_QUIET
  )
  execute_process(
    COMMAND git pull --force
    WORKING_DIRECTORY "${BENCHMARKS_DIR}"
    RESULT_VARIABLE GIT_PULL_RESULT
    OUTPUT_VARIABLE GIT_OUTPUT
    ERROR_VARIABLE GIT_ERROR
  )
  if(NOT GIT_PULL_RESULT EQUAL 0)
    message(WARNING "Failed to update benchmarks: ${GIT_ERROR}")
  endif()
else()
  message(STATUS "Cloning benchmarks repository...")
  execute_process(
    COMMAND git clone ${BENCHMARKS_REPO} ${BENCHMARKS_DIR}
    WORKING_DIRECTORY "${TEST_RUN_DIR}"
    RESULT_VARIABLE GIT_CLONE_RESULT
    OUTPUT_VARIABLE GIT_OUTPUT
    ERROR_VARIABLE GIT_ERROR
  )
  if(NOT GIT_CLONE_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to clone benchmarks repository: ${GIT_ERROR}")
  endif()

  execute_process(
    COMMAND git checkout ${BENCHMARKS_BRANCH}
    WORKING_DIRECTORY "${BENCHMARKS_DIR}"
    RESULT_VARIABLE GIT_CHECKOUT_RESULT
    OUTPUT_QUIET ERROR_QUIET
  )
endif()

# Verify PIAFS diff utility exists (should be built by CMake)
if(NOT EXISTS "${DIFF_TOOL_BIN}")
  message(FATAL_ERROR "PIAFS diff utility not found at: ${DIFF_TOOL_BIN}\nIt should have been built by CMake. Try running 'make PIAFS_DIFF' first.")
else()
  message(STATUS "PIAFS diff utility found: ${DIFF_TOOL_BIN}")
endif()

# Verify all expected benchmarks are present
if(DEFINED EXPECTED_BENCHMARKS AND NOT EXPECTED_BENCHMARKS STREQUAL "")
  message(STATUS "Verifying expected benchmarks...")
  string(REPLACE "," ";" BENCHMARK_LIST "${EXPECTED_BENCHMARKS}")

  set(MISSING_BENCHMARKS "")
  set(DISABLED_BENCHMARKS "")
  set(VALID_BENCHMARKS 0)

  foreach(BENCHMARK ${BENCHMARK_LIST})
    if(EXISTS "${BENCHMARKS_DIR}/${BENCHMARK}")
      if(EXISTS "${BENCHMARKS_DIR}/${BENCHMARK}/.disabled")
        list(APPEND DISABLED_BENCHMARKS "${BENCHMARK}")
        message(STATUS "  ${BENCHMARK}: DISABLED")
      elseif(EXISTS "${BENCHMARKS_DIR}/${BENCHMARK}/run.sh")
        math(EXPR VALID_BENCHMARKS "${VALID_BENCHMARKS} + 1")
        message(STATUS "  ${BENCHMARK}: OK")
      else()
        list(APPEND MISSING_BENCHMARKS "${BENCHMARK}")
        message(STATUS "  ${BENCHMARK}: MISSING run.sh")
      endif()
    else()
      list(APPEND MISSING_BENCHMARKS "${BENCHMARK}")
      message(STATUS "  ${BENCHMARK}: NOT FOUND")
    endif()
  endforeach()

  # Report summary
  message(STATUS "")
  message(STATUS "Benchmark verification summary:")
  message(STATUS "  Valid benchmarks: ${VALID_BENCHMARKS}")

  if(DISABLED_BENCHMARKS)
    list(LENGTH DISABLED_BENCHMARKS NUM_DISABLED)
    message(STATUS "  Disabled benchmarks: ${NUM_DISABLED}")
    foreach(BENCH ${DISABLED_BENCHMARKS})
      message(STATUS "    - ${BENCH}")
    endforeach()
  endif()

  if(MISSING_BENCHMARKS)
    list(LENGTH MISSING_BENCHMARKS NUM_MISSING)
    message(WARNING "Missing ${NUM_MISSING} expected benchmark(s):")
    foreach(BENCH ${MISSING_BENCHMARKS})
      message(WARNING "  - ${BENCH}")
    endforeach()
    message(FATAL_ERROR "Not all expected benchmarks are available. Repository may be incomplete.")
  endif()

  message(STATUS "All expected benchmarks verified successfully")
else()
  message(STATUS "No benchmark verification requested (EXPECTED_BENCHMARKS not set)")
endif()

message(STATUS "Benchmark setup complete")
