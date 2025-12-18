# CMake script to run an individual PIAFS regression test
#
# Required variables:
# - TEST_NAME: Name of the test (benchmark directory name)
# - PIAFS_BINARY_DIR: Build directory of PIAFS
# - PIAFS_EXEC: Path to the PIAFS executable
# - MPIEXEC: MPI executor command
# - PIAFS_SOURCE_DIR: Source directory (for locating reference solutions)

cmake_minimum_required(VERSION 3.10)

set(TEST_RUN_DIR "${PIAFS_BINARY_DIR}/test_run_temp")
set(BENCHMARKS_DIR "${TEST_RUN_DIR}/benchmarks")
set(TEST_DIR "${TEST_RUN_DIR}/${TEST_NAME}")
set(DIFF_TOOL "${TEST_RUN_DIR}/PIAFS_DIFF")
set(DIFF_TOLERANCE "1.0e-14")

# Create test directory
file(REMOVE_RECURSE "${TEST_DIR}")
file(MAKE_DIRECTORY "${TEST_DIR}")

# Copy benchmark files (excluding output and git files)
message(STATUS "Setting up test ${TEST_NAME}...")
file(GLOB_RECURSE BENCHMARK_FILES
  LIST_DIRECTORIES false
  "${BENCHMARKS_DIR}/${TEST_NAME}/*"
)

foreach(FILE ${BENCHMARK_FILES})
  # Skip output files, logs, and git files
  get_filename_component(FNAME "${FILE}" NAME)
  if(NOT FNAME MATCHES "^(op|initial|out\\.log|README\\.md|\\.git)")
    file(RELATIVE_PATH REL_PATH "${BENCHMARKS_DIR}/${TEST_NAME}" "${FILE}")
    get_filename_component(REL_DIR "${REL_PATH}" DIRECTORY)
    if(REL_DIR)
      file(MAKE_DIRECTORY "${TEST_DIR}/${REL_DIR}")
    endif()
    file(COPY "${FILE}" DESTINATION "${TEST_DIR}/${REL_DIR}")
  endif()
endforeach()

# Check if test is disabled
if(EXISTS "${BENCHMARKS_DIR}/${TEST_NAME}/.disabled")
  message(STATUS "Test ${TEST_NAME} is disabled, skipping")
  return()
endif()

# Check for run script
if(NOT EXISTS "${TEST_DIR}/run.sh")
  message(FATAL_ERROR "Test ${TEST_NAME}: run.sh not found")
endif()

# Make run script executable
file(CHMOD "${TEST_DIR}/run.sh" PERMISSIONS
  OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE
)

# Set environment variables for the test
set(ENV{PIAFS_EXEC_W_PATH} "${PIAFS_EXEC}")
set(ENV{MPI_EXEC} "${MPIEXEC}")
# Set GPU flags for srun (needed for GPU-enabled builds on SLURM systems)
if(ENABLE_GPU)
  set(ENV{PIAFS_EXEC_OTHER_ARGS} "--gpus-per-node=1")
  # Disable GPU for automated tests (login nodes don't have GPU access)
  # GPU chemistry is fully implemented and can be enabled manually in GPU sessions
  set(ENV{PIAFS_USE_GPU} "0")
else()
  set(ENV{PIAFS_EXEC_OTHER_ARGS} "")
  set(ENV{PIAFS_USE_GPU} "0")
endif()

# Run the test
message(STATUS "Running test ${TEST_NAME}...")
execute_process(
  COMMAND ./run.sh
  WORKING_DIRECTORY "${TEST_DIR}"
  RESULT_VARIABLE RUN_RESULT
  OUTPUT_FILE "${TEST_DIR}/out.log"
  ERROR_FILE "${TEST_DIR}/out.log"
  TIMEOUT 600
)

if(NOT RUN_RESULT EQUAL 0)
  message(FATAL_ERROR "Test ${TEST_NAME} failed to execute (exit code: ${RUN_RESULT})")
endif()

# Check for diff_file_list
if(NOT EXISTS "${TEST_DIR}/diff_file_list")
  message(FATAL_ERROR "Test ${TEST_NAME}: diff_file_list not found after run")
endif()

# Compare output files
file(STRINGS "${TEST_DIR}/diff_file_list" FILES_TO_COMPARE)
set(COMPARISON_FAILED FALSE)
set(COMPARISON_COUNT 0)
set(PASS_COUNT 0)
set(FAIL_COUNT 0)
set(MISSING_OUTPUT_FILES "")

# First verify all expected output files were generated
foreach(FILE ${FILES_TO_COMPARE})
  string(STRIP "${FILE}" FILE)
  if(FILE AND NOT EXISTS "${TEST_DIR}/${FILE}")
    list(APPEND MISSING_OUTPUT_FILES "${FILE}")
  endif()
endforeach()

if(MISSING_OUTPUT_FILES)
  message(FATAL_ERROR "Test ${TEST_NAME} FAILED: PIAFS did not generate expected output files: ${MISSING_OUTPUT_FILES}\nThis usually means the simulation failed to run (check ${TEST_DIR}/out.log)")
endif()

foreach(FILE ${FILES_TO_COMPARE})
  string(STRIP "${FILE}" FILE)
  if(FILE)
    message(STATUS "  Comparing ${FILE}...")

    if(NOT EXISTS "${TEST_DIR}/${FILE}")
      message(STATUS "    ERROR: Output file not found: ${TEST_DIR}/${FILE}")
      set(COMPARISON_FAILED TRUE)
      math(EXPR FAIL_COUNT "${FAIL_COUNT} + 1")
      continue()
    endif()

    if(NOT EXISTS "${BENCHMARKS_DIR}/${TEST_NAME}/${FILE}")
      message(STATUS "    ERROR: Reference file not found: ${BENCHMARKS_DIR}/${TEST_NAME}/${FILE}")
      set(COMPARISON_FAILED TRUE)
      math(EXPR FAIL_COUNT "${FAIL_COUNT} + 1")
      continue()
    endif()

    execute_process(
      COMMAND ${DIFF_TOOL} -v -r ${DIFF_TOLERANCE} ${FILE} ${BENCHMARKS_DIR}/${TEST_NAME}/${FILE}
      WORKING_DIRECTORY "${TEST_DIR}"
      RESULT_VARIABLE DIFF_RESULT
      OUTPUT_VARIABLE DIFF_OUTPUT
      ERROR_VARIABLE DIFF_ERROR
    )

    math(EXPR COMPARISON_COUNT "${COMPARISON_COUNT} + 1")

    if(NOT DIFF_RESULT EQUAL 0)
      message(STATUS "    FAILED: Differences found")
      message(STATUS "")
      message(STATUS "    ========================================")
      message(STATUS "    Diff output for ${FILE}:")
      message(STATUS "    ========================================")
      if(DIFF_OUTPUT)
        message(STATUS "${DIFF_OUTPUT}")
      endif()
      if(DIFF_ERROR)
        message(STATUS "${DIFF_ERROR}")
      endif()
      message(STATUS "    ========================================")
      message(STATUS "")
      set(COMPARISON_FAILED TRUE)
      math(EXPR FAIL_COUNT "${FAIL_COUNT} + 1")
    else()
      message(STATUS "    PASSED")
      if(DIFF_OUTPUT)
        message(STATUS "${DIFF_OUTPUT}")
      endif()
      math(EXPR PASS_COUNT "${PASS_COUNT} + 1")
    endif()
  endif()
endforeach()

# Report results
message(STATUS "Test ${TEST_NAME} complete: ${PASS_COUNT}/${COMPARISON_COUNT} comparisons passed")

if(COMPARISON_FAILED)
  message(FATAL_ERROR "Test ${TEST_NAME} FAILED: ${FAIL_COUNT} comparisons failed")
endif()

message(STATUS "Test ${TEST_NAME} PASSED")
