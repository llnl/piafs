# CMake script to run a PIAFS regression test using GPU test scripts
# This is used on Matrix/Tuolumne systems where GPU testing infrastructure exists
#
# Required variables:
# - TEST_NAME: Name of the test (benchmark directory name)
# - PIAFS_BINARY_DIR: Build directory of PIAFS
# - PIAFS_EXEC: Path to the PIAFS executable
# - MPIEXEC: MPI executor command
# - PIAFS_SOURCE_DIR: Source directory (for locating reference solutions)
# - HPC_TEST_SCRIPTS_DIR: Directory containing the run_*gpu.sh scripts
# - MPI_RANKS: Number of MPI ranks/GPUs to use (1, 2, or 4)
# - HOSTNAME: System hostname for detecting Matrix vs Tuolumne
# - DISABLE_GPU_FOR_TEST: Set to ON to run this test with GPU disabled (for unsupported features)

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
message(STATUS "Setting up GPU test ${TEST_NAME} with ${MPI_RANKS} GPU(s)...")
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

# Detect which system we're on and generate the appropriate command
string(REGEX MATCH "^matrix" IS_MATRIX "${HOSTNAME}")
string(REGEX MATCH "^tuolumne" IS_TUOLUMNE "${HOSTNAME}")

# Check if we need to generate initial.inp
set(NEEDS_INIT FALSE)
set(INIT_CODE "")
set(COMPILER "gcc")

if(EXISTS "${TEST_DIR}/aux")
  # Check for init.c, init.C, exact.c, or exact.C
  if(EXISTS "${TEST_DIR}/aux/init.c")
    set(NEEDS_INIT TRUE)
    set(INIT_CODE "init.c")
    set(COMPILER "gcc")
  elseif(EXISTS "${TEST_DIR}/aux/init.C")
    set(NEEDS_INIT TRUE)
    set(INIT_CODE "init.C")
    set(COMPILER "g++")
  elseif(EXISTS "${TEST_DIR}/aux/exact.c")
    set(NEEDS_INIT TRUE)
    set(INIT_CODE "exact.c")
    set(COMPILER "gcc")
  elseif(EXISTS "${TEST_DIR}/aux/exact.C")
    set(NEEDS_INIT TRUE)
    set(INIT_CODE "exact.C")
    set(COMPILER "g++")
  endif()
endif()

# Determine if GPU should be enabled for this test
if(DISABLE_GPU_FOR_TEST)
  set(USE_GPU_VALUE "0")
  message(STATUS "GPU disabled for test ${TEST_NAME} (unsupported interpolation scheme)")
else()
  set(USE_GPU_VALUE "1")
endif()

# Create a custom run script that uses the GPU test script template
set(RUN_SCRIPT "${TEST_DIR}/run_gpu.sh")
file(WRITE "${RUN_SCRIPT}" "#!/bin/bash\n\n")
file(APPEND "${RUN_SCRIPT}" "rm -rf *core* op* *.dat plots out.log\n")
file(APPEND "${RUN_SCRIPT}" "export PIAFS_USE_GPU=${USE_GPU_VALUE}\n\n")

# Add initial solution generation if needed
if(NEEDS_INIT)
  file(APPEND "${RUN_SCRIPT}" "# Generate initial solution\n")
  file(APPEND "${RUN_SCRIPT}" "echo \"Compiling code to generate initial solution...\"\n")
  file(APPEND "${RUN_SCRIPT}" "cd aux\n")
  file(APPEND "${RUN_SCRIPT}" "${COMPILER} ${INIT_CODE} -lm -o INIT_EXEC\n")
  file(APPEND "${RUN_SCRIPT}" "if [ ! -f INIT_EXEC ]; then\n")
  file(APPEND "${RUN_SCRIPT}" "  echo \"ERROR: Failed to compile initial solution generator\"\n")
  file(APPEND "${RUN_SCRIPT}" "  exit 1\n")
  file(APPEND "${RUN_SCRIPT}" "fi\n")
  file(APPEND "${RUN_SCRIPT}" "cd ..\n")
  file(APPEND "${RUN_SCRIPT}" "echo \"Generating initial solution...\"\n")
  file(APPEND "${RUN_SCRIPT}" "./aux/INIT_EXEC > out.log 2>&1\n")
  file(APPEND "${RUN_SCRIPT}" "rm -f aux/INIT_EXEC\n\n")
endif()

# Add PIAFS execution command
file(APPEND "${RUN_SCRIPT}" "# Run PIAFS\n")
file(APPEND "${RUN_SCRIPT}" "echo \"Running PIAFS with ${MPI_RANKS} GPU(s)...\"\n")

if(IS_MATRIX)
  # Matrix uses srun with -G flag for GPU allocation
  file(APPEND "${RUN_SCRIPT}" "srun -p pdebug -n ${MPI_RANKS} -G ${MPI_RANKS} -N 1 ${PIAFS_EXEC} 2>&1 | tee -a out.log\n")
elseif(IS_TUOLUMNE)
  # Tuolumne uses flux run with --gpus-per-task
  file(APPEND "${RUN_SCRIPT}" "flux run --exclusive --nodes=1 --ntasks ${MPI_RANKS} --gpus-per-task 1 -q=pdebug ${PIAFS_EXEC} 2>&1 | tee -a out.log\n")
else()
  message(FATAL_ERROR "Unknown system: ${HOSTNAME}. Expected matrix* or tuolumne*")
endif()

# Make run script executable
file(CHMOD "${RUN_SCRIPT}" PERMISSIONS
  OWNER_READ OWNER_WRITE OWNER_EXECUTE
  GROUP_READ GROUP_EXECUTE
  WORLD_READ WORLD_EXECUTE
)

# Run the test
message(STATUS "Running GPU test ${TEST_NAME} with ${MPI_RANKS} GPU(s)...")
execute_process(
  COMMAND ${RUN_SCRIPT}
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
      COMMAND ${DIFF_TOOL} -r ${DIFF_TOLERANCE} ${FILE} ${BENCHMARKS_DIR}/${TEST_NAME}/${FILE}
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
      math(EXPR PASS_COUNT "${PASS_COUNT} + 1")
    endif()
  endif()
endforeach()

# Report results
message(STATUS "Test ${TEST_NAME} complete (${MPI_RANKS} GPU(s)): ${PASS_COUNT}/${COMPARISON_COUNT} comparisons passed")

if(COMPARISON_FAILED)
  message(FATAL_ERROR "Test ${TEST_NAME} FAILED: ${FAIL_COUNT} comparisons failed")
endif()

message(STATUS "Test ${TEST_NAME} PASSED")

