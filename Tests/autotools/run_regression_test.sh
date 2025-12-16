#!/bin/bash

# Run a single PIAFS regression benchmark (Autotools)
# Usage: run_regression_test.sh <benchmark_name>

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <benchmark_name>"
  exit 1
fi

BENCHMARK_NAME="$1"

# Environment variables
BENCHMARKS_DIR="${BENCHMARKS_DIR:-${PIAFS_DIR}/test_run_temp/benchmarks}"
TEST_RUN_DIR="${PIAFS_DIR}/test_run_temp/regression_${BENCHMARK_NAME}"
DIFF_BINARY="${PIAFS_DIR}/test_run_temp/PIAFS_DIFF"
DIFF_TOLERANCE="${DIFF_TOLERANCE:-1.0e-14}"

echo "========================================="
echo "Running regression test: $BENCHMARK_NAME"
echo "========================================="

# Check if benchmark exists
if [ ! -d "$BENCHMARKS_DIR/$BENCHMARK_NAME" ]; then
  echo "ERROR: Benchmark not found: $BENCHMARKS_DIR/$BENCHMARK_NAME"
  exit 1
fi

# Check if benchmark is disabled
if [ -f "$BENCHMARKS_DIR/$BENCHMARK_NAME/.disabled" ]; then
  echo "SKIPPED: Benchmark is disabled (.disabled file present)"
  exit 77  # Special exit code for "test skipped" in autotools
fi

# Create test run directory
rm -rf "$TEST_RUN_DIR"
mkdir -p "$TEST_RUN_DIR"

# Copy benchmark files (excluding output files and git)
echo "Copying benchmark files..."
cd "$BENCHMARKS_DIR/$BENCHMARK_NAME"
for file in *; do
  # Skip output files, git files, and README
  if [[ ! "$file" =~ ^(op.*|initial.*|out\.log|README\.md|\.git.*)$ ]]; then
    if [ -d "$file" ]; then
      cp -r "$file" "$TEST_RUN_DIR/"
    else
      cp "$file" "$TEST_RUN_DIR/"
    fi
  fi
done

# Run the benchmark
echo "Executing benchmark..."
cd "$TEST_RUN_DIR"

# Export environment variables for the test
export PIAFS_EXEC_W_PATH
export MPI_EXEC

# Run the benchmark script
if [ -f "run.sh" ]; then
  chmod +x run.sh
  if ! ./run.sh; then
    echo "ERROR: Benchmark execution failed"
    exit 1
  fi
else
  echo "ERROR: run.sh not found in benchmark directory"
  exit 1
fi

# Compare outputs
echo "Comparing outputs..."
if [ ! -f "diff_file_list" ]; then
  echo "ERROR: diff_file_list not found"
  exit 1
fi

# First verify all expected output files were generated
missing_files=""
while IFS= read -r output_file; do
  # Skip empty lines and comments
  [[ -z "$output_file" || "$output_file" =~ ^# ]] && continue
  output_file=$(echo "$output_file" | xargs) # trim whitespace

  if [ ! -f "$output_file" ]; then
    missing_files="${missing_files}  - $output_file\n"
  fi
done < "diff_file_list"

if [ -n "$missing_files" ]; then
  echo ""
  echo "ERROR: PIAFS did not generate expected output files:"
  echo -e "$missing_files"
  echo "This usually means the simulation failed to run."
  echo "Check $TEST_RUN_DIR/out.log for details."
  exit 1
fi

comparison_failed=0
diff_output=""
while IFS= read -r output_file; do
  # Skip empty lines and comments
  [[ -z "$output_file" || "$output_file" =~ ^# ]] && continue

  output_file=$(echo "$output_file" | xargs) # trim whitespace

  if [ ! -f "$output_file" ]; then
    echo "  ✗ $output_file: OUTPUT FILE NOT FOUND"
    comparison_failed=1
    continue
  fi

  reference_file="$BENCHMARKS_DIR/$BENCHMARK_NAME/$output_file"
  if [ ! -f "$reference_file" ]; then
    echo "  ✗ $output_file: REFERENCE FILE NOT FOUND"
    comparison_failed=1
    continue
  fi

  # Run diff utility and capture output
  diff_result=$("$DIFF_BINARY" -r "$DIFF_TOLERANCE" "$output_file" "$reference_file" 2>&1)
  if [ $? -eq 0 ]; then
    echo "  ✓ $output_file"
  else
    echo "  ✗ $output_file: COMPARISON FAILED"
    comparison_failed=1
    diff_output="${diff_output}\n--- Diff output for $output_file ---\n${diff_result}\n"
  fi
done < "diff_file_list"

if [ $comparison_failed -eq 0 ]; then
  echo ""
  echo "PASSED: $BENCHMARK_NAME"
  exit 0
else
  echo ""
  echo "FAILED: $BENCHMARK_NAME"
  if [ -n "$diff_output" ]; then
    echo ""
    echo "========================================="
    echo "Detailed Comparison Results:"
    echo "========================================="
    echo -e "$diff_output"
  fi
  exit 1
fi
