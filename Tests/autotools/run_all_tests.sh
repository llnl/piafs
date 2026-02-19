#!/bin/bash

# Main test runner for PIAFS regression tests (Autotools)
# This script coordinates running setup and all individual benchmark tests
# Output to stdout/stderr for automake capture in CI

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# In CI environments, /dev/tty may not exist
# Always use stdout/stderr which work in all environments
OUTPUT=/dev/stdout

# Run setup first
echo "=========================================" >&2
echo "SETUP: Cloning benchmarks and compiling diff utility" >&2
echo "=========================================" >&2
"$SCRIPT_DIR/setup_benchmarks.sh" 2>&1

if [ $? -ne 0 ]; then
  echo "FAIL: Benchmark setup failed" >&2
  exit 1
fi
echo "PASS: Benchmark setup completed" >&2

# Get list of benchmarks from environment or use all
if [ -n "$EXPECTED_BENCHMARKS" ]; then
  BENCHMARKS="$EXPECTED_BENCHMARKS"
else
  echo "WARNING: No EXPECTED_BENCHMARKS set, discovering from repository..." >&2
  BENCHMARKS_DIR="${BENCHMARKS_DIR:-${PIAFS_DIR}/test_run_temp/benchmarks}"
  BENCHMARKS=$(cd "$BENCHMARKS_DIR" && find . -maxdepth 1 -type d -name '*_*' | sed 's|^\./||' | sort)
fi

echo "" >&2
echo "=========================================" >&2
echo "Running Regression Tests" >&2
echo "=========================================" >&2

# Run each benchmark test
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0
failed_list=""

for benchmark in $BENCHMARKS; do
  total_tests=$((total_tests + 1))
  printf "TEST %2d/%2d: %-40s " "$total_tests" "20" "$benchmark" >&2

  # Capture test output to display on failure
  test_output=$("$SCRIPT_DIR/run_regression_test.sh" "$benchmark" 2>&1)
  result=$?

  if [ $result -eq 0 ]; then
    echo "PASS" >&2
    passed_tests=$((passed_tests + 1))
  elif [ $result -eq 77 ]; then
    echo "SKIP" >&2
    skipped_tests=$((skipped_tests + 1))
  else
    echo "FAIL" >&2
    failed_tests=$((failed_tests + 1))
    failed_list="$failed_list  - $benchmark\n"

    # Display detailed output for failed test
    echo "" >&2
    echo "==========================================" >&2
    echo "Failed Test Details: $benchmark" >&2
    echo "==========================================" >&2
    echo "$test_output" >&2
    echo "" >&2
  fi
done

# Print summary
echo "" >&2
echo "=========================================" >&2
echo "Regression Test Summary" >&2
echo "=========================================" >&2
echo "Total:   $total_tests" >&2
echo "Passed:  $passed_tests" >&2
echo "Failed:  $failed_tests" >&2
echo "Skipped: $skipped_tests" >&2

if [ $failed_tests -gt 0 ]; then
  echo "" >&2
  echo "Failed tests:" >&2
  echo -e "$failed_list" >&2
fi

echo "=========================================" >&2

if [ $failed_tests -gt 0 ]; then
  exit 1
else
  exit 0
fi
