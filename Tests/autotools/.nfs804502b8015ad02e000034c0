#!/bin/bash

# Main test runner for PIAFS regression tests (Autotools)
# This script coordinates running setup and all individual benchmark tests
# Output to /dev/tty to bypass automake's output capture

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use /dev/tty if available, otherwise fall back to stderr
if [ -w /dev/tty ]; then
  OUTPUT=/dev/tty
else
  OUTPUT=/dev/stderr
fi

# Run setup first
echo "=========================================" > "$OUTPUT"
echo "SETUP: Cloning benchmarks and compiling diff utility" > "$OUTPUT"
echo "=========================================" > "$OUTPUT"
"$SCRIPT_DIR/setup_benchmarks.sh" 2>&1 | tee -a "$OUTPUT" > /dev/null

if [ ${PIPESTATUS[0]} -ne 0 ]; then
  echo "FAIL: Benchmark setup failed" > "$OUTPUT"
  exit 1
fi
echo "PASS: Benchmark setup completed" > "$OUTPUT"

# Get list of benchmarks from environment or use all
if [ -n "$EXPECTED_BENCHMARKS" ]; then
  BENCHMARKS="$EXPECTED_BENCHMARKS"
else
  echo "WARNING: No EXPECTED_BENCHMARKS set, discovering from repository..." > "$OUTPUT"
  BENCHMARKS_DIR="${BENCHMARKS_DIR:-${PIAFS_DIR}/test_run_temp/benchmarks}"
  BENCHMARKS=$(cd "$BENCHMARKS_DIR" && find . -maxdepth 1 -type d -name '*_*' | sed 's|^\./||' | sort)
fi

echo "" > "$OUTPUT"
echo "=========================================" > "$OUTPUT"
echo "Running Regression Tests" > "$OUTPUT"
echo "=========================================" > "$OUTPUT"

# Run each benchmark test
total_tests=0
passed_tests=0
failed_tests=0
skipped_tests=0
failed_list=""

for benchmark in $BENCHMARKS; do
  total_tests=$((total_tests + 1))
  printf "TEST %2d/%2d: %-40s " "$total_tests" "20" "$benchmark" > "$OUTPUT"

  # Capture test output to display on failure
  test_output=$("$SCRIPT_DIR/run_regression_test.sh" "$benchmark" 2>&1)
  result=$?

  if [ $result -eq 0 ]; then
    echo "PASS" > "$OUTPUT"
    passed_tests=$((passed_tests + 1))
  elif [ $result -eq 77 ]; then
    echo "SKIP" > "$OUTPUT"
    skipped_tests=$((skipped_tests + 1))
  else
    echo "FAIL" > "$OUTPUT"
    failed_tests=$((failed_tests + 1))
    failed_list="$failed_list  - $benchmark\n"

    # Display detailed output for failed test
    echo "" > "$OUTPUT"
    echo "==========================================" > "$OUTPUT"
    echo "Failed Test Details: $benchmark" > "$OUTPUT"
    echo "==========================================" > "$OUTPUT"
    echo "$test_output" > "$OUTPUT"
    echo "" > "$OUTPUT"
  fi
done

# Print summary
echo "" > "$OUTPUT"
echo "=========================================" > "$OUTPUT"
echo "Regression Test Summary" > "$OUTPUT"
echo "=========================================" > "$OUTPUT"
echo "Total:   $total_tests" > "$OUTPUT"
echo "Passed:  $passed_tests" > "$OUTPUT"
echo "Failed:  $failed_tests" > "$OUTPUT"
echo "Skipped: $skipped_tests" > "$OUTPUT"

if [ $failed_tests -gt 0 ]; then
  echo "" > "$OUTPUT"
  echo "Failed tests:" > "$OUTPUT"
  echo -e "$failed_list" > "$OUTPUT"
fi

echo "=========================================" > "$OUTPUT"

if [ $failed_tests -gt 0 ]; then
  exit 1
else
  exit 0
fi
