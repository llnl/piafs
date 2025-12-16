#!/bin/bash

# Wrapper script for individual regression tests
# This script is called by automake with the benchmark name as the script name

# Extract benchmark name from script name
SCRIPT_NAME=$(basename "$0")
BENCHMARK_NAME="${SCRIPT_NAME%.sh}"

# Remove the "test_" prefix if present
BENCHMARK_NAME="${BENCHMARK_NAME#test_}"

# Get the directory where this script lives
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Call the actual test runner
exec "$SCRIPT_DIR/run_regression_test.sh" "$BENCHMARK_NAME"
