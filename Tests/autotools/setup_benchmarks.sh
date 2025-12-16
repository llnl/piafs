#!/bin/bash

# Setup script for PIAFS regression benchmarks (Autotools)
# This script clones/updates the benchmarks repository and verifies all expected benchmarks exist

set -e

echo "========================================="
echo "Setting up PIAFS regression benchmarks"
echo "========================================="

# Get parameters from environment or use defaults
BENCHMARKS_REPO="${BENCHMARKS_REPO:-ssh://git@czgitlab.llnl.gov:7999/piafs/piafs_benchmarks.git}"
BENCHMARKS_BRANCH="${BENCHMARKS_BRANCH:-master}"
BENCHMARKS_DIR="${BENCHMARKS_DIR:-${PIAFS_DIR}/test_run_temp/benchmarks}"
EXPECTED_BENCHMARKS="${EXPECTED_BENCHMARKS:-}"

# Ensure test run directory exists
mkdir -p "$(dirname "$BENCHMARKS_DIR")"

# Clone or update benchmarks repository
if [ -d "$BENCHMARKS_DIR/.git" ]; then
  echo "Updating existing benchmarks repository..."
  cd "$BENCHMARKS_DIR"
  git reset --hard HEAD
  git checkout "$BENCHMARKS_BRANCH"
  git pull --force
else
  echo "Cloning benchmarks repository..."
  if [ -d "$BENCHMARKS_DIR" ]; then
    rm -rf "$BENCHMARKS_DIR"
  fi
  git clone "$BENCHMARKS_REPO" "$BENCHMARKS_DIR"
  cd "$BENCHMARKS_DIR"
  git checkout "$BENCHMARKS_BRANCH"
fi

echo "Benchmarks repository ready at: $BENCHMARKS_DIR"

# Verify expected benchmarks if list provided
if [ -n "$EXPECTED_BENCHMARKS" ]; then
  echo ""
  echo "Verifying expected benchmarks..."
  missing_benchmarks=""

  # Handle space-separated list (POSIX-compatible)
  for benchmark in $EXPECTED_BENCHMARKS; do
    benchmark=$(echo "$benchmark" | xargs) # trim whitespace
    if [ -d "$BENCHMARKS_DIR/$benchmark" ] && [ -f "$BENCHMARKS_DIR/$benchmark/run.sh" ]; then
      echo "  ✓ $benchmark"
    else
      echo "  ✗ $benchmark (MISSING)"
      missing_benchmarks="$missing_benchmarks $benchmark"
    fi
  done

  if [ -n "$missing_benchmarks" ]; then
    echo ""
    echo "ERROR: The following expected benchmarks are missing:"
    echo "$missing_benchmarks"
    exit 1
  fi

  echo ""
  echo "All expected benchmarks verified successfully!"
fi

# Compile PIAFS diff utility
echo ""
echo "Compiling PIAFS diff utility..."
PIAFS_SRC_DIR="${PIAFS_SRC_DIR:-${PIAFS_DIR}}"
DIFF_SOURCE="$PIAFS_SRC_DIR/Extras/piafsDiff_RegTests.c"
DIFF_BINARY="${PIAFS_DIR}/test_run_temp/PIAFS_DIFF"

if [ -f "$DIFF_SOURCE" ]; then
  gcc "$DIFF_SOURCE" -lm -o "$DIFF_BINARY"
  echo "PIAFS diff utility compiled: $DIFF_BINARY"
else
  echo "ERROR: PIAFS diff source not found at: $DIFF_SOURCE"
  exit 1
fi

echo ""
echo "Setup completed successfully!"
exit 0
