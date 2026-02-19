#!/usr/bin/env bash

set -eu -o pipefail

# Run clang-tidy static analysis on C/C++ code files
# Only check core source files that don't require MPI or CUDA headers
# Exclude GPU code (.cu), MPI code, and unit tests

echo "Running clang-tidy static analysis on core source files..."

# Check if clang-tidy is available
if ! command -v clang-tidy &> /dev/null; then
    echo "Error: clang-tidy not found. Please install clang-tidy."
    exit 1
fi

# Create a temporary file to store results
tmpfile=$(mktemp)
trap "rm -f $tmpfile" EXIT

violations_found=0
files_checked=0

# Find C/C++ files but exclude GPU code, MPI code, and tests
find . -type d \( -name .git \
                  -o -name build -o -name install \
                  -o -name bin -o -name lib -o -name lib64 \
                  -o -path ./docs/build \
                  -o -name test_run_temp \
                  -o -name __pycache__ \
                  -o -name Examples \
                  -o -name Extras \
                  -o -name Tests \
                  -o -name GPU \
                  -o -name MPIFunctions \
               \) -prune -o \
       -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.cxx" \) \
    -print0 | while IFS= read -r -d '' file; do
        # Skip files that are not regular text files
        if ! grep -Iq . "$file" 2>/dev/null; then
            continue
        fi

        # Skip files that include mpi.h or gpu headers
        if grep -q "include.*mpi\.h" "$file" 2>/dev/null || \
           grep -q "include.*gpu" "$file" 2>/dev/null || \
           grep -q "include.*cuda" "$file" 2>/dev/null; then
            continue
        fi

        files_checked=$((files_checked + 1))
        echo "Checking: $file"

        # Run clang-tidy and capture only real warnings (not missing dependencies)
        if clang-tidy "$file" -- -I./include 2>&1 | \
           grep -E "(warning:|error:)" | \
           grep -v "mpi.h" | \
           grep -v "CUDA installation" | \
           grep -v "libdevice" | \
           grep -v "gtest/gtest.h" >> "$tmpfile"; then
            violations_found=1
        fi
    done

echo ""
echo "Checked $files_checked files."

if [ "$violations_found" -eq 0 ] || [ ! -s "$tmpfile" ]; then
    echo ""
    echo "======================================================================"
    echo "clang-tidy analysis complete: No issues found in core source files."
    echo "======================================================================"
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "clang-tidy found potential issues in core source files:"
    echo "======================================================================"
    cat "$tmpfile"
    echo ""
    echo "======================================================================"
    echo "Please review and fix the issues listed above."
    echo "Note: Some warnings may be false positives."
    echo "Note: GPU and MPI code are not checked (require special dependencies)."
    echo "======================================================================"
    echo ""
    exit 1
fi
