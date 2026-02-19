#!/usr/bin/env bash

set -eu -o pipefail

# Run clang-tidy static analysis on C/C++/CUDA code files
# Includes GPU code (CUDA/HIP) and MPI code

echo "Running clang-tidy static analysis..."

# Check if clang-tidy is available
if ! command -v clang-tidy &> /dev/null; then
    echo "Error: clang-tidy not found. Please install clang-tidy."
    exit 1
fi

# Set CUDA path if not already set
if [ -z "${CUDA_PATH:-}" ]; then
    CUDA_PATH="/usr/local/cuda"
fi

echo "Using CUDA_PATH: $CUDA_PATH"

# Create a temporary file to store results
tmpfile=$(mktemp)
trap "rm -f $tmpfile" EXIT

violations_found=0
files_checked=0

# Find all C/C++/CUDA files except in Examples, Extras, and Tests
find . -type d \( -name .git \
                  -o -name build -o -name install \
                  -o -name bin -o -name lib -o -name lib64 \
                  -o -path ./docs/build \
                  -o -name test_run_temp \
                  -o -name __pycache__ \
                  -o -name Examples \
                  -o -name Extras \
                  -o -name Tests \
               \) -prune -o \
       -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.cxx" -o -name "*.cu" \) \
    -print0 | while IFS= read -r -d '' file; do
        # Skip files that are not regular text files
        if ! grep -Iq . "$file" 2>/dev/null; then
            continue
        fi

        files_checked=$((files_checked + 1))
        echo "Checking: $file"

        # Determine if this is a CUDA file
        if [[ "$file" == *.cu ]]; then
            # CUDA file - use CUDA-specific flags
            clang_flags="--cuda-path=$CUDA_PATH --cuda-gpu-arch=sm_70 -x cuda -I./include"
        else
            # Regular C/C++ file
            clang_flags="-I./include"
        fi

        # Run clang-tidy and capture warnings
        # Filter out gtest errors (Tests directory already excluded)
        if clang-tidy "$file" -- $clang_flags 2>&1 | \
           grep -E "warning:" | \
           grep -v "gtest/gtest.h" >> "$tmpfile"; then
            violations_found=1
        fi
    done

echo ""
echo "Checked $files_checked files."

if [ "$violations_found" -eq 0 ] || [ ! -s "$tmpfile" ]; then
    echo ""
    echo "======================================================================"
    echo "clang-tidy analysis complete: No issues found."
    echo "======================================================================"
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "clang-tidy found potential issues:"
    echo "======================================================================"
    cat "$tmpfile"
    echo ""
    echo "======================================================================"
    echo "Please review and fix the issues listed above."
    echo "Note: Some warnings may be false positives."
    echo "======================================================================"
    echo ""
    exit 1
fi
