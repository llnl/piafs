#!/usr/bin/env bash

set -eu -o pipefail

# Run clang-tidy static analysis on C/C++/CUDA code files
# Exclude documentation, build artifacts, and test directories

echo "Running clang-tidy static analysis..."

# Check if clang-tidy is available
if ! command -v clang-tidy &> /dev/null; then
    echo "Error: clang-tidy not found. Please install clang-tidy."
    exit 1
fi

# Create a temporary file to store results
tmpfile=$(mktemp)
trap "rm -f $tmpfile" EXIT

violations_found=0

# Find all C/C++/CUDA files and run clang-tidy on them
find . -type d \( -name .git \
                  -o -name build -o -name install \
                  -o -name bin -o -name lib -o -name lib64 \
                  -o -path ./docs/build \
                  -o -name test_run_temp \
                  -o -name __pycache__ \
                  -o -name Examples \
                  -o -name Extras \
               \) -prune -o \
       -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.cxx" -o -name "*.cu" \) \
    -print0 | while IFS= read -r -d '' file; do
        # Skip files that are not regular text files
        if ! grep -Iq . "$file" 2>/dev/null; then
            continue
        fi

        echo "Checking: $file"

        # Run clang-tidy and capture output
        # Use --quiet to suppress unnecessary output
        # We run with a simple compilation database or without one
        if clang-tidy "$file" -- -I./include 2>&1 | grep -E "warning:|error:" >> "$tmpfile"; then
            violations_found=1
        fi
    done

if [ "$violations_found" -eq 0 ] && [ ! -s "$tmpfile" ]; then
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
    echo "Note: Some warnings may be false positives or style preferences."
    echo "======================================================================"
    echo ""
    exit 1
fi
