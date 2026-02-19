#!/usr/bin/env bash

set -eu -o pipefail

# Check that lines don't exceed 120 characters in code files
# Exclude documentation, input files, and data files

MAX_LENGTH=120
violations_found=0

echo "Checking for lines longer than ${MAX_LENGTH} characters..."

find . -type d \( -name .git \
                  -o -name build -o -name install \
                  -o -name bin -o -name lib -o -name lib64 \
                  -o -path ./docs/build \
                  -o -name test_run_temp \
                  -o -name __pycache__ \
               \) -prune -o \
       -type f \( \( -name "*.h" -o -name "*.c" \
                  -o -name "*.cpp" -o -name "*.cxx" \
                  -o -name "*.cu" \
                  -o -name "*.py" \
                  -o -name "*.sh" \
                  -o -name "*.yml" \
                  -o -name "*.yaml" \) \
                 -a \( ! -name "Doxyfile" \) \
              \) \
    -exec grep -Iq . {} \; \
    -print0 | while IFS= read -r -d '' file; do
        long_lines=$(awk -v max="$MAX_LENGTH" 'length > max {print NR": "substr($0,1,80)"..."}' "$file")
        if [ -n "$long_lines" ]; then
            echo ""
            echo "File: $file"
            echo "$long_lines"
            violations_found=1
        fi
    done

if [ "$violations_found" -eq 0 ]
then
    echo "All lines are within ${MAX_LENGTH} characters."
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "Lines exceeding ${MAX_LENGTH} characters found in code files."
    echo "Please manually wrap or refactor these lines for better readability."
    echo "======================================================================"
    echo ""
    exit 1
fi
