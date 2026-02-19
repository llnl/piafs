#!/usr/bin/env bash

set -eu -o pipefail

# Check that header files use proper include guards
# Format: #ifndef FILENAME_H / #define FILENAME_H / #endif
# Exclude documentation, input files, and data files

violations_found=0

echo "Checking include guards in header files..."

find . -type d \( -name .git \
                  -o -name build -o -name install \
                  -o -name bin -o -name lib -o -name lib64 \
                  -o -path ./docs/build \
                  -o -name test_run_temp \
                  -o -name __pycache__ \
               \) -prune -o \
       -type f -name "*.h" \
    -exec grep -Iq . {} \; \
    -print0 | while IFS= read -r -d '' file; do
        # Get expected guard name from filename
        basename=$(basename "$file" .h)
        # Convert to uppercase and replace special chars with underscore
        expected_guard=$(echo "${basename}_H" | tr '[:lower:]' '[:upper:]' | sed 's/[^A-Z0-9_]/_/g')

        # Check if file has proper include guard structure
        has_ifndef=$(grep -q "^#ifndef ${expected_guard}$" "$file" && echo "yes" || echo "no")
        has_define=$(grep -q "^#define ${expected_guard}$" "$file" && echo "yes" || echo "no")
        has_endif=$(grep -q "^#endif.*${expected_guard}" "$file" && echo "yes" || echo "no")

        if [ "$has_ifndef" != "yes" ] || [ "$has_define" != "yes" ] || [ "$has_endif" != "yes" ]; then
            echo ""
            echo "File: $file"
            echo "  Expected include guard: $expected_guard"
            if [ "$has_ifndef" != "yes" ]; then
                echo "  Missing: #ifndef $expected_guard"
            fi
            if [ "$has_define" != "yes" ]; then
                echo "  Missing: #define $expected_guard"
            fi
            if [ "$has_endif" != "yes" ]; then
                echo "  Missing: #endif /* $expected_guard */"
            fi
            violations_found=1
        fi
    done

if [ "$violations_found" -eq 0 ]
then
    echo "All header files have proper include guards."
    exit 0
else
    echo ""
    echo "======================================================================"
    echo "Header files must use include guards with the format:"
    echo "  #ifndef FILENAME_H"
    echo "  #define FILENAME_H"
    echo "  ..."
    echo "  #endif /* FILENAME_H */"
    echo ""
    echo "Please manually add or fix include guards in the files listed above."
    echo "======================================================================"
    echo ""
    exit 1
fi
