#!/usr/bin/env bash

set -eu -o pipefail

# Check that indentation uses consistent spacing (multiples of 2 spaces)
# Only check code files (C, C++, Python, shell scripts)

INDENT_SIZE=2

check_file_indentation() {
    local file="$1"
    local violations=0
    local line_num=0

    while IFS= read -r line; do
        line_num=$((line_num + 1))

        # Skip empty lines and lines that are all whitespace
        if [[ -z "${line// }" ]]; then
            continue
        fi

        # Extract leading whitespace
        leading_space="${line%%[! ]*}"

        # Skip if no leading whitespace
        if [[ "$leading_space" == "$line" ]] && [[ -n "$line" ]]; then
            leading_space=""
        fi

        # Check if leading whitespace contains tabs (should be caught by tab check, but just in case)
        if [[ "$leading_space" =~ $'\t' ]]; then
            continue  # Skip, will be caught by tab check
        fi

        # Count spaces
        space_count=${#leading_space}

        # Check if indentation is a multiple of INDENT_SIZE
        if (( space_count > 0 )) && (( space_count % INDENT_SIZE != 0 )); then
            echo "$file:$line_num: Indentation is $space_count spaces (not a multiple of $INDENT_SIZE)"
            violations=$((violations + 1))
        fi
    done < "$file"

    return $violations
}

echo "Checking indentation (must be multiples of $INDENT_SIZE spaces)..."

total_violations=0
files_with_violations=""

while IFS= read -r -d '' file; do
    if check_file_indentation "$file"; then
        :  # No violations
    else
        violations=$?
        total_violations=$((total_violations + violations))
        files_with_violations="$files_with_violations$file "
    fi
done < <(find . -type d \( -name .git \
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
                          -o -name "*.sh" \) \
                         -a \( ! -name "Doxyfile" \) \
                      \) -print0)

if (( total_violations == 0 )); then
    echo "✓ All files have consistent indentation"
    exit 0
else
    echo ""
    echo "✗ Found $total_violations indentation violations in the following files:"
    echo "$files_with_violations" | tr ' ' '\n' | grep -v '^$'
    echo ""
    echo "Fix by ensuring all indentation uses multiples of $INDENT_SIZE spaces."
    echo "Most editors can auto-format code to fix indentation."
    echo ""
    exit 1
fi
