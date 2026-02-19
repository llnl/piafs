#!/usr/bin/env bash

set -eu -o pipefail

# Check that code files end with a newline (POSIX standard)
# Exclude documentation, input files, and data files

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
    -exec bash -c 'for file; do
        if [ -n "$(tail -c 1 "$file")" ]; then
            echo "" >> "$file"
        fi
    done' bash {} +

gitdiff=`git diff`

if [ -z "$gitdiff" ]
then
    exit 0
else
    echo -e "\nAll code files must end with a newline (POSIX standard). Apply this patch to fix:"
    echo -e "  git apply newline_eof.patch\n"
    echo -e "Or apply directly with:"
    echo -e "  curl <CI-URL> | git apply\n"
    echo "====== BEGIN PATCH ======"
    git --no-pager diff
    echo "====== END PATCH ======"
    echo ""
    exit 1
fi
