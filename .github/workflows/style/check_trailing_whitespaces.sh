#!/usr/bin/env bash

set -eu -o pipefail

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
                  -o -name "*.md" -o -name "*.rst" \
                  -o -name "*.sh" \
                  -o -name "*.txt" \
                  -o -name "*.yml" \
                  -o -name "*.yaml" \
                  -o -name "*.inp" \) \
                 -a \( ! -name "Doxyfile" \) \
              \) \
    -exec grep -Iq . {} \; \
    -exec sed -i 's/[[:blank:]]\+$//g' {} +

gitdiff=`git diff`

if [ -z "$gitdiff" ]
then
    exit 0
else
    echo -e "\nTrailing whitespaces at the end of a line are not allowed. Changes suggested by"
    echo -e "  ${0}\n"
    git --no-pager diff
    echo ""
    exit 1
fi
