#!/bin/bash

# Generate individual test wrapper scripts for each regression benchmark
# This script is called during the build to create test_<benchmark>.sh scripts

if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_dir> [benchmark1 benchmark2 ...]"
  exit 1
fi

OUTPUT_DIR="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$OUTPUT_DIR"

# Generate wrapper script for each benchmark
for benchmark in "$@"; do
  cat > "$OUTPUT_DIR/test_${benchmark}.sh" << WRAPPER
#!/bin/bash
exec "$SCRIPT_DIR/run_regression_test.sh" "$benchmark"
WRAPPER
  chmod +x "$OUTPUT_DIR/test_${benchmark}.sh"
done

echo "Generated $(($# + 1)) test scripts in $OUTPUT_DIR"
