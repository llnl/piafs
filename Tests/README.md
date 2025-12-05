# PIAFS Test Suite

This directory contains the PIAFS regression test suite that compares simulation outputs against benchmark solutions.

## Overview

The test suite:
1. Clones benchmark solutions from a git repository
2. Runs PIAFS on test cases
3. Compares output files against benchmarks using a custom diff tool
4. Reports pass/fail status for each test case

## Running Tests

### Using Autotools (make)

After building with autotools:

```bash
# Configure and build
autoreconf -i  # Only needed once, or after modifying build files
./configure
make

# Run the test suite
make check

# Custom MPI executor (for HPC platforms)
./configure --with-mpiexec="srun"          # For Slurm
./configure --with-mpiexec="mpirun -np"    # For standard mpirun
./configure --with-mpiexec="jsrun"         # For IBM JSRun
make check

# The test log is saved to Tests/test_mpi.sh.log
# View the full test output:
cat Tests/test_mpi.sh.log
```

**Notes:**
- The autotools build system automatically copies the PIAFS binary from `src/PIAFS` to `bin/PIAFS` before running tests.
- Out-of-source builds are fully supported. The test system automatically locates source files (like the diff tool) in the source directory while using binaries from the build directory.
- For out-of-source builds, run `make check` from your build directory.
- Use `--with-mpiexec` to specify a custom MPI launcher for HPC environments (default: `mpiexec`).

### Using CMake

After building with CMake:

```bash
# Configure and build
mkdir build && cd build
cmake ..
make

# Run the test suite (any of these commands)
make test
ctest
ctest --verbose  # for detailed output
ctest --output-on-failure  # show output only for failed tests

# Custom MPI executor (for HPC platforms)
cmake -DMPIEXEC="srun" ..                  # For Slurm
cmake -DMPIEXEC="mpirun -np" ..            # For standard mpirun
cmake -DMPIEXEC="jsrun" ..                 # For IBM JSRun
make
ctest --verbose
```

### Manual Execution

You can also run the test script directly:

```bash
# Set the PIAFS_DIR environment variable
export PIAFS_DIR=/path/to/piafs/build/directory

# Run the test script
cd Tests
./test_mpi.sh
```

## Environment Variables

The test script supports the following environment variables:

- `PIAFS_DIR`: Required. Path to PIAFS build directory containing the `bin/PIAFS` executable
- `PIAFS_SRC_DIR`: Optional. Path to PIAFS source directory (automatically set by build systems for out-of-source builds)
- `MPIEXEC`: Optional. MPI executor command for running parallel tests (default: `mpiexec --oversubscribe`)
  - Examples: `mpiexec`, `mpirun`, `srun`, `jsrun`
  - Set via `--with-mpiexec` (autotools) or `-DMPIEXEC` (CMake)
- `PIAFS_EXEC_OTHER_ARGS`: Optional. Additional arguments to pass to PIAFS

## Test Structure

Each test case directory in the benchmarks repository should contain:
- Input files for the simulation
- `run.sh`: Script to execute the test case
- `diff_file_list`: List of output files to compare against benchmarks
- `.disabled`: Optional file to disable a test

## Benchmark Repository

The test suite automatically clones benchmarks from:
- Repository: `ssh://git@czgitlab.llnl.gov:7999/piafs/piafs_benchmarks.git`
- Branch: `master`

## Test Output

Test results are stored in timestamped directories: `_test_<timestamp>/`

The test suite reports:
- Number of tests passed
- Number of tests failed
- Number of tests skipped
- Detailed diff output for failed tests

Exit code:
- `0`: All tests passed
- `1`: One or more tests failed

## Comparison Tool

The test suite compiles and uses `PIAFS_DIFF` (from `Extras/piafsDiff_RegTests.c`) to compare output files with a relative tolerance of 1.0e-14.

## Continuous Integration

The test suite is integrated into both build systems and can be used for automated testing in CI/CD pipelines.
