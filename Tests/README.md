# PIAFS Testing

This directory contains the testing infrastructure for PIAFS, including both unit tests and regression tests.

## Directory Structure

- `unit/` - Unit tests for individual PIAFS components
- `cmake/` - CMake scripts for regression testing
- `autotools/` - Autotools scripts for regression testing
- `test_mpi.sh` - Legacy regression test script (deprecated, use new autotools scripts)

## Regression Testing

PIAFS supports regression testing through both CMake and Autotools build systems using a unified architecture:

- **CMake builds**: Use CTest with individual test registration for parallel execution
- **Autotools builds**: Use individual test scripts with sequential execution via `make check`

Both systems:
- Test against the same benchmark repository: `piafs_benchmarks`
- Maintain identical benchmark lists
- Use the same verification and comparison logic

### Architecture

Both build systems share a common testing architecture with parallel implementations:

#### CMake Components

1. **Tests/CMakeLists.txt** - Main regression test configuration
   - Builds the PIAFS diff utility
   - Registers individual benchmark tests
   - Defines test fixtures and dependencies

2. **Tests/cmake/SetupBenchmarks.cmake** - Benchmark setup script
   - Clones/updates the piafs_benchmarks repository
   - Verifies all expected benchmarks exist
   - Compiles the PIAFS diff utility

3. **Tests/cmake/RunRegressionTest.cmake** - Individual test runner
   - Copies benchmark files to test directory
   - Executes the test run script
   - Compares output against reference solutions
   - Reports pass/fail status

#### Autotools Components

1. **Makefile.am** - Main regression test configuration
   - Defines `REGRESSION_BENCHMARKS` list (synchronized with CMake)
   - Generates individual test wrapper scripts
   - Sets up test environment

2. **Tests/autotools/setup_benchmarks.sh** - Benchmark setup script
   - Clones/updates the piafs_benchmarks repository
   - Verifies all expected benchmarks exist
   - Compiles the PIAFS diff utility

3. **Tests/autotools/run_regression_test.sh** - Individual test runner
   - Takes benchmark name as argument
   - Copies benchmark files to test directory
   - Executes the test run script
   - Compares output against reference solutions
   - Reports pass/fail status

4. **Tests/autotools/generate-test-scripts.sh** - Test script generator
   - Creates individual test wrapper scripts for each benchmark
   - Called automatically during test preparation

### Usage

#### Running All Regression Tests

```bash
cd build
ctest -L regression
```

#### Running Setup Only

```bash
ctest -R setup_benchmarks
```

#### Running Individual Tests

```bash
ctest -R regression_<test_name>
```

For example:
```bash
ctest -R regression_1d_euler_densitysinewave
```

#### Using the Convenience Target

```bash
make test_regression
```

### Test Registration

All known regression tests are registered at CMake configure time. The list of benchmarks is defined in `Tests/CMakeLists.txt` in the `REGRESSION_BENCHMARKS` variable.

When `setup_benchmarks` runs, it:
1. Clones/updates the benchmarks repository
2. Verifies all expected benchmarks are present
3. Reports any missing or disabled benchmarks
4. Fails if any expected benchmarks are missing

To add a new regression test:
1. Add the benchmark name to `REGRESSION_BENCHMARKS` in `Tests/CMakeLists.txt`
2. Ensure the benchmark exists in the piafs_benchmarks repository
3. Reconfigure CMake: `cmake ..`

### Test Requirements

Each benchmark test must have:
- `run.sh` - Execution script that runs PIAFS
- `diff_file_list` - List of output files to compare (created by run.sh)
- Reference solution files in the benchmarks repository

Optional:
- `.disabled` - Presence of this file disables the test

### Environment Variables

The following variables are automatically set:
- `PIAFS_EXEC_W_PATH` - Path to the PIAFS executable
- `MPI_EXEC` - MPI executor command (e.g., mpiexec)

### Fixtures and Dependencies

Regression tests depend on two fixtures:
1. `prepare_binary` - Ensures PIAFS binary is built
2. `regression_benchmarks` - Ensures benchmarks are downloaded and diff tool is compiled

### Comparison Tolerance

The diff utility uses a tolerance of `1.0e-14` for floating-point comparisons.

### Output Directory

Test runs are executed in:
```
<build_dir>/test_run_temp/<test_name>/
```

Benchmarks are cloned to:
```
<build_dir>/test_run_temp/benchmarks/
```

### Running Tests with Autotools

For Autotools builds, regression tests now mirror the CMake approach:

**Run all tests:**
```bash
make check
```

This runs all 20 regression benchmarks and displays individual PASS/FAIL/SKIP status:
```
TEST 01/20: 1d_beam_grating                      PASS
TEST 02/20: 1d_euler_densitysinewave              PASS
TEST 03/20: 1d_euler_firstorder                   PASS
...
Regression Test Summary
========================================
Total:   20
Passed:  20
Failed:  0
Skipped: 0
```

**Run individual tests manually:**
```bash
# Run setup first
Tests/autotools/setup_benchmarks.sh

# Run specific benchmark test
Tests/autotools/run_regression_test.sh 1d_euler_firstorder
```

The test process:
1. `run_all_tests.sh` coordinates all testing:
   - Runs `setup_benchmarks.sh` first (clones/verifies benchmarks, compiles diff utility)
   - Executes each benchmark test via `run_regression_test.sh`
   - Displays individual PASS/FAIL/SKIP for each test
   - Provides summary with failed test list
2. Each benchmark test (`run_regression_test.sh`):
   - Copies benchmark files to isolated test directory
   - Executes the benchmark's run.sh script
   - Compares outputs against reference solutions using PIAFS_DIFF
   - Returns exit code (0=pass, 77=skip, 1=fail)

### CMake vs Autotools Testing

**CMake advantages:**
- **Parallel Execution** - Individual tests can run in parallel with `ctest -j<N>`
- **Better Integration** - Native CTest support with modern CI systems
- **Fixture Management** - Automatic dependency handling via fixtures
- **IDE Integration** - Built-in support in modern IDEs

**Autotools advantages:**
- **Traditional Workflow** - Standard `make check` interface
- **Simpler Configuration** - No CMake knowledge required
- **Backward Compatibility** - Works with older build systems

**Both systems now provide:**
- **Individual test execution** - Each benchmark runs as a separate test
- **Individual test reporting** - Pass/fail status for each benchmark
- **Benchmark verification** - All expected benchmarks checked before running
- **Identical test coverage** - Same 20 regression benchmarks
- **Identical comparison logic** - Same diff utility and tolerance (1.0e-14)
- **Test isolation** - Each test runs in a separate directory

## Available Regression Benchmarks

### Current Test Suite (20 benchmarks)

**1D Euler Tests:**
- `1d_euler_densitysinewave` - Density wave advection
- `1d_euler_firstorder` - 1st order upwind scheme (default), Sod shock tube
- `1d_euler_forwardeuler` - Forward Euler time integration (default)
- `1d_euler_laxshocktube` - Lax shock tube with CRWENO5
- `1d_euler_muscl2` - 2nd order MUSCL with limiter
- `1d_euler_rk22` - 2nd order RK time integration
- `1d_euler_rk33` - 3rd order RK time integration
- `1d_euler_secondorder_central` - 2nd order central scheme

**2D Navier-Stokes Tests:**
- `2d_navstok_densitysinewave` - 2D density wave
- `2d_navstok_flatplatelaminar` - Flat plate laminar flow
- `2d_navstok_lidcav` - Lid-driven cavity
- `2d_navstok_riemann4` - 2D Riemann problem
- `2d_navstok_vortconv` - Vortex convection

**2D Ensemble Tests:**
- `2d_ensemble_navstok_vortconv` - Ensemble vortex convection

**3D Navier-Stokes Tests:**
- `3d_navstok_riemann4_xy` - 3D Riemann (xy plane)
- `3d_navstok_riemann4_xz` - 3D Riemann (xz plane)
- `3d_navstok_riemann4_yz` - 3D Riemann (yz plane)

**Beam/Grating Tests:**
- `1d_beam_grating` - 1D beam grating
- `2d_beam_grating` - 2D beam grating
- `3d_beam_grating` - 3D beam grating

### Test Coverage

**Time Integration:** 83% (5 of 6 schemes)
- euler, rk22, rk33, rk44, ssprk3

**Hyperbolic Spatial Schemes:** 89% (8 of 9 schemes)
- 1st order upwind, 2nd order central, muscl2, muscl3, upw5, cupw5, weno5, crweno5

**Interpolation:** 100%
- components, characteristic

## Unit Testing

See `unit/README.md` for unit test documentation (if available).
