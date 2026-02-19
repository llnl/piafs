SUBDIRS = src
dist_doc_DATA = README
docdir = doc

EXTRA_DIST = \
  include \
  include/physicalmodels \
  Examples \
	Extras \
	Tests

# Testing support
# Regression benchmarks list (keep in sync with Tests/CMakeLists.txt)
REGRESSION_BENCHMARKS = \
	1d_beam_grating \
	1d_euler_densitysinewave \
	1d_euler_firstorder \
	1d_euler_laxshocktube \
	1d_euler_muscl2 \
	1d_euler_rk22 \
	1d_euler_rk33 \
	1d_euler_secondorder_central \
	2d_beam_grating \
	2d_ensemble_navstok_vortconv \
	2d_navstok_densitysinewave \
	2d_navstok_flatplatelaminar \
	2d_navstok_lidcav \
	2d_navstok_riemann4 \
	2d_navstok_vortconv \
	3d_beam_grating \
	3d_navstok_riemann4_xy \
	3d_navstok_riemann4_xz \
	3d_navstok_riemann4_yz

# Prepare PIAFS binary and set environment for tests
TESTS_ENVIRONMENT = \
	test -d $(abs_top_builddir)/bin || mkdir -p $(abs_top_builddir)/bin; \
	if test -f $(abs_top_builddir)/src/PIAFS@PIAFS_NAME_SUFFIX@; then \
		cp -f $(abs_top_builddir)/src/PIAFS@PIAFS_NAME_SUFFIX@ $(abs_top_builddir)/bin/PIAFS@PIAFS_NAME_SUFFIX@; \
	fi; \
	PIAFS_DIR=$(abs_top_builddir) \
	PIAFS_SRC_DIR=$(abs_top_srcdir) \
	MPIEXEC="@MPIEXEC@" \
	PIAFS_EXEC_W_PATH=$(abs_top_builddir)/bin/PIAFS@PIAFS_NAME_SUFFIX@ \
	MPI_EXEC="@MPIEXEC@" \
	BENCHMARKS_DIR=$(abs_top_builddir)/test_run_temp/benchmarks \
	EXPECTED_BENCHMARKS="$(REGRESSION_BENCHMARKS)"

# Run all regression tests through wrapper script with individual reporting
# For out-of-source builds, create symlink to test scripts in build directory
check-local:
	@test -d Tests || mkdir -p Tests
	@test -d Tests/autotools || mkdir -p Tests/autotools
	@for script in run_all_tests.sh setup_benchmarks.sh run_regression_test.sh generate-test-scripts.sh regression_test_wrapper.sh; do \
		if test ! -e Tests/autotools/$$script; then \
			ln -sf $(abs_top_srcdir)/Tests/autotools/$$script Tests/autotools/$$script; \
		fi; \
	done

TESTS = Tests/autotools/run_all_tests.sh
