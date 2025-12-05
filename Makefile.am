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
TESTS_ENVIRONMENT = PIAFS_DIR=$(abs_top_builddir) PIAFS_SRC_DIR=$(abs_top_srcdir) MPIEXEC="@MPIEXEC@"
TESTS = Tests/test_mpi.sh

check-local:
	@echo "Preparing PIAFS test suite..."
	@test -d $(abs_top_builddir)/bin || mkdir -p $(abs_top_builddir)/bin
	@if test -f $(abs_top_builddir)/src/PIAFS; then \
		cp -f $(abs_top_builddir)/src/PIAFS $(abs_top_builddir)/bin/PIAFS; \
	fi
	@echo "Running PIAFS test suite..."
	@$(TESTS_ENVIRONMENT) $(abs_top_srcdir)/Tests/test_mpi.sh
