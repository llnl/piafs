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
# Prepare PIAFS binary before running tests
TESTS_ENVIRONMENT = \
	test -d $(abs_top_builddir)/bin || mkdir -p $(abs_top_builddir)/bin; \
	if test -f $(abs_top_builddir)/src/PIAFS@PIAFS_NAME_SUFFIX@; then \
		cp -f $(abs_top_builddir)/src/PIAFS@PIAFS_NAME_SUFFIX@ $(abs_top_builddir)/bin/PIAFS@PIAFS_NAME_SUFFIX@; \
	fi; \
	PIAFS_DIR=$(abs_top_builddir) PIAFS_SRC_DIR=$(abs_top_srcdir) MPIEXEC="@MPIEXEC@"

TESTS = Tests/test_mpi.sh
