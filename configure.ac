AC_INIT([piafs],[0.1],[debojyoti.ghosh@gmail.com])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])
echo -n AC_PACKAGE_VERSION>VERSION
AC_SUBST(PACKAGE_VERSION)
AC_MSG_NOTICE([piafs AC_PACKAGE_VERSION])

AM_INIT_AUTOMAKE([subdir-objects])
PKG_PROG_PKG_CONFIG
AC_PREFIX_DEFAULT('$(top_srcdir)')
includedir="${ac_pwd}/${srcdir}/include"
if ! test -d ${includedir} ; then
  includedir="${srcdir}/include"
fi

AC_ARG_ENABLE([serial],AS_HELP_STRING([--enable-serial],[Enable serial mode]))
AC_ARG_WITH([mpi_dir],AS_HELP_STRING([--with-mpi-dir],[Specify path where MPI is installed.]))
AC_ARG_WITH([mpiexec],AS_HELP_STRING([--with-mpiexec],[Specify MPI run command (e.g., mpirun, mpiexec, srun) for tests.]))
AC_ARG_ENABLE([omp],AS_HELP_STRING([--enable-omp],[Enable OpenMP threads]))

AC_PROG_CC
AC_PROG_CXX

CFLAGS="$CFLAGS -Wno-alloc-size-larger-than"
CXXFLAGS="$CXXFLAGS -Wno-alloc-size-larger-than"

CFLAGS="$CFLAGS -I$includedir"
if test "$GCC" = "yes" ; then
  CFLAGS="$CFLAGS -std=c99"
  LIBS="$LIBS -lm"
  MPI_CC=mpicc
else
  AC_MSG_WARN([Compilation with non-GNU compilers not supported.])
fi

CXXFLAGS="$CXXFLAGS -I$includedir"
if test "$GXX" = "yes" ; then
  CXXFLAGS="$CXXFLAGS -std=c++11"
  LIBS="$LIBS -lm"
  MPI_CXX=mpicxx
else
  AC_MSG_WARN([Compilation with non-GNU compilers not supported.])
fi

if test "x$enable_serial" = "xyes"; then
  CFLAGS="$CFLAGS -Dserial"
  CXXFLAGS="$CXXFLAGS -Dserial"
fi

if test "x$enable_omp" = "xyes" ; then
  AC_MSG_NOTICE([Compiling with OpenMP support.])
  if test "$CC" = "gcc"  ; then
    CFLAGS="$CFLAGS -fopenmp"
    LIBS="$LIBS -lgomp"
  fi
  CXXFLAGS="$CXXFLAGS -Dwith_omp"
  if test "$CXX" = "g++"  ; then
    CXXFLAGS="$CXXFLAGS -fopenmp"
    LIBS="$LIBS -lgomp"
  fi
  CXXFLAGS="$CXXFLAGS -Dwith_omp"
fi

if test "x$enable_serial" = "x"; then

  if test "x$with_mpi_dir" != "x" ; then
    AC_MSG_CHECKING([for ${MPI_CC} in $with_mpi_dir/bin/])
    if test -e $with_mpi_dir/bin/${MPI_CC} ; then
      AC_MSG_RESULT([yes])
      MPICC=$with_mpi_dir/bin/${MPI_CC}
      MPI_DIR=$with_mpi_dir
    else
      AC_MSG_RESULT([no])
    fi
    AC_MSG_CHECKING([for ${MPI_CXX} in $with_mpi_dir/bin/])
    if test -e $with_mpi_dir/bin/${MPI_CXX} ; then
      AC_MSG_RESULT([yes])
      MPICXX=$with_mpi_dir/bin/${MPI_CXX}
    else
      AC_MSG_RESULT([no])
    fi
  else
    if test "x$MPI_DIR" != "x" ; then
      AC_MSG_NOTICE([Found environment variable MPI_DIR=${MPI_DIR}.])
      AC_MSG_CHECKING([for ${MPI_CC} in ${MPI_DIR}/bin/])
      if test -e ${MPI_DIR}/bin/${MPI_CC} ; then
        AC_MSG_RESULT([yes])
        MPICC=${MPI_DIR}/bin/${MPI_CC}
      else
        AC_MSG_RESULT([no])
      fi
      AC_MSG_CHECKING([for ${MPI_CXX} in ${MPI_DIR}/bin/])
      if test -e ${MPI_DIR}/bin/${MPI_CXX} ; then
        AC_MSG_RESULT([yes])
        MPICXX=${MPI_DIR}/bin/${MPI_CXX}
      else
        AC_MSG_RESULT([no])
      fi
    fi
  fi

  if test "x$MPICC" = "x" ; then
    AC_CHECK_PROG(MPICC,${MPI_CC},${MPI_CC})
  fi
  if test "x$MPICXX" = "x" ; then
    AC_CHECK_PROG(MPICXX,${MPI_CXX},${MPI_CXX})
  fi

fi
  
if test "x$enable_serial" = "x"; then

  if test "x$MPICC" = "x" ; then
    AC_MSG_WARN([Cannot find ${MPI_CC}. Will compile with gcc. Use --with-mpi-dir to specify the location of ${MPI_CC}.])
    CFLAGS="$CFLAGS -Dserial"
    flag_serial="yes"
  else
    CC=$MPICC
  fi

  if test "x$MPICXX" = "x" ; then
    AC_MSG_WARN([Cannot find ${MPI_CXX}. Will compile with gcc. Use --with-mpi-dir to specify the location of ${MPI_CXX}.])
    CXXFLAGS="$CXXFLAGS -Dserial"
    flag_serial="yes"
  else
    CXX=$MPICXX
  fi

fi

AC_PROG_RANLIB
AC_PROG_INSTALL

dnl Build binary name suffix from configuration
PIAFS_NAME_SUFFIX=""

dnl Add compiler info
if test "$GXX" = "yes" ; then
  PIAFS_NAME_SUFFIX="${PIAFS_NAME_SUFFIX}-gcc"
elif test "$CXX" = "clang++" -o "$CXX" = "clang" ; then
  PIAFS_NAME_SUFFIX="${PIAFS_NAME_SUFFIX}-clang"
else
  dnl Extract compiler name from CXX
  CXX_BASENAME=`basename "$CXX" 2>/dev/null`
  PIAFS_NAME_SUFFIX="${PIAFS_NAME_SUFFIX}-${CXX_BASENAME}"
fi

dnl Add MPI/serial info
if test "x$enable_serial" = "xyes"; then
  PIAFS_NAME_SUFFIX="${PIAFS_NAME_SUFFIX}-serial"
else
  PIAFS_NAME_SUFFIX="${PIAFS_NAME_SUFFIX}-mpi"
fi

dnl Add OpenMP info
if test "x$enable_omp" = "xyes"; then
  PIAFS_NAME_SUFFIX="${PIAFS_NAME_SUFFIX}-omp"
fi

dnl Add GPU info (autotools doesn't currently support GPU, but add for consistency)
dnl This would need to be set based on configure options if GPU support is added
dnl For now, leave empty - can be extended later

AC_SUBST(PIAFS_NAME_SUFFIX)

dnl Get git information
AC_CHECK_PROG([GIT_FOUND], [git], [yes], [no])
PIAFS_GIT_HASH="unknown"
PIAFS_GIT_BRANCH="unknown"
PIAFS_GIT_DIRTY="unknown"
if test "$GIT_FOUND" = "yes"; then
  PIAFS_GIT_HASH=`git rev-parse --short HEAD 2>/dev/null || echo "unknown"`
  PIAFS_GIT_BRANCH=`git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown"`
  if test -n "`git status --porcelain 2>/dev/null`"; then
    PIAFS_GIT_DIRTY="yes"
  else
    PIAFS_GIT_DIRTY="no"
  fi
fi
AC_SUBST(PIAFS_GIT_HASH)
AC_SUBST(PIAFS_GIT_BRANCH)
AC_SUBST(PIAFS_GIT_DIRTY)

dnl Get build date
PIAFS_BUILD_DATE=`date -u +"%Y-%m-%d %H:%M:%S" 2>/dev/null || date +"%Y-%m-%d %H:%M:%S" || echo "unknown"`
AC_SUBST(PIAFS_BUILD_DATE)

dnl Get compiler version
PIAFS_COMPILER_VERSION="unknown"
if test "$GXX" = "yes"; then
  PIAFS_COMPILER_VERSION=`$CXX --version | head -n1 | sed 's/.*version //' | sed 's/ .*//' 2>/dev/null || echo "unknown"`
fi
AC_SUBST(PIAFS_COMPILER_VERSION)

dnl Get MPI version
PIAFS_MPI_VERSION="unknown"
if test "x$enable_serial" != "xyes"; then
  dnl Try to get MPI version from mpicc or mpicxx
  if test -n "$MPICC"; then
    PIAFS_MPI_VERSION=`$MPICC --version 2>/dev/null | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1 || echo "unknown"`
  fi
  if test "$PIAFS_MPI_VERSION" = "unknown" -a -n "$MPICXX"; then
    PIAFS_MPI_VERSION=`$MPICXX --version 2>/dev/null | head -n1 | grep -o '[0-9]\+\.[0-9]\+' | head -n1 || echo "unknown"`
  fi
fi
AC_SUBST(PIAFS_MPI_VERSION)

dnl Detect MPI implementation - skip regression tests for OpenMPI
dnl (benchmarks were generated with MPICH)
SKIP_REGRESSION_TESTS="no"
if test "x$enable_serial" != "xyes"; then
  AC_MSG_CHECKING([MPI implementation])
  if test -n "$MPICC"; then
    MPI_IMPL_CHECK=`$MPICC --version 2>/dev/null | head -n1`
    if echo "$MPI_IMPL_CHECK" | grep -q "Open MPI"; then
      AC_MSG_RESULT([OpenMPI detected])
      AC_MSG_NOTICE([Skipping regression tests - benchmarks were generated with MPICH])
      SKIP_REGRESSION_TESTS="yes"
    else
      AC_MSG_RESULT([non-OpenMPI (likely MPICH or compatible)])
    fi
  else
    AC_MSG_RESULT([unknown])
  fi
fi
AM_CONDITIONAL([SKIP_REGRESSION_TESTS], [test "x$SKIP_REGRESSION_TESTS" = "xyes"])

dnl Set build configuration
if test "x$enable_serial" = "xyes"; then
  PIAFS_MPI_MODE="serial"
else
  PIAFS_MPI_MODE="mpi"
fi
AC_SUBST(PIAFS_MPI_MODE)

if test "x$enable_omp" = "xyes"; then
  PIAFS_OPENMP="enabled"
else
  PIAFS_OPENMP="disabled"
fi
AC_SUBST(PIAFS_OPENMP)

dnl Get compiler name
PIAFS_COMPILER="unknown"
if test "$GXX" = "yes"; then
  PIAFS_COMPILER="gcc"
elif test "$CXX" = "clang++" -o "$CXX" = "clang"; then
  PIAFS_COMPILER="clang"
else
  PIAFS_COMPILER=`basename "$CXX" 2>/dev/null || echo "unknown"`
fi
AC_SUBST(PIAFS_COMPILER)

dnl Set build type (default to Release for autotools)
PIAFS_BUILD_TYPE="Release"
AC_SUBST(PIAFS_BUILD_TYPE)

dnl Set MPI executor for tests
if test "x$with_mpiexec" != "x" ; then
  MPIEXEC="$with_mpiexec"
  AC_MSG_NOTICE([Using custom MPI executor for tests: $MPIEXEC])
else
  dnl Auto-detect MPI launcher like CMake does
  AC_MSG_CHECKING([for MPI executor])

  dnl Check for Slurm's srun
  AC_PATH_PROG([SRUN], [srun], [no])
  if test "x$SRUN" != "xno" ; then
    MPIEXEC="$SRUN"
    AC_MSG_RESULT([found srun (Slurm)])
  else
    dnl Check for LSF's jsrun (IBM systems like Summit)
    AC_PATH_PROG([JSRUN], [jsrun], [no])
    if test "x$JSRUN" != "xno" ; then
      MPIEXEC="$JSRUN"
      AC_MSG_RESULT([found jsrun (LSF)])
    else
      dnl Check for standard mpiexec
      AC_PATH_PROG([MPIEXEC_BIN], [mpiexec], [no])
      if test "x$MPIEXEC_BIN" != "xno" ; then
        MPIEXEC="$MPIEXEC_BIN"
        AC_MSG_RESULT([found mpiexec])
      else
        dnl Check for mpirun as fallback
        AC_PATH_PROG([MPIRUN], [mpirun], [no])
        if test "x$MPIRUN" != "xno" ; then
          MPIEXEC="$MPIRUN"
          AC_MSG_RESULT([found mpirun])
        else
          MPIEXEC="mpiexec"
          AC_MSG_RESULT([not found, defaulting to mpiexec])
          AC_MSG_WARN([MPI executor not found. Tests may fail. Use --with-mpiexec to specify.])
        fi
      fi
    fi
  fi
fi
AC_SUBST(MPIEXEC)

dnl Unit tests are only supported via CMake build system
dnl (CMake auto-downloads compatible Google Test version)

dnl Generate build_info.h
dnl Set PIAFS_VERSION from PACKAGE_VERSION for substitution
PIAFS_VERSION="${PACKAGE_VERSION}"
AC_SUBST(PIAFS_VERSION)
AC_CONFIG_FILES([build_info.h:build_info.h.in])

AC_CONFIG_FILES([
 Makefile
 src/Makefile
 src/ArrayFunctions/Makefile
 src/BoundaryConditions/Makefile
 src/CommonFunctions/Makefile
 src/FirstDerivative/Makefile
 src/HyParFunctions/Makefile
 src/InterpolationFunctions/Makefile
 src/IOFunctions/Makefile
 src/LimiterFunctions/Makefile
 src/MathFunctions/Makefile
 src/MPIFunctions/Makefile
 src/PhysicalModels/Makefile
   src/PhysicalModels/Chemistry/Makefile
   src/PhysicalModels/Euler1D/Makefile
   src/PhysicalModels/NavierStokes2D/Makefile
   src/PhysicalModels/NavierStokes3D/Makefile
 src/Simulation/Makefile
 src/TimeIntegration/Makefile
 src/TridiagLU/Makefile
])

AC_OUTPUT
