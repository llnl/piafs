AC_INIT([piafs2d],[0.1],[debojyoti.ghosh@gmail.com])
AC_CONFIG_SRCDIR([config.h.in])
AC_CONFIG_HEADERS([config.h])
echo -n AC_PACKAGE_VERSION>VERSION
AC_SUBST(PACKAGE_VERSION)
AC_MSG_NOTICE([piafs2d AC_PACKAGE_VERSION])

AM_INIT_AUTOMAKE
AC_PREFIX_DEFAULT('$(top_srcdir)')
includedir="${ac_pwd}/${srcdir}/include"
if ! test -d ${includedir} ; then
  includedir="${srcdir}/include"
fi

AC_ARG_ENABLE([serial],AS_HELP_STRING([--enable-serial],[Enable serial mode]))
AC_ARG_WITH([mpi_dir],AS_HELP_STRING([--with-mpi-dir],[Specify path where MPI is installed.]))
AC_ARG_ENABLE([omp],AS_HELP_STRING([--enable-omp],[Enable OpenMP threads]))

AC_PROG_CC
AC_PROG_CXX

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
   src/PhysicalModels/Euler1D/Makefile
   src/PhysicalModels/NavierStokes2D/Makefile
 src/Simulation/Makefile
 src/TimeIntegration/Makefile
 src/TridiagLU/Makefile
])

AC_OUTPUT
