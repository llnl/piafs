# Unit tests using Google Test
#
# NOTE: The system Google Test may require C++14 or C++17, which can conflict
# with the project's C++11 requirement. If unit tests fail to compile with
# autotools, use CMake instead (recommended):
#   mkdir build && cd build && cmake .. && make && ctest
#
# CMake automatically downloads a compatible Google Test version.

if HAVE_GTEST

# Include directories
AM_CPPFLAGS = \
	-I$(top_srcdir)/include \
	-I$(top_srcdir)/include/physicalmodels \
	$(GTEST_CFLAGS)

# Test programs
check_PROGRAMS = test_array_functions test_math_functions test_common_functions test_boundary_conditions test_first_derivative test_interpolation_functions test_limiter_functions test_io_functions test_time_integration test_tridiag_lu

# Test array functions
test_array_functions_SOURCES = ArrayFunctions/test_array_functions.cpp
test_array_functions_CXXFLAGS = -std=c++17
test_array_functions_LDADD = \
	$(top_builddir)/src/ArrayFunctions/libArrayFunctions.a \
	$(top_builddir)/src/MathFunctions/libMathFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test math functions
test_math_functions_SOURCES = MathFunctions/test_math_functions.cpp
test_math_functions_CXXFLAGS = -std=c++17
test_math_functions_LDADD = \
	$(top_builddir)/src/MathFunctions/libMathFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test common functions
test_common_functions_SOURCES = CommonFunctions/test_common_functions.cpp
test_common_functions_CXXFLAGS = -std=c++17
test_common_functions_LDADD = \
	$(top_builddir)/src/CommonFunctions/libCommonFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test boundary functions
test_boundary_conditions_SOURCES = BoundaryConditions/test_boundary_conditions.cpp
test_boundary_conditions_CXXFLAGS = -std=c++17
test_boundary_conditions_LDADD = \
	$(top_builddir)/src/BoundaryConditions/libBoundaryConditions.a \
	$(top_builddir)/src/ArrayFunctions/libArrayFunctions.a \
	$(top_builddir)/src/MathFunctions/libMathFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test first derivative functions
test_first_derivative_SOURCES = FirstDerivative/test_first_derivative.cpp
test_first_derivative_CXXFLAGS = -std=c++17
test_first_derivative_LDADD = \
	$(top_builddir)/src/FirstDerivative/libFirstDerivative.a \
	$(top_builddir)/src/ArrayFunctions/libArrayFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test interpolation functions
test_interpolation_functions_SOURCES = InterpolationFunctions/test_interpolation_functions.cpp
test_interpolation_functions_CXXFLAGS = -std=c++17
test_interpolation_functions_LDADD = \
	$(top_builddir)/src/InterpolationFunctions/libInterpolationFunctions.a \
	$(top_builddir)/src/ArrayFunctions/libArrayFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test limiter functions
test_limiter_functions_SOURCES = LimiterFunctions/test_limiter_functions.cpp
test_limiter_functions_CXXFLAGS = -std=c++17
test_limiter_functions_LDADD = \
	$(top_builddir)/src/LimiterFunctions/libLimiterFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test IO functions
test_io_functions_SOURCES = IOFunctions/test_io_functions.cpp
test_io_functions_CXXFLAGS = -std=c++17
test_io_functions_LDADD = \
	$(top_builddir)/src/IOFunctions/libIOFunctions.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test time integration functions
test_time_integration_SOURCES = TimeIntegration/test_time_integration.cpp
test_time_integration_CXXFLAGS = -std=c++17
test_time_integration_LDADD = \
	$(top_builddir)/src/TimeIntegration/libTimeIntegration.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Test tridiagonal LU solver functions
test_tridiag_lu_SOURCES = TridiagLU/test_tridiag_lu.cpp
test_tridiag_lu_CXXFLAGS = -std=c++17
test_tridiag_lu_LDADD = \
	$(top_builddir)/src/TridiagLU/libTridiagLU.a \
	$(GTEST_LIBS) \
	-lpthread \
	-lm

# Register tests with 'make check'
TESTS = $(check_PROGRAMS)

endif # HAVE_GTEST
