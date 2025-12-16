/*! @file test_array_functions.cpp
    @brief Unit tests for array functions
    @author Unit Test Suite
*/

#include <gtest/gtest.h>
#include <arrayfunctions.h>
#include <cmath>
#include <vector>

// Test fixture for array functions
class ArrayFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests
    }

    void TearDown() override {
        // Cleanup
    }

    // Helper function to compare arrays
    void expectArraysEqual(const double* a, const double* b, int size, double tol = 1e-14) {
        for (int i = 0; i < size; i++) {
            EXPECT_NEAR(a[i], b[i], tol) << "Arrays differ at index " << i;
        }
    }
};

// ============================================================================
// Comprehensive test for all array functions
// ============================================================================

TEST_F(ArrayFunctionsTest, ArrayFunctionsComprehensive) {
    // ========================================================================
    // Array Index Conversion Macros
    // ========================================================================

    // Test ArrayIndex1D_2D
    {
    int N = 2;
    int imax[2] = {3, 4};  // 3x4 array
    int ghost = 1;
    int i[2], index;

    // Test corner (0,0)
    i[0] = 0; i[1] = 0;
    _ArrayIndex1D_(N, imax, i, ghost, index);
    EXPECT_EQ(index, 6);  // (0+1) + (0+1)*(3+2) = 1 + 5 = 6

    // Test corner (2,3)
    i[0] = 2; i[1] = 3;
    _ArrayIndex1D_(N, imax, i, ghost, index);
    EXPECT_EQ(index, 23);  // (2+1) + (3+1)*(3+2) = 3 + 20 = 23
    }

    // Test ArrayIndex1D_3D
    {
    int N = 3;
    int imax[3] = {2, 3, 4};  // 2x3x4 array
    int ghost = 0;
    int i[3], index;

    // Test corner (0,0,0)
    i[0] = 0; i[1] = 0; i[2] = 0;
    _ArrayIndex1D_(N, imax, i, ghost, index);
    EXPECT_EQ(index, 0);

    // Test (1,2,3)
    i[0] = 1; i[1] = 2; i[2] = 3;
    _ArrayIndex1D_(N, imax, i, ghost, index);
    EXPECT_EQ(index, 23);  // 1 + 2*2 + 3*2*3 = 1 + 4 + 18 = 23
    }

    // Test ArrayIndexnD_RoundTrip
    {
    int N = 3;
    int imax[3] = {4, 5, 6};
    int ghost = 1;
    int i_orig[3] = {2, 3, 4};
    int i_computed[3];
    int index;

    // Convert to 1D
    _ArrayIndex1D_(N, imax, i_orig, ghost, index);

    // Convert back to nD
    _ArrayIndexnD_(N, index, imax, i_computed, ghost);

    // Check they match
    for (int d = 0; d < N; d++) {
        EXPECT_EQ(i_orig[d], i_computed[d]);
    }
    }

    // ========================================================================
    // Array Set/Copy/Scale Operations
    // ========================================================================

    // Test ArraySetValue
    {
    double x[5];
    _ArraySetValue_(x, 5, 3.14);

    for (int i = 0; i < 5; i++) {
        EXPECT_DOUBLE_EQ(x[i], 3.14);
    }
    }

    // Test ArrayCopy1D
    {
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double y[5];

    _ArrayCopy1D_(x, y, 5);

    expectArraysEqual(x, y, 5);
    }

    // Test ArrayScale1D
    {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double expected[4] = {2.0, 4.0, 6.0, 8.0};

    _ArrayScale1D_(x, 2.0, 4);

    expectArraysEqual(x, expected, 4);
    }

    // Test ArrayScaleCopy1D
    {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4];
    double expected[4] = {2.5, 5.0, 7.5, 10.0};

    _ArrayScaleCopy1D_(x, 2.5, y, 4);

    expectArraysEqual(y, expected, 4);
    }

    // Test ArrayAddCopy1D
    {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4];
    double expected[4] = {11.0, 12.0, 13.0, 14.0};

    _ArrayAddCopy1D_(x, 10.0, y, 4);

    expectArraysEqual(y, expected, 4);
    }

    // ========================================================================
    // Array Arithmetic Operations
    // ========================================================================

    // Test ArrayAdd1D
    {
    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double b[4] = {5.0, 6.0, 7.0, 8.0};
    double x[4];
    double expected[4] = {6.0, 8.0, 10.0, 12.0};

    _ArrayAdd1D_(x, a, b, 4);

    expectArraysEqual(x, expected, 4);
    }

    // Test ArraySubtract1D
    {
    double a[4] = {10.0, 20.0, 30.0, 40.0};
    double b[4] = {1.0, 2.0, 3.0, 4.0};
    double x[4];
    double expected[4] = {9.0, 18.0, 27.0, 36.0};

    _ArraySubtract1D_(x, a, b, 4);

    expectArraysEqual(x, expected, 4);
    }

    // Test ArrayMultiply1D
    {
    double a[4] = {1.0, 2.0, 3.0, 4.0};
    double b[4] = {2.0, 3.0, 4.0, 5.0};
    double x[4];
    double expected[4] = {2.0, 6.0, 12.0, 20.0};

    _ArrayMultiply1D_(x, a, b, 4);

    expectArraysEqual(x, expected, 4);
    }

    // Test ArraySum1D
    {
    double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double sum;

    _ArraySum1D_(x, sum, 5);

    EXPECT_DOUBLE_EQ(sum, 15.0);
    }

    // Test ArrayProduct1D
    {
    double x[4] = {2.0, 3.0, 4.0, 5.0};
    double product;

    _ArrayProduct1D_(x, 4, product);

    EXPECT_DOUBLE_EQ(product, 120.0);
    }

    // ========================================================================
    // Advanced Array Operations (AXPY, AXBY, etc.)
    // ========================================================================

    // Test ArrayAXPY
    {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {10.0, 20.0, 30.0, 40.0};
    double a = 2.0;
    double expected[4] = {12.0, 24.0, 36.0, 48.0};  // y = a*x + y

    _ArrayAXPY_(x, a, y, 4);

    expectArraysEqual(y, expected, 4);
    }

    // Test ArrayAYPX
    {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {10.0, 20.0, 30.0, 40.0};
    double a = 2.0;
    double expected[4] = {21.0, 42.0, 63.0, 84.0};  // y = a*y + x

    _ArrayAYPX_(x, a, y, 4);

    expectArraysEqual(y, expected, 4);
    }

    // Test ArrayAXBY
    {
    double x[4] = {1.0, 2.0, 3.0, 4.0};
    double y[4] = {5.0, 6.0, 7.0, 8.0};
    double z[4];
    double a = 2.0, b = 3.0;
    double expected[4] = {17.0, 22.0, 27.0, 32.0};  // z = a*x + b*y

    _ArrayAXBY_(z, a, x, b, y, 4);

    expectArraysEqual(z, expected, 4);
    }

    // Test ArrayAXBYCZ
    {
    double x[3] = {1.0, 2.0, 3.0};
    double y[3] = {4.0, 5.0, 6.0};
    double z[3] = {7.0, 8.0, 9.0};
    double w[3];
    double a = 2.0, b = 3.0, c = 1.0;
    double expected[3] = {21.0, 27.0, 33.0};  // w = a*x + b*y + c*z

    _ArrayAXBYCZ_(w, a, x, b, y, c, z, 3);

    expectArraysEqual(w, expected, 3);
    }

    // Test ArrayConvexCombination1D
    {
    double x[3] = {10.0, 20.0, 30.0};
    double y[3] = {0.0, 0.0, 0.0};
    double a[3] = {0.5, 0.25, 0.75};
    double z[3];
    double expected[3] = {5.0, 5.0, 22.5};  // z = a*x + (1-a)*y

    _ArrayConvexCombination1D_(z, a, x, y, 3);

    expectArraysEqual(z, expected, 3);
    }

    // ========================================================================
    // nD Array Functions
    // ========================================================================

    // Test ArrayCopynD_2D
    {
    int ndims = 2;
    int dim[2] = {3, 4};  // 3x4 array
    int nvars = 1;
    int g1 = 0, g2 = 0;
    int index[2];

    double x[12], y[12];
    for (int i = 0; i < 12; i++) x[i] = i * 1.0;

    int result = ArrayCopynD(ndims, x, y, dim, g1, g2, index, nvars);

    EXPECT_EQ(result, 0);
    expectArraysEqual(x, y, 12);
    }

    // Test ArrayCopynD_WithGhosts
    {
    int ndims = 1;
    int dim[1] = {3};  // 3 element array
    int nvars = 1;
    int g1 = 1, g2 = 1;
    int index[1];

    double x[5] = {-1.0, 1.0, 2.0, 3.0, -1.0};  // ghost, data, ghost
    double y[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

    int result = ArrayCopynD(ndims, x, y, dim, g1, g2, index, nvars);

    EXPECT_EQ(result, 0);
    // Only interior points should be copied
    EXPECT_DOUBLE_EQ(y[1], 1.0);
    EXPECT_DOUBLE_EQ(y[2], 2.0);
    EXPECT_DOUBLE_EQ(y[3], 3.0);
    }

    // Test ArrayMaxnD
    {
    int ndims = 1;
    int dim[1] = {5};
    int nvars = 1;
    int ghosts = 0;
    int index[1];

    double x[5] = {1.5, -3.7, 2.1, -8.2, 4.0};

    double max_val = ArrayMaxnD(nvars, ndims, dim, ghosts, index, x);

    EXPECT_DOUBLE_EQ(max_val, 8.2);  // Maximum absolute value
    }

    // Test ArraySumSquarenD
    {
    int ndims = 1;
    int dim[1] = {3};
    int nvars = 1;
    int ghosts = 0;
    int index[1];

    double x[3] = {1.0, 2.0, 3.0};

    double sum = ArraySumSquarenD(nvars, ndims, dim, ghosts, index, x);

    EXPECT_DOUBLE_EQ(sum, 14.0);  // 1 + 4 + 9
    }

    // Test ArraySumAbsnD
    {
    int ndims = 1;
    int dim[1] = {4};
    int nvars = 1;
    int ghosts = 0;
    int index[1];

    double x[4] = {1.0, -2.0, 3.0, -4.0};

    double sum = ArraySumAbsnD(nvars, ndims, dim, ghosts, index, x);

    EXPECT_DOUBLE_EQ(sum, 10.0);  // |1| + |-2| + |3| + |-4|
    }

    // ========================================================================
    // Array Increment Index
    // ========================================================================

    // Test ArrayIncrementIndex_2D
    {
    int N = 2;
    int imax[2] = {2, 3};  // 2x3 grid
    int i[2] = {0, 0};
    int done;

    std::vector<std::pair<int, int>> expected = {
        {0, 0}, {1, 0}, {0, 1}, {1, 1}, {0, 2}, {1, 2}
    };

    for (size_t idx = 0; idx < expected.size(); idx++) {
        EXPECT_EQ(i[0], expected[idx].first);
        EXPECT_EQ(i[1], expected[idx].second);
        _ArrayIncrementIndex_(N, imax, i, done);
        if (idx < expected.size() - 1) {
            EXPECT_EQ(done, 0);
        } else {
            EXPECT_EQ(done, 1);
        }
    }
    }

    // Test ArrayBlockMultiply
    {
    int n = 3;    // 3 blocks
    int bs = 2;   // block size 2
    double x[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double a[3] = {2.0, 0.5, 3.0};
    double expected[6] = {2.0, 4.0, 1.5, 2.0, 15.0, 18.0};

    _ArrayBlockMultiply_(x, a, n, bs);

    expectArraysEqual(x, expected, 6);
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
