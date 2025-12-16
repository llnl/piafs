/*! @file test_math_functions.cpp
    @brief Unit tests for math functions
    @author Unit Test Suite
*/

#include <gtest/gtest.h>
#include <mathfunctions.h>
#include <cmath>
#include <vector>

// Test fixture for math functions
class MathFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Common setup for tests
    }

    void TearDown() override {
        // Cleanup
    }
};

//=============================================================================
// Comprehensive test for all math functions
// ============================================================================

TEST_F(MathFunctionsTest, MathFunctionsComprehensive) {
    // Test FindInterval_UniformGrid
    {
        double x[11];
        for (int i = 0; i < 11; i++) {
            x[i] = i * 1.0;
        }
        int N = 11;
        int imin, imax;
        FindInterval(2.0, 5.0, x, N, &imin, &imax);
        EXPECT_EQ(imin, 2);
        EXPECT_EQ(imax, 6);
    }

    // Test FindInterval_NonUniformGrid
    {
        double x[8] = {0.0, 0.5, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0};
        int N = 8;
        int imin, imax;
        FindInterval(1.5, 4.0, x, N, &imin, &imax);
        EXPECT_EQ(imin, 3);
        EXPECT_EQ(imax, 5);
    }

    // Test FindInterval_ExactBoundaries
    {
        double x[6] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
        int N = 6;
        int imin, imax;
        FindInterval(1.0, 4.0, x, N, &imin, &imax);
        EXPECT_EQ(imin, 1);
        EXPECT_EQ(imax, 5);
    }

    // Test FindInterval_SinglePoint
    {
        double x[10];
        for (int i = 0; i < 10; i++) {
            x[i] = i * 1.0;
        }
        int N = 10;
        int imin, imax;
        FindInterval(5.0, 5.0, x, N, &imin, &imax);
        EXPECT_EQ(imin, 5);
        EXPECT_EQ(imax, 6);
    }

    // Test FindInterval_OutOfBoundsLower
    {
        double x[5] = {5.0, 6.0, 7.0, 8.0, 9.0};
        int N = 5;
        int imin, imax;
        FindInterval(0.0, 2.0, x, N, &imin, &imax);
        EXPECT_EQ(imax, -1);
    }

    // Test FindInterval_OutOfBoundsUpper
    {
        double x[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
        int N = 5;
        int imin, imax;
        FindInterval(10.0, 15.0, x, N, &imin, &imax);
        EXPECT_EQ(imin, N);
    }

    // Test FindInterval_EntireGrid
    {
        double x[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
        int N = 5;
        int imin, imax;
        FindInterval(0.0, 10.0, x, N, &imin, &imax);
        EXPECT_EQ(imin, 0);
        EXPECT_EQ(imax, N);
    }

    // Test FindInterval_RefinedGrid
    {
        double x[101];
        for (int i = 0; i < 101; i++) {
            x[i] = i * 0.01;
        }
        int N = 101;
        int imin, imax;
        FindInterval(0.25, 0.75, x, N, &imin, &imax);
        EXPECT_EQ(imin, 25);
        EXPECT_EQ(imax, 76);
    }

    // Test FindInterval_NegativeValues
    {
        double x[7] = {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0};
        int N = 7;
        int imin, imax;
        FindInterval(-1.5, 1.5, x, N, &imin, &imax);
        EXPECT_EQ(imin, 2);
        EXPECT_EQ(imax, 5);
    }

    // Test FindInterval_InvertedInterval
    {
        double x[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
        int N = 5;
        int imin, imax;
        FindInterval(3.0, 1.0, x, N, &imin, &imax);
        EXPECT_TRUE(imin >= imax || imin == N);
    }

    // Test FindInterval_VerySmallInterval
    {
        double x[5] = {0.0, 1.0, 2.0, 3.0, 4.0};
        int N = 5;
        int imin, imax;
        FindInterval(1.3, 1.7, x, N, &imin, &imax);
        EXPECT_GE(imin, 1);
        EXPECT_LE(imin, 2);
        EXPECT_GE(imax, 1);
        EXPECT_LE(imax, 3);
    }

    // Test FindInterval_ExponentialGrid
    {
        double x[6] = {1.0, 2.0, 4.0, 8.0, 16.0, 32.0};
        int N = 6;
        int imin, imax;
        FindInterval(3.0, 10.0, x, N, &imin, &imax);
        EXPECT_EQ(imin, 2);
        EXPECT_EQ(imax, 4);
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
