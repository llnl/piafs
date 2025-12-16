/*! @file test_common_functions.cpp
    @brief Unit tests for common functions
    @author Unit Test Suite
*/

#include <gtest/gtest.h>
#include <common.h>
#include <cmath>
#include <cstring>

// Test fixture for common functions
class CommonFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}

    void expectArraysNear(const double* a, const double* b, int size, double tol = 1e-10) {
        for (int i = 0; i < size; i++) {
            EXPECT_NEAR(a[i], b[i], tol) << "Arrays differ at index " << i;
        }
    }
};

// ============================================================================
// Comprehensive test for all common functions
// ============================================================================

TEST_F(CommonFunctionsTest, CommonFunctionsComprehensive) {
    // Test GetStringFromInteger
    {
        char result[100];
        GetStringFromInteger(41, result, 5);
        EXPECT_STREQ(result, "00041");

        GetStringFromInteger(41, result, 2);
        EXPECT_STREQ(result, "41");

        GetStringFromInteger(41, result, 1);
        EXPECT_STREQ(result, "1");

        GetStringFromInteger(0, result, 3);
        EXPECT_STREQ(result, "000");

        GetStringFromInteger(123456, result, 8);
        EXPECT_STREQ(result, "00123456");

        GetStringFromInteger(999, result, 3);
        EXPECT_STREQ(result, "999");

        GetStringFromInteger(12345, result, 3);
        EXPECT_STREQ(result, "345");

        GetStringFromInteger(7, result, 4);
        EXPECT_STREQ(result, "0007");
    }

    // Test takeLog
    {
        double array[5] = {1.0, 2.0, 10.0, 100.0, std::exp(1.0)};
        double expected[5];
        for (int i = 0; i < 5; i++) expected[i] = std::log(array[i]);
        takeLog(array, 5);
        expectArraysNear(array, expected, 5);
    }
    {
        double array[1] = {std::exp(2.0)};
        takeLog(array, 1);
        EXPECT_NEAR(array[0], 2.0, 1e-10);
    }
    {
        double array[3] = {0.1, 0.01, 0.001};
        double expected[3] = {std::log(0.1), std::log(0.01), std::log(0.001)};
        takeLog(array, 3);
        expectArraysNear(array, expected, 3);
    }
    {
        double array[3] = {1.0, -2.0, 3.0};
        takeLog(array, 3);
        EXPECT_NEAR(array[0], 0.0, 1e-10);
        EXPECT_NEAR(array[2], std::log(3.0), 1e-10);
        EXPECT_TRUE(std::isnan(array[1]));
    }
    {
        double array[3] = {1.0, 0.0, 2.0};
        takeLog(array, 3);
        EXPECT_NEAR(array[0], 0.0, 1e-10);
        EXPECT_NEAR(array[2], std::log(2.0), 1e-10);
        EXPECT_TRUE(std::isnan(array[1]));
    }
    {
        double array[3] = {1e10, 1e20, 1e100};
        double expected[3] = {std::log(1e10), std::log(1e20), std::log(1e100)};
        takeLog(array, 3);
        expectArraysNear(array, expected, 3);
    }

    // Test takeExp
    {
        double array[4] = {0.0, 1.0, 2.0, 3.0};
        double expected[4] = {1.0, std::exp(1.0), std::exp(2.0), std::exp(3.0)};
        takeExp(array, 4);
        expectArraysNear(array, expected, 4);
    }
    {
        double array[3] = {-1.0, -2.0, -3.0};
        double expected[3] = {std::exp(-1.0), std::exp(-2.0), std::exp(-3.0)};
        takeExp(array, 3);
        expectArraysNear(array, expected, 3);
    }
    {
        double array[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
        double expected[5];
        for (int i = 0; i < 5; i++) expected[i] = std::exp(array[i]);
        takeExp(array, 5);
        expectArraysNear(array, expected, 5);
    }
    {
        double array[1] = {0.0};
        takeExp(array, 1);
        EXPECT_NEAR(array[0], 1.0, 1e-10);
    }
    {
        double array[2] = {-10.0, -100.0};
        takeExp(array, 2);
        EXPECT_NEAR(array[0], std::exp(-10.0), 1e-10);
        EXPECT_NEAR(array[1], std::exp(-100.0), 1e-120);
    }
    {
        double array[3] = {0.01, 0.001, 0.0001};
        double expected[3] = {std::exp(0.01), std::exp(0.001), std::exp(0.0001)};
        takeExp(array, 3);
        expectArraysNear(array, expected, 3);
    }

    // Test Log/Exp inverse relationship
    {
        double original[5] = {1.0, 2.0, 5.0, 10.0, 100.0};
        double array[5];
        for (int i = 0; i < 5; i++) array[i] = original[i];
        takeLog(array, 5);
        takeExp(array, 5);
        expectArraysNear(array, original, 5);
    }
    {
        double original[5] = {-2.0, -1.0, 0.0, 1.0, 2.0};
        double array[5];
        for (int i = 0; i < 5; i++) array[i] = original[i];
        takeExp(array, 5);
        takeLog(array, 5);
        expectArraysNear(array, original, 5);
    }

    // Test checkNanInf
    {
        double array[5] = {1.0, -2.0, 3.5, 0.0, -100.0};
        EXPECT_NO_THROW(checkNanInf(array, 5, "test_location"));
    }
    {
        double array[3] = {1.0, NAN, 3.0};
        EXPECT_EXIT(checkNanInf(array, 3, "test_nan"),
                    ::testing::ExitedWithCode(1),
                    "NaN detected");
    }
    {
        double array[3] = {1.0, INFINITY, 3.0};
        EXPECT_EXIT(checkNanInf(array, 3, "test_inf"),
                    ::testing::ExitedWithCode(1),
                    "Inf detected");
    }
    {
        double array[3] = {1.0, -INFINITY, 3.0};
        EXPECT_EXIT(checkNanInf(array, 3, "test_neg_inf"),
                    ::testing::ExitedWithCode(1),
                    "Inf detected");
    }
    {
        double array[1];
        EXPECT_NO_THROW(checkNanInf(array, 0, "test_empty"));
    }
    {
        double array[1000];
        for (int i = 0; i < 1000; i++) array[i] = i * 0.5;
        EXPECT_NO_THROW(checkNanInf(array, 1000, "test_large"));
    }
    {
        double array[4] = {1e308, -1e308, 1e-308, -1e-308};
        EXPECT_NO_THROW(checkNanInf(array, 4, "test_extreme"));
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
