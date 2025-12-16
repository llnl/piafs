#include <gtest/gtest.h>
#include "limiters.h"
#include <cmath>

// Test fixture for LimiterFunctions tests
class LimiterFunctionsTest : public ::testing::Test {};

TEST_F(LimiterFunctionsTest, LimiterFunctionsComprehensive) {
    // Test LimiterMinMod
    {
        EXPECT_DOUBLE_EQ(LimiterMinMod(1.5), 1.0);
        EXPECT_DOUBLE_EQ(LimiterMinMod(0.5), 0.5);
        EXPECT_DOUBLE_EQ(LimiterMinMod(0.0), 0.0);
        EXPECT_DOUBLE_EQ(LimiterMinMod(1.0), 1.0);
        EXPECT_DOUBLE_EQ(LimiterMinMod(-0.5), 0.0);
    }

    // Test LimiterVanLeer
    {
        EXPECT_DOUBLE_EQ(LimiterVanLeer(0.5), (0.5 + std::abs(0.5)) / (1.0 + std::abs(0.5)));
        EXPECT_DOUBLE_EQ(LimiterVanLeer(1.5), (1.5 + std::abs(1.5)) / (1.0 + std::abs(1.5)));
        EXPECT_DOUBLE_EQ(LimiterVanLeer(0.0), 0.0);
        EXPECT_DOUBLE_EQ(LimiterVanLeer(-0.5), 0.0);
    }

    // Test LimiterSuperBee
    {
        EXPECT_DOUBLE_EQ(LimiterSuperBee(-0.5), 0.0);
        EXPECT_DOUBLE_EQ(LimiterSuperBee(0.25), 2.0 * 0.25);
        EXPECT_DOUBLE_EQ(LimiterSuperBee(0.5), 1.0);
        EXPECT_DOUBLE_EQ(LimiterSuperBee(0.75), 1.0);
        EXPECT_DOUBLE_EQ(LimiterSuperBee(1.5), 1.5);
        EXPECT_DOUBLE_EQ(LimiterSuperBee(2.5), 2.0);
    }

    // Test LimiterGeneralizedMinMod
    {
        EXPECT_DOUBLE_EQ(LimiterGeneralizedMinMod(-0.5), 0.0);
        EXPECT_DOUBLE_EQ(LimiterGeneralizedMinMod(0.5), 0.5);
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
