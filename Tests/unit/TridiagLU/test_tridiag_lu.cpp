#include <gtest/gtest.h>
#include "tridiagLU.h"
#include <cmath>
#include <cstring>

// Test fixture for TridiagLU tests
class TridiagLUTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test simple tridiagonal system in serial mode
TEST_F(TridiagLUTest, SerialTridiagonalSolve) {
    // Test solving a simple diagonal system
    // System: 4x0 = 8, 4x1 = 12, 4x2 = 16, 4x3 = 20
    // Solution should be x = [2, 3, 4, 5]

    int n = 4;   // size of system
    int ns = 1;  // number of systems

    double a[4], b[4], c[4], x[4];

    // Initialize tridiagonal matrix (actually diagonal in this case)
    a[0] = 0.0;  a[1] = 0.0;  a[2] = 0.0;  a[3] = 0.0;
    b[0] = 4.0;  b[1] = 4.0;  b[2] = 4.0;  b[3] = 4.0;
    c[0] = 0.0;  c[1] = 0.0;  c[2] = 0.0;  c[3] = 0.0;

    // Right-hand side
    x[0] = 8.0;  x[1] = 12.0;  x[2] = 16.0;  x[3] = 20.0;

    // Initialize TridiagLU parameters
    TridiagLU params;
    std::memset(&params, 0, sizeof(TridiagLU));
    std::strcpy(params.reducedsolvetype, "gather-and-solve");

    // Solve in serial mode (NULL MPI communicator)
    int result = tridiagLU(a, b, c, x, n, ns, &params, nullptr);

    EXPECT_EQ(result, 0);

    // Check solution
    EXPECT_NEAR(x[0], 2.0, 1e-10);
    EXPECT_NEAR(x[1], 3.0, 1e-10);
    EXPECT_NEAR(x[2], 4.0, 1e-10);
    EXPECT_NEAR(x[3], 5.0, 1e-10);
}

// Test multiple tridiagonal systems solved together
TEST_F(TridiagLUTest, SerialMultipleSystems) {
    // Test solving 2 diagonal systems of size 3 each
    int n = 3;   // size of each system
    int ns = 2;  // number of systems

    double a[6], b[6], c[6], x[6];

    // Both systems are diagonal
    for (int i = 0; i < n; i++) {
        for (int sys = 0; sys < ns; sys++) {
            a[i*ns + sys] = 0.0;
            b[i*ns + sys] = 2.0;
            c[i*ns + sys] = 0.0;

            // Different RHS for each system
            x[i*ns + sys] = 2.0 * (i + 1) * (sys + 1);
        }
    }

    // Initialize TridiagLU parameters
    TridiagLU params;
    std::memset(&params, 0, sizeof(TridiagLU));
    std::strcpy(params.reducedsolvetype, "gather-and-solve");

    // Solve in serial mode
    int result = tridiagLU(a, b, c, x, n, ns, &params, nullptr);

    EXPECT_EQ(result, 0);

    // Expected solutions (RHS / diagonal):
    // System 0: [2/2, 4/2, 6/2] = [1, 2, 3]
    // System 1: [4/2, 8/2, 12/2] = [2, 4, 6]
    EXPECT_NEAR(x[0*ns + 0], 1.0, 1e-10);
    EXPECT_NEAR(x[1*ns + 0], 2.0, 1e-10);
    EXPECT_NEAR(x[2*ns + 0], 3.0, 1e-10);

    EXPECT_NEAR(x[0*ns + 1], 2.0, 1e-10);
    EXPECT_NEAR(x[1*ns + 1], 4.0, 1e-10);
    EXPECT_NEAR(x[2*ns + 1], 6.0, 1e-10);
}

// Note: Parallel tridiagonal solver tests with MPI are better suited
// for integration tests rather than unit tests

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
