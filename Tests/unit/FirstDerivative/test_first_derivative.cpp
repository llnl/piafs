#include <gtest/gtest.h>
#include "firstderivative.h"
#include "hypar.h"
#include "mpivars_struct.h"

// Test fixture for FirstDerivative tests
class FirstDerivativeTest : public ::testing::Test {
protected:
    void SetUp() override {
        solver.ndims = 1;
        solver.nvars = 1;
        solver.ghosts = 3;

        solver.dim_local = new int[solver.ndims];
        solver.dim_local[0] = 10;
        int size = solver.dim_local[0] + 2 * solver.ghosts;

        f = new double[size];
        for (int i = 0; i < size; ++i) {
            f[i] = 2.0 * static_cast<double>(i);
        }

        Df = new double[size];

        mpi.rank = 0;
        mpi.nproc = 1;
        mpi.iproc = new int[solver.ndims];
        for (int i = 0; i < solver.ndims; ++i) {
            mpi.iproc[i] = 1;
        }
    }

    void TearDown() override {
        delete[] f;
        delete[] Df;
        delete[] solver.dim_local;
        delete[] mpi.iproc;
    }

    HyPar solver;
    MPIVariables mpi;
    double* f;
    double* Df;
};

TEST_F(FirstDerivativeTest, FirstDerivativeComprehensive) {
    int dir = 0;

    // Test FirstOrder Forward
    {
        int bias = 1;
        FirstDerivativeFirstOrder(Df, f, dir, bias, &solver, &mpi);
        for (int i = 0; i < (solver.dim_local[0] + 2 * solver.ghosts) - 1; ++i) {
            EXPECT_DOUBLE_EQ(Df[i], 2.0);
        }
    }

    // Test FirstOrder Backward
    {
        int bias = -1;
        FirstDerivativeFirstOrder(Df, f, dir, bias, &solver, &mpi);
        for (int i = 1; i < (solver.dim_local[0] + 2 * solver.ghosts); ++i) {
            EXPECT_DOUBLE_EQ(Df[i], 2.0);
        }
    }

    // Test SecondOrderCentral
    {
        int bias = 0;
        FirstDerivativeSecondOrderCentral(Df, f, dir, bias, &solver, &mpi);
        for (int i = 1; i < (solver.dim_local[0] + 2 * solver.ghosts) - 1; ++i) {
            EXPECT_DOUBLE_EQ(Df[i], 2.0);
        }
    }

    // Test FourthOrderCentral
    {
        int bias = 0;
        FirstDerivativeFourthOrderCentral(Df, f, dir, bias, &solver, &mpi);
        // For linear function, fourth order should also give exact derivative
        for (int i = 2; i < (solver.dim_local[0] + 2 * solver.ghosts) - 2; ++i) {
            EXPECT_NEAR(Df[i], 2.0, 1e-10);
        }
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
