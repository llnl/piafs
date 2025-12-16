#include <gtest/gtest.h>
#include "boundaryconditions.h"
#include "mpivars_struct.h"
#include "arrayfunctions.h"

// Test fixture for BoundaryConditions tests
class BoundaryConditionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ndims = 1;
        nvars = 1;
        ghosts = 3;
        size[0] = 10;
        phi_size = size[0] + 2 * ghosts;

        phi = new double[phi_size];
        for (int i = 0; i < phi_size; ++i) {
            phi[i] = static_cast<double>(i);
        }

        boundary.is = new int[ndims];
        boundary.ie = new int[ndims];

        mpi.rank = 0;
        mpi.nproc = 1;
        mpi.iproc = new int[ndims];
        for (int i = 0; i < ndims; ++i) {
            mpi.iproc[i] = 1;
        }
    }

    void TearDown() override {
        delete[] phi;
        delete[] boundary.is;
        delete[] boundary.ie;
        delete[] mpi.iproc;
    }

    int ndims, nvars, ghosts;
    int size[1];
    int phi_size;
    double* phi;
    MPIVariables mpi;
    DomainBoundary boundary;
};

TEST_F(BoundaryConditionsTest, BoundaryConditionsComprehensive) {
    // Test BCPeriodicU - Left Face
    {
        for (int i = 0; i < ghosts; ++i) {
            phi[i] = -1.0;
        }
        for (int i = ghosts; i < size[0] + ghosts; ++i) {
            phi[i] = static_cast<double>(i);
        }
        for (int i = size[0] + ghosts; i < phi_size; ++i) {
            phi[i] = static_cast<double>(i);
        }

        boundary.dim = 0;
        boundary.face = 1;
        boundary.on_this_proc = 1;
        boundary.is[0] = -ghosts;
        boundary.ie[0] = 0;

        BCPeriodicU(&boundary, &mpi, ndims, nvars, size, ghosts, phi, 0.0);

        for (int i = 0; i < ghosts; ++i) {
            int dest_idx = i;
            int source_idx = size[0] + i;
            EXPECT_DOUBLE_EQ(phi[dest_idx], phi[source_idx])
                << "Left ghost point " << dest_idx << " should equal interior point " << source_idx;
        }
    }

    // Test BCPeriodicU - Right Face
    {
        for (int i = 0; i < ghosts; ++i) {
            phi[i] = static_cast<double>(i);
        }
        for (int i = ghosts; i < size[0] + ghosts; ++i) {
            phi[i] = static_cast<double>(i);
        }
        for (int i = size[0] + ghosts; i < phi_size; ++i) {
            phi[i] = -1.0;
        }

        boundary.dim = 0;
        boundary.face = -1;
        boundary.on_this_proc = 1;
        boundary.is[0] = size[0];
        boundary.ie[0] = size[0] + ghosts;

        BCPeriodicU(&boundary, &mpi, ndims, nvars, size, ghosts, phi, 0.0);

        for (int i = 0; i < ghosts; ++i) {
            int dest_idx = size[0] + ghosts + i;
            int source_idx = ghosts + i;
            EXPECT_DOUBLE_EQ(phi[dest_idx], phi[source_idx])
                << "Right ghost point " << dest_idx << " should equal interior point " << source_idx;
        }
    }

    // Test BCExtrapolateU - Left Face
    {
        for (int i = 0; i < ghosts; ++i) {
            phi[i] = -1.0;
        }
        for (int i = ghosts; i < phi_size; ++i) {
            phi[i] = static_cast<double>(i);
        }

        boundary.dim = 0;
        boundary.face = 1;
        boundary.on_this_proc = 1;
        boundary.is[0] = -ghosts;
        boundary.ie[0] = 0;

        BCExtrapolateU(&boundary, &mpi, ndims, nvars, size, ghosts, phi, 0.0);

        // BCExtrapolateU copies interior points in reverse order to ghost points
        // indexb[0]=0 -> indexi[0]=ghosts-1-0=2 -> phi[0] = phi[2+ghosts=5]
        // indexb[0]=1 -> indexi[0]=ghosts-1-1=1 -> phi[1] = phi[1+ghosts=4]
        // indexb[0]=2 -> indexi[0]=ghosts-1-2=0 -> phi[2] = phi[0+ghosts=3]
        for (int i = 0; i < ghosts; ++i) {
            int source_idx = (ghosts - 1 - i) + ghosts;
            EXPECT_DOUBLE_EQ(phi[i], phi[source_idx]);
        }
    }

    // Test BCExtrapolateU - Right Face
    {
        for (int i = 0; i < size[0] + ghosts; ++i) {
            phi[i] = static_cast<double>(i);
        }
        for (int i = size[0] + ghosts; i < phi_size; ++i) {
            phi[i] = -1.0;
        }

        boundary.dim = 0;
        boundary.face = -1;
        boundary.on_this_proc = 1;
        boundary.is[0] = size[0];
        boundary.ie[0] = size[0] + ghosts;

        BCExtrapolateU(&boundary, &mpi, ndims, nvars, size, ghosts, phi, 0.0);

        // BCExtrapolateU copies interior points in reverse order to ghost points
        // indexb[0]=0 -> indexi[0]=size[0]-0-1=9 -> phi[13] = phi[9+ghosts=12]
        // indexb[0]=1 -> indexi[0]=size[0]-1-1=8 -> phi[14] = phi[8+ghosts=11]
        // indexb[0]=2 -> indexi[0]=size[0]-2-1=7 -> phi[15] = phi[7+ghosts=10]
        for (int i = 0; i < ghosts; ++i) {
            int source_idx = (size[0] - i - 1) + ghosts;
            EXPECT_DOUBLE_EQ(phi[size[0] + ghosts + i], phi[source_idx]);
        }
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
