#include <gtest/gtest.h>
#include "io.h"
#include "hypar.h"
#include "mpivars_struct.h"
#include "arrayfunctions.h"
#include <fstream>
#include <string>
#include <vector>

// Test fixture for IOFunctions tests
class IOFunctionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        ndims_ = 1;
        nvars_ = 1;
        ghosts_ = 0;
        dim_local_[0] = 10;
        npoints_local_ = dim_local_[0];

        int total_points_local = npoints_local_;
        x_ = new double[total_points_local * ndims_];
        u_ = new double[total_points_local * nvars_];

        for (int i = 0; i < total_points_local; ++i) {
            x_[i] = static_cast<double>(i);
            u_[i] = static_cast<double>(i * 2.0);
        }

        index_ = new int[ndims_];
    }

    void TearDown() override {
        delete[] x_;
        delete[] u_;
        delete[] index_;
    }

    int ndims_, nvars_, ghosts_, npoints_local_;
    int dim_local_[1];
    double *x_, *u_;
    int *index_;
    std::string test_filename = "test_io.txt";
};

TEST_F(IOFunctionsTest, IOFunctionsComprehensive) {
    // Test WriteText_1D
    {
        WriteText(ndims_, nvars_, dim_local_, x_, u_, (char*)test_filename.c_str(), index_);

        std::ifstream file(test_filename);
        ASSERT_TRUE(file.is_open());

        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            int i_read;
            double x_read, u_read;
            std::sscanf(line.c_str(), "%d %lf %lf", &i_read, &x_read, &u_read);

            EXPECT_EQ(i_read, line_num);
            EXPECT_DOUBLE_EQ(x_read, x_[line_num]);
            EXPECT_DOUBLE_EQ(u_read, u_[line_num]);
            line_num++;
        }
        EXPECT_EQ(line_num, npoints_local_);

        file.close();
        std::remove(test_filename.c_str());
    }
}

// Main function
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
