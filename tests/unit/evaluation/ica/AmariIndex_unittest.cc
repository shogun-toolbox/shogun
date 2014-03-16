#include <shogun/lib/common.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3

#include <shogun/evaluation/ica/AmariIndex.h>

using namespace shogun;

TEST(AmariIndex, amari_zero)
{
	int d = 10;
	SGMatrix<float64_t> A = SGMatrix<float64_t>::create_identity_matrix(d,1);
	SGMatrix<float64_t> W = SGMatrix<float64_t>::create_identity_matrix(d,1);

	float64_t error = amari_index(A,W,false);

	EXPECT_NEAR(error, 0.0, 1e-5);
}

#endif //HAVE_EIGEN3
