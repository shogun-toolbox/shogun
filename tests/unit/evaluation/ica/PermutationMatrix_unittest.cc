#include <shogun/lib/common.h>
#include <gtest/gtest.h>

#include <shogun/evaluation/ica/PermutationMatrix.h>

using namespace shogun;

TEST(PermutationMatrix, is_perm)
{
	int d = 10;
	SGMatrix<float64_t> A = SGMatrix<float64_t>::create_identity_matrix(d,1);

	bool isperm = is_permutation_matrix(A);
	EXPECT_TRUE(isperm);
}

TEST(PermutationMatrix, is_not_perm)
{
	int d = 10;
	SGMatrix<float64_t> A = SGMatrix<float64_t>::create_identity_matrix(d,1);
	A(1,2) = 1;
	A(2,3) = 1;

	bool isperm = is_permutation_matrix(A);
	EXPECT_FALSE(isperm);
}

