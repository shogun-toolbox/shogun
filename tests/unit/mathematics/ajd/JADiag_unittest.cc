#include <shogun/lib/common.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGNDArray.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/JADiag.h>

#include <shogun/evaluation/ica/PermutationMatrix.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;
typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;

using namespace shogun;

TEST(CJADiag, diagonalize)
{
	// Generating diagonal matrices
	index_t * C_dims = SG_MALLOC(index_t, 3);
	C_dims[0] = 10;
	C_dims[1] = 10;
	C_dims[2] = 30;
	SGNDArray< float64_t > C(C_dims, 3);

	CMath::init_random(17);

	for (int i = 0; i < C_dims[2]; i++)
	{
		Eigen::Map<EMatrix> tmp(C.get_matrix(i),C_dims[0], C_dims[1]);
		tmp.setIdentity();

		for (int j = 0; j < C_dims[0]; j++)
			tmp(j,j) *= CMath::abs(CMath::random(1,5));

	}

	// Mixing and demixing matrices
	EMatrix B(C_dims[0], C_dims[1]);
	B.setRandom();
	EMatrix A = B.inverse();

	for (int i = 0; i < C_dims[2]; i++)
	{
		Eigen::Map<EMatrix> Ci(C.get_matrix(i),C_dims[0], C_dims[1]);
		Ci = A * Ci * A.transpose();
	}

	/** Diagonalize **/
	SGMatrix<float64_t> V = CJADiag::diagonalize(C);

	// Test output size
	EXPECT_EQ(V.num_rows, C_dims[0]);
	EXPECT_EQ(V.num_cols, C_dims[1]);

	// Close to a permutation matrix (with random scales)
	Eigen::Map<EMatrix> EV(V.matrix,C_dims[0], C_dims[1]);
	SGMatrix<float64_t> P(C_dims[0],C_dims[1]);
	Eigen::Map<EMatrix> EP(P.matrix,C_dims[0], C_dims[1]);
	EP = EV * A;

	// Test if output is correct
	bool isperm = is_permutation_matrix(P);
	EXPECT_EQ(isperm,true);
}
