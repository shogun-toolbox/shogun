#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGNDArray.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/FFDiag.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;
typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;

using namespace shogun;


/*
Function that tests if a Matrix is a permutation matrix
*/
namespace {
bool is_permutation_matrix(EMatrix &mat)
{
	// scale	
	mat *= 100;

	// round
	for (int i = 0; i < mat.rows(); i++)
	{
		for (int j = 0; j < mat.cols(); j++)
		{
			if (CMath::abs(CMath::round(mat(i,j))) >= 1.0)
				mat(i,j) = 1.0;
			else
				mat(i,j) = 0.0;
		}
	}	

	// Debug print
	//std::cout << mat << std::endl;

	// check only a single 1 per row
	for (int i = 0; i < mat.rows(); i++)
	{
		int num_ones = 0;
		for (int j = 0; j < mat.cols(); j++)
		{
			if (mat(i,j) >= 1.0)
				num_ones++;
		}

		if (num_ones != 1)
			return false;
	}
	
	// check only a single 1 per col
	for (int j = 0; j < mat.cols(); j++)
	{
		int num_ones = 0;
		for (int i = 0; i < mat.rows(); i++)
		{
			if (mat(i,j) >= 1.0)
				num_ones++; 
		}
		
		if (num_ones != 1)
			return false;
	}
	
	return true;
}
}

TEST(CFFDiag, diagonalize)
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
		{
			tmp(j,j) *= CMath::abs(CMath::random(1,5)); 
		}
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
	SGMatrix<float64_t> V = CFFDiag::diagonalize(C);

	// Test output size
	EXPECT_EQ(V.num_rows, C_dims[0]);
	EXPECT_EQ(V.num_cols, C_dims[1]);

	// Close to a permutation matrix (with random scales)
	Eigen::Map<EMatrix> EV(V.matrix,C_dims[0], C_dims[1]);
	EMatrix perm = EV * A;

	// Test if output is correct
	bool isperm = is_permutation_matrix(perm);
	EXPECT_EQ(isperm,true);
}

#endif //HAVE_EIGEN3
