#include <shogun/base/init.h>
#include <shogun/lib/common.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3

#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGNDArray.h>

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/ajd/JADiag.h>

using namespace Eigen;

typedef Matrix< float64_t, Dynamic, Dynamic, ColMajor > EMatrix;
typedef Matrix< float64_t, Dynamic, 1, ColMajor > EVector;

using namespace shogun;

/*
Function that tests if a SGMatrix is a permutation matrix
*/
bool is_permutation_matrix(SGMatrix<float64_t> &mat)
{
	// scale
	for(int i = 0; i < mat.num_rows; i++)
		for(int j = 0; j < mat.num_cols; j++)
			mat(i,j) *= 10;
			
	// check only a single 1 per row
	for(int i = 0; i < mat.num_rows; i++)
	{
		int num_ones = 0;
		for(int j = 0; j < mat.num_cols; j++)
		{
			if(CMath::abs(CMath::round(mat(i,j))) >= 1.0)
				num_ones++;
		}

		if(num_ones != 1)
			return false;
	}
	
	// check only a single 1 per col
	for(int j = 0; j < mat.num_cols; j++)
	{
		int num_ones = 0;
		for(int i = 0; i < mat.num_rows; i++)
		{
			if(CMath::abs(CMath::round(mat(i,j))) >= 1.0)
				num_ones++; 
		}
		
		if(num_ones != 1)
			return false;
	}
	
	return true;
}

TEST(CJADiag, diagonalize)
{
	// Generating diagonal matrices
	index_t * C_dims = SG_MALLOC(index_t, 3);
	C_dims[0] = 10;
	C_dims[1] = 10;
	C_dims[2] = 30;
	SGNDArray< float64_t > C(C_dims, 3);
	
	for(int i = 0; i < C_dims[2]; i++)
	{
		Eigen::Map<EMatrix> tmp(C.get_matrix(i),C_dims[0], C_dims[1]);
		tmp.setIdentity();
		
		for(int j = 0; j < C_dims[0]; j++)
		{
			// change to chi square when it is easy to do so			
			tmp(j,j) += CMath::random(); 
		}
	}
	
	// Mixing and demixing matrices
	EMatrix B(C_dims[0], C_dims[1]); B.setRandom();
	EMatrix A = B.inverse();
	
	for(int i = 0; i < C_dims[2]; i++)
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
	EMatrix Eperm = EV * A;

	// Test if permutation matrix
	SGMatrix<float64_t> perm(C_dims[0], C_dims[1]);
	memcpy(Eperm.data(), perm.matrix, C_dims[0]*C_dims[1]*sizeof(float64_t));

	bool isperm = is_permutation_matrix(perm);

	EXPECT_EQ(isperm,true);
}

#endif //HAVE_EIGEN3
