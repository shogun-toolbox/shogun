#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/eigen3.h>
#include <math.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

// TEST 1
TEST(Statistics, log_det_test_1)
{
	// create a small test matrix, symmetric positive definite
	index_t size = 3;
	SGMatrix<float64_t> m(size, size);
	
	// initialize the matrix
	m(0, 0) =   4; m(0, 1) =  12; m(0, 2) = -16;
	m(1, 0) =  12; m(1, 1) =  37; m(1, 2) = -43;
	m(2, 0) = -16; m(2, 1) = -43; m(2, 2) =  98;
	
	/* the cholesky decomposition gives m = L.L', where
	 * L = [(2, 0, 0), (6, 1, 0), (-8, 5, 3)].
	 * 2 * (log(2) + log(1) + log(3)) = 3.58351893846
	 */
	Map<MatrixXd> M(m.matrix, m.num_rows, m.num_cols);
	EXPECT_NEAR(CStatistics::log_det(m), log(M.determinant()), 1E-10);

}

// TEST 2
TEST(Statistics, log_det_test_2)
{
	// create a fixed symmetric positive definite matrix
	index_t size = 100;
	VectorXd A = VectorXd::LinSpaced(size, 1, size);
	MatrixXd M = A * A.transpose() + MatrixXd::Identity(size, size);

	// copy the matrix to a SGMatrix to pass it to log_det
	SGMatrix<float64_t> K(size,size);
	for( int32_t j = 0; j < size; ++j ) {
		for( int32_t i = 0; i < size; ++i ) {
			K(i,j) = M(i,j);
		}
	}
	
	// check if log_det is equal to log(det(M))
	EXPECT_NEAR(CStatistics::log_det(K), 12.731839097176634, 1E-10);

}

// TEST 3 - Sparse matrix
TEST(Statistics, log_det_test_3)
{
	// create a sparse test matrix, symmetric positive definite
	// whose the diagonal contains all 100's
	// the rest of first row and first column contains all 1's
	
	index_t size = 1000;

	// initialize the matrix
	SGSparseMatrix<float64_t> M(size, size);

	SGSparseVector<float64_t> *vec;

	// for first row
	SGSparseVectorEntry<float64_t> *entries = new SGSparseVectorEntry<float64_t>[size];
	entries[0].feat_index = 0;		// the digonal index for row #1
	entries[0].entry = 100;
	for( index_t i = 1; i < size; ++i )
	{
		entries[i].feat_index = i;	// fill the index for row #1
		entries[i].entry = 1;
	}
	vec = new SGSparseVector<float64_t>(entries, size);
	M[0] = vec->get();

	// fill the rest of the rows
	for( index_t i = 1; i < size; ++i )
	{
		entries = new SGSparseVectorEntry<float64_t>[2];
		entries[0].feat_index = 0;	// the first column
		entries[0].entry = 1;
		entries[1].feat_index = i;	// the diagonal element
		entries[1].entry = 100;
		vec = new SGSparseVector<float64_t>(entries, 2);
		M[i] = vec->get();
	}

	// check if log_det is equal to log(det(M))
	EXPECT_NEAR(CStatistics::log_det(M), 4605.0649365774307, 1E-10);

}
#endif // HAVE_EIGEN3
