#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/eigen3.h>
#include <math.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace Eigen;

TEST(Statistics,log_det)
{
	/* TEST 1 */
	/* create a small test matrix, positive definite, symmetric */
	index_t size = 3;
	SGMatrix<float64_t> m(size, size);
	
	/* initialize the matrix */
	m(0, 0) =   4; m(0, 1) =  12; m(0, 2) = -16;
	m(1, 0) =  12; m(1, 1) =  37; m(1, 2) = -43;
	m(2, 0) = -16; m(2, 1) = -43; m(2, 2) =  98;
	
	/* the cholesky decomposition gives m = L.L', where
	 * L = [(2, 0, 0), (6, 1, 0), (-8, 5, 3)].
	 * 2 * (log(2) + log(1) + log(3)) = 3.58352
	 */
	EXPECT_FLOAT_EQ(CStatistics::log_det(m), 3.58352);

	// TEST 2
	// create random positive definite symmetric matrix
	size = 100;
	MatrixXd A(size, size);	// random matrix
        MatrixXd M(size, size);

	// to make all the values between [0, 1] (is it really needed?)
	A = MatrixXd::Random(size,size) * 0.5 + MatrixXd::Constant(size,size,0.5);
	M = A * A.transpose();	// positive definite matrix

	// copy the matrix to a SGMatrix to pass it to log_det
	SGMatrix<float64_t> K(size,size);
	for( int32_t j = 0; j < size; ++j ) {
		for( int32_t i = 0; i < size; ++i ) {
			K(i,j) = M(i,j);
		}
	}
	
	// check if log_det is equal to log(det(M))
	EXPECT_FLOAT_EQ(CStatistics::log_det(K), log(M.determinant()));

}

#endif
