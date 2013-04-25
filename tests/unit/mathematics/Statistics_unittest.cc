#include <shogun/lib/config.h>
#ifdef HAVE_EIGEN3

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
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
	
	index_t size=1000;

	// initialize the matrix
	SGSparseMatrix<float64_t> M(size, size);
	typedef SGSparseVectorEntry<float64_t> Entry;
	SGSparseVector<float64_t> *vec=SG_MALLOC(SGSparseVector<float64_t>, size);

	// for first row
	Entry *first=SG_MALLOC(Entry, size);
	first[0].feat_index=0;		// the digonal index for row #1
	first[0].entry=100;
	for( index_t i=1; i<size; ++i )
	{
		first[i].feat_index=i;	// fill the index for row #1
		first[i].entry=1;
	}
	vec[0].features=first;
	vec[0].num_feat_entries=size;
	M[0]=vec[0].get();

	// fill the rest of the rows
	Entry** rest=SG_MALLOC(Entry*, size-1);
	for( index_t i=0; i<size-1; ++i )
	{
		rest[i]=SG_MALLOC(Entry, 2);
		rest[i][0].feat_index=0;	// the first column
		rest[i][0].entry=1;
		rest[i][1].feat_index=i+1;	// the diagonal element
		rest[i][1].entry=100;
		vec[i+1].features=rest[i];
		vec[i+1].num_feat_entries=2;
		M[i+1]=vec[i+1].get();
	}

	// check if log_det is equal to log(det(M))
	EXPECT_NEAR(CStatistics::log_det(M), 4605.0649365774307, 1E-10);
	SG_FREE(vec);
	SG_FREE(rest);

}

// TEST 4 - Sampling from Multivariate Gaussian distribution with Dense 
// covariance matrix.
TEST(Statistics, sample_from_gaussian_dense1)
{

	int32_t N=1000;
	int32_t dim=100;

	// create a mean vector
	SGVector<float64_t> mean(dim);
	Map<VectorXd> mu(mean.vector, mean.vlen);
	mu=VectorXd::Constant(dim, 1, 0.0);

	// create a random covariance matrix
	SGMatrix<float64_t> cov(dim, dim);
	Map<MatrixXd> c(cov.matrix, cov.num_rows, cov.num_cols);
	c=MatrixXd::Random(dim, dim)*0.005+MatrixXd::Constant(dim, dim, 0.01);
	c=c*c.transpose()+MatrixXd::Identity(dim, dim)*0.01;

	SGMatrix<float64_t> samples=CStatistics::sample_from_gaussian(mean, cov, N);

	// calculate the sample mean and covariance
	SGVector<float64_t> s_mean=CStatistics::matrix_mean(samples);
	SGMatrix<float64_t> s_cov=CStatistics::covariance_matrix(samples);
	Map<VectorXd> s_mu(s_mean.vector, s_mean.vlen);
	Map<MatrixXd> s_c(s_cov.matrix, s_cov.num_rows, s_cov.num_cols);

	ASSERT_EQ(mu.rows(), s_mu.rows());
	ASSERT_EQ(c.rows(), s_c.rows());
	ASSERT_EQ(c.cols(), s_c.cols());
	EXPECT_NEAR((s_mu-mu).norm(), 0.0, 0.5);
	EXPECT_NEAR((s_c-c).norm(), 0.0, 1.0);

}

// TEST 5 - Sampling from Multivariate Gaussian distribution with Dense 
// covariance matrix. Using precision_matrix instead
TEST(Statistics, sample_from_gaussian_dense2)
{

	int32_t N=1000;
	int32_t dim=100;

	// create a mean vector
	SGVector<float64_t> mean(dim);
	Map<VectorXd> mu(mean.vector, mean.vlen);
	mu=VectorXd::Constant(dim, 1, 0.0);

	// create a random covariance matrix
	SGMatrix<float64_t> cov(dim, dim);
	Map<MatrixXd> c(cov.matrix, cov.num_rows, cov.num_cols);
	c=MatrixXd::Random(dim, dim)*0.5+MatrixXd::Constant(dim, dim, 1);
	c=c*c.transpose()+MatrixXd::Identity(dim, dim);

	SGMatrix<float64_t> samples=CStatistics::sample_from_gaussian(mean, cov, N, 
		true);

	// calculate the sample mean and covariance
	SGVector<float64_t> s_mean=CStatistics::matrix_mean(samples);
	SGMatrix<float64_t> s_cov=CStatistics::covariance_matrix(samples);
	Map<VectorXd> s_mu(s_mean.vector, s_mean.vlen);
	Map<MatrixXd> s_c(s_cov.matrix, s_cov.num_rows, s_cov.num_cols);

	ASSERT_EQ(mu.rows(), s_mu.rows());
	ASSERT_EQ(c.rows(), s_c.rows());
	ASSERT_EQ(c.cols(), s_c.cols());
	EXPECT_NEAR((s_mu-mu).norm(), 0.0, 0.5);
	EXPECT_NEAR((s_c-c.inverse()).norm(), 0.0, 5.0);

}

// TEST 6 - Sampling from Multivariate Gaussian distribution with Sparse
// covariance matrix. Using precision_matrix instead
TEST(Statistics, sample_from_gaussian_sparse1)
{

	int32_t N=1000;
	int32_t dim=100;

	// initialize the covariance matrix
	SGSparseMatrix<float64_t> cov(dim, dim);
	typedef SGSparseVectorEntry<float64_t> Entry;
	SGSparseVector<float64_t> *vec=SG_MALLOC(SGSparseVector<float64_t>, dim);

	// for first row
	Entry *first=SG_MALLOC(Entry, dim);
	first[0].feat_index=0;		// the digonal index for row #1
	first[0].entry=0.5;
	for( index_t i=1; i<dim; ++i )
	{
		first[i].feat_index=i;	// fill the index for row #1
		first[i].entry=0.05;
	}
	vec[0].features=first;
	vec[0].num_feat_entries=dim;
	cov[0]=vec[0].get();

	// fill the rest of the rows
	Entry** rest=SG_MALLOC(Entry*, dim-1);
	for( index_t i=0; i<dim-1; ++i )
	{
		rest[i]=SG_MALLOC(Entry, 2);
		rest[i][0].feat_index=0;	// the first column
		rest[i][0].entry=0.05;
		rest[i][1].feat_index=i+1;	// the diagonal element
		rest[i][1].entry=0.5;
		vec[i+1].features=rest[i];
		vec[i+1].num_feat_entries=2;
		cov[i+1]=vec[i+1].get();
	}

	// create a mean vector
	SGVector<float64_t> mean(dim);
	Map<VectorXd> mu(mean.vector, mean.vlen);
	mu=VectorXd::Constant(dim, 1, 5.0);

	SGMatrix<float64_t> samples=CStatistics::sample_from_gaussian(mean, cov, N);

	// calculate the sample mean and covariance
	SGVector<float64_t> s_mean=CStatistics::matrix_mean(samples);
	SGMatrix<float64_t> s_cov=CStatistics::covariance_matrix(samples);
	Map<VectorXd> s_mu(s_mean.vector, s_mean.vlen);
	Map<MatrixXd> s_c(s_cov.matrix, s_cov.num_rows, s_cov.num_cols);

	// create a similar dense cov matrix as of the original one
	// for calculating the norm of the difference
	MatrixXd d_cov=MatrixXd::Identity(dim, dim)*0.5;
	for( index_t i=1; i<dim; ++i )
	{
		d_cov(i,0)=0.05;
		d_cov(0,i)=0.05;
	}

	EXPECT_NEAR((s_mu-mu).norm(), 0.0, 0.5);
	EXPECT_NEAR((d_cov-s_c).norm(), 0.0, 2.5);

	SG_FREE(vec);
	SG_FREE(rest);

}
#endif // HAVE_EIGEN3
