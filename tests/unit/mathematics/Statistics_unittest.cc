#include <shogun/lib/config.h>

#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/Statistics.h>
#include <math.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

TEST(Statistics, fit_sigmoid)
{
	SGVector<float64_t> scores(10);

	for (index_t i=0; i<scores.vlen; ++i)
		scores[i]=i%2==0 ? 1 : -1;

	CStatistics::SigmoidParamters params=CStatistics::fit_sigmoid(scores);

	// compare against python implementation of john plat's algoorithm
	//	SG_SPRINT("a=%f, b=%f\n", params.a, params.b);

	EXPECT_NEAR(params.a, -1.791759, 1E-5);
	EXPECT_NEAR(params.b, 0.000000, 10E-5);
}

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
#ifdef HAVE_LAPACK
	SGMatrix<float64_t> s_cov=CStatistics::covariance_matrix(samples);
	Map<MatrixXd> s_c(s_cov.matrix, s_cov.num_rows, s_cov.num_cols);
#endif // HAVE_LAPACK
	Map<VectorXd> s_mu(s_mean.vector, s_mean.vlen);

#ifdef HAVE_LAPACK
	ASSERT_EQ(c.rows(), s_c.rows());
	ASSERT_EQ(c.cols(), s_c.cols());
	EXPECT_NEAR((s_c-c).norm(), 0.0, 1.0);
#endif // HAVE_LAPACK
	ASSERT_EQ(mu.rows(), s_mu.rows());
	EXPECT_NEAR((s_mu-mu).norm(), 0.0, 0.5);

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
#ifdef HAVE_LAPACK
	SGMatrix<float64_t> s_cov=CStatistics::covariance_matrix(samples);
	Map<MatrixXd> s_c(s_cov.matrix, s_cov.num_rows, s_cov.num_cols);
#endif // HAVE_LAPACK
	Map<VectorXd> s_mu(s_mean.vector, s_mean.vlen);

#ifdef HAVE_LAPACK
	ASSERT_EQ(c.rows(), s_c.rows());
	ASSERT_EQ(c.cols(), s_c.cols());
	EXPECT_NEAR((s_c-c.inverse()).norm(), 0.0, 5.0);
#endif // HAVE_LAPACK
	ASSERT_EQ(mu.rows(), s_mu.rows());
	EXPECT_NEAR((s_mu-mu).norm(), 0.0, 0.5);

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
#ifdef HAVE_LAPACK
	SGMatrix<float64_t> s_cov=CStatistics::covariance_matrix(samples);
	Map<MatrixXd> s_c(s_cov.matrix, s_cov.num_rows, s_cov.num_cols);
#endif // HAVE_LAPACK
	Map<VectorXd> s_mu(s_mean.vector, s_mean.vlen);

	// create a similar dense cov matrix as of the original one
	// for calculating the norm of the difference
	MatrixXd d_cov=MatrixXd::Identity(dim, dim)*0.5;
	for( index_t i=1; i<dim; ++i )
	{
		d_cov(i,0)=0.05;
		d_cov(0,i)=0.05;
	}

	EXPECT_NEAR((s_mu-mu).norm(), 0.0, 0.5);
#ifdef HAVE_LAPACK
	EXPECT_NEAR((d_cov-s_c).norm(), 0.0, 2.5);
#endif // HAVE_LAPACK

	SG_FREE(vec);
	SG_FREE(rest);

}

TEST(Statistics, dlgamma)
{
	// get digamma of positive x and compare result
	// with result from gsl implementation
	float64_t psi=CStatistics::dlgamma(0.3);
	EXPECT_NEAR(psi, -3.50252422220013, 1e-14);

	psi=CStatistics::dlgamma(1.2);
	EXPECT_NEAR(psi, -0.289039896592188, 1e-15);

	psi=CStatistics::dlgamma(3.7);
	EXPECT_NEAR(psi, 1.16715353936151, 1e-14);

	psi=CStatistics::dlgamma(14.9);
	EXPECT_NEAR(psi, 2.66742897621604, 1e-14);

	psi=CStatistics::dlgamma(1063.7);
	EXPECT_NEAR(psi, 6.96903854425933, 1e-14);

	// get digamma of negative x and compare result
	// with result from gsl implementation
	psi=CStatistics::dlgamma(-0.6);
	EXPECT_NEAR(psi, -0.894717877918449, 1e-15);

	psi=CStatistics::dlgamma(-1.4);
	EXPECT_NEAR(psi, 1.67366650039252, 1e-14);

	psi=CStatistics::dlgamma(-5.4);
	EXPECT_NEAR(psi, 2.79690872657593, 1e-14);

	psi=CStatistics::dlgamma(-15.3);
	EXPECT_NEAR(psi, 5.042677398790, 1e-12);

	psi=CStatistics::dlgamma(-122.7);
	EXPECT_NEAR(psi, 2.531311127723, 1e-12);
}

TEST(Statistics, lnormal_cdf)
{
	float64_t lphi=CStatistics::lnormal_cdf(-20);
	EXPECT_NEAR(lphi, -203.917155, 1e-6);

	lphi=CStatistics::lnormal_cdf(-3.4);
	EXPECT_NEAR(lphi, -7.995637, 1e-6);

	lphi=CStatistics::lnormal_cdf(-0.3);
	EXPECT_NEAR(lphi, -0.962102, 1e-6);

	lphi=CStatistics::lnormal_cdf(0.7);
	EXPECT_NEAR(lphi, -0.277023, 1e-6);

	lphi=CStatistics::lnormal_cdf(10.0);
	EXPECT_NEAR(lphi, 0.0, 1e-6);

	lphi=CStatistics::lnormal_cdf(20.0);
	EXPECT_NEAR(lphi, 0.0, 1e-6);

	const index_t dim=10;
	SGVector<float64_t> t(dim);
	t[0]=0.197197002817524946749;
	t[1]=0.0056939283141627128337;
	t[2]=11.17348207854067787537;
	t[3]=0.513878566283254256675;
	t[4]=4.34696415708135575073;
	t[5]=-8.69392831416271150147;
	t[6]=-17.38785662832542300293;
	t[7]=-34.42874909956949380785;
	t[8]=-67.46964157081356461276;
	t[9]=-136.67410392703391153191;

	SGVector<float64_t> res(dim);
	for(index_t i = 0; i < dim; i++)
		res[i]=CStatistics::lnormal_cdf(t[i]);

	float64_t rel_tolorance = 1e-2;
	float64_t abs_tolorance;

	abs_tolorance = CMath::get_abs_tolorance(-0.54789890348158110100, rel_tolorance);
	EXPECT_NEAR(res[0],  -0.54789890348158110100,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.68861439622252362813, rel_tolorance);
	EXPECT_NEAR(res[1],  -0.68861439622252362813,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(0.00000000000000000000, rel_tolorance);
	EXPECT_NEAR(res[2],  0.00000000000000000000,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.36192936264423153370, rel_tolorance);
	EXPECT_NEAR(res[3],  -0.36192936264423153370,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-0.00000690176209610232, rel_tolorance);
	EXPECT_NEAR(res[4],  -0.00000690176209610232,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-40.88657697941886226545, rel_tolorance);
	EXPECT_NEAR(res[5],  -40.88657697941886226545,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-154.94677031185918281153, rel_tolorance);
	EXPECT_NEAR(res[6],  -154.94677031185918281153,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-597.12805462746462126233, rel_tolorance);
	EXPECT_NEAR(res[7],  -597.12805462746462126233,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-2281.20710267821459638071, rel_tolorance);
	EXPECT_NEAR(res[8],  -2281.20710267821459638071,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolorance(-9345.74193347713662660681, rel_tolorance);
	EXPECT_NEAR(res[9],  -9345.74193347713662660681,  abs_tolorance);
	
}

#endif // HAVE_EIGEN3

TEST(Statistics, chi2_cdf)
{
	float64_t chi2c=CStatistics::chi2_cdf(1.0, 5.0);
	EXPECT_NEAR(chi2c, 0.03743423, 1e-7);

	chi2c=CStatistics::chi2_cdf(10.0, 5.0);
	EXPECT_NEAR(chi2c, 0.92476475, 1e-7);

	chi2c=CStatistics::chi2_cdf(1.0, 15.0);
	EXPECT_NEAR(chi2c, 0.00000025, 1e-7);
}

TEST(Statistics, fdistribution_cdf)
{
	float64_t fdcdf=CStatistics::fdistribution_cdf(0.5, 3.0, 5.0);
	EXPECT_NEAR(fdcdf,0.30154736, 1e-7);

	fdcdf=CStatistics::fdistribution_cdf(100, 3.0, 5.0);
	EXPECT_NEAR(fdcdf, 0.99993031, 1e-7);

	fdcdf=CStatistics::fdistribution_cdf(1.0, 30.0, 15.0);
	EXPECT_NEAR(fdcdf, 0.48005131, 1e-7);
}

// TEST 1
TEST(Statistics, log_det_general_test_1)
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
	EXPECT_NEAR(CStatistics::log_det_general(m), log(M.determinant()), 1E-10);

}

// TEST 2
TEST(Statistics, log_det_general_test_2)
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
	EXPECT_NEAR(CStatistics::log_det_general(K), 12.731839097176634, 1E-10);
}

TEST(Statistics,log_det_general_test_3)
{
	float64_t rel_tolorance = 1e-10;
	float64_t abs_tolorance, result;

	index_t size = 5;
	SGMatrix<float64_t> A(size, size);
	A(0,0) = 17.0;
	A(0,1) = 24.0;
	A(0,2) = 1.0;
	A(0,3) = 8.0;
	A(0,4) = 15.0;
	A(1,0) = 23.0;
	A(1,1) = 5.0;
	A(1,2) = 7.0;
	A(1,3) = 14.0;
	A(1,4) = 16.0;
	A(2,0) = 4.0;
	A(2,1) = 6.0;
	A(2,2) = 13.0;
	A(2,3) = 20.0;
	A(2,4) = 22.0;
	A(3,0) = 10.0;
	A(3,1) = 12.0;
	A(3,2) = 19.0;
	A(3,3) = 21.0;
	A(3,4) = 3.0;
	A(4,0) = 11.0;
	A(4,1) = 18.0;
	A(4,2) = 25.0;
	A(4,3) = 2.0;
	A(4,4) = 9.0;
	result = CStatistics::log_det_general(A);
	abs_tolorance = CMath::get_abs_tolorance(15.438851375567365, rel_tolorance);
	EXPECT_NEAR(result, 15.438851375567365, abs_tolorance);
}

TEST(Statistics,log_det_general_test_4)
{
	float64_t result;
	index_t size = 6;
	SGMatrix<float64_t> A(size, size);
	A(0,0) = 35.000000;
	A(0,1) = 1.000000;
	A(0,2) = 6.000000;
	A(0,3) = 26.000000;
	A(0,4) = 19.000000;
	A(0,5) = 24.000000;
	A(1,0) = 3.000000;
	A(1,1) = 32.000000;
	A(1,2) = 7.000000;
	A(1,3) = 21.000000;
	A(1,4) = 23.000000;
	A(1,5) = 25.000000;
	A(2,0) = 31.000000;
	A(2,1) = 9.000000;
	A(2,2) = 2.000000;
	A(2,3) = 22.000000;
	A(2,4) = 27.000000;
	A(2,5) = 20.000000;
	A(3,0) = 8.000000;
	A(3,1) = 28.000000;
	A(3,2) = 33.000000;
	A(3,3) = 17.000000;
	A(3,4) = 10.000000;
	A(3,5) = 15.000000;
	A(4,0) = 30.000000;
	A(4,1) = 5.000000;
	A(4,2) = 34.000000;
	A(4,3) = 12.000000;
	A(4,4) = 14.000000;
	A(4,5) = 16.000000;
	A(5,0) = 4.000000;
	A(5,1) = 36.000000;
	A(5,2) = 29.000000;
	A(5,3) = 13.000000;
	A(5,4) = 18.000000;
	A(5,5) = 11.000000;
	result = CStatistics::log_det_general(A);
	EXPECT_EQ(result, CMath::INFTY);
}

TEST(Statistics, vector_mean_test)
{
	CMath::init_random(17);
	SGVector<float64_t> a(10);
	a.random(-1024.0, 1024.0);
	floatmax_t sum_a=0;
	for(int i=0; i<a.vlen; i++)
		sum_a+=a[i];

	EXPECT_EQ(sum_a/a.vlen, CStatistics::mean(a));
}

TEST(Statistics, vector_mean_overflow_test)
{
	SGVector<float64_t> a(10);
	a.set_const(std::numeric_limits<float64_t>::max());
	EXPECT_EQ(std::numeric_limits<float64_t>::max(), CStatistics::mean(a));
}