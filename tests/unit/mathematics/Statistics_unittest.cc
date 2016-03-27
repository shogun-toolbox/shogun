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

	float64_t rel_tolerance = 1e-2;
	float64_t abs_tolerance;

	abs_tolerance = CMath::get_abs_tolerance(-0.54789890348158110100, rel_tolerance);
	EXPECT_NEAR(res[0],  -0.54789890348158110100,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.68861439622252362813, rel_tolerance);
	EXPECT_NEAR(res[1],  -0.68861439622252362813,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.00000000000000000000, rel_tolerance);
	EXPECT_NEAR(res[2],  0.00000000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.36192936264423153370, rel_tolerance);
	EXPECT_NEAR(res[3],  -0.36192936264423153370,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.00000690176209610232, rel_tolerance);
	EXPECT_NEAR(res[4],  -0.00000690176209610232,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-40.88657697941886226545, rel_tolerance);
	EXPECT_NEAR(res[5],  -40.88657697941886226545,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-154.94677031185918281153, rel_tolerance);
	EXPECT_NEAR(res[6],  -154.94677031185918281153,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-597.12805462746462126233, rel_tolerance);
	EXPECT_NEAR(res[7],  -597.12805462746462126233,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-2281.20710267821459638071, rel_tolerance);
	EXPECT_NEAR(res[8],  -2281.20710267821459638071,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-9345.74193347713662660681, rel_tolerance);
	EXPECT_NEAR(res[9],  -9345.74193347713662660681,  abs_tolerance);

}

TEST(Statistics, normal_cdf)
{
	// assert with value calculated via Octave normcdf() method
	float64_t phi=CStatistics::normal_cdf(-2);
	EXPECT_NEAR(phi, 0.0227501319481792190, 1e-15);

	phi=CStatistics::normal_cdf(-3, 4);
	EXPECT_NEAR(phi, 0.2266273523768682074, 1e-15);

	phi=CStatistics::normal_cdf(-0.3);
	EXPECT_NEAR(phi, 0.3820885778110473807, 1e-15);

	phi=CStatistics::normal_cdf(0.7);
	EXPECT_NEAR(phi, 0.7580363477769269664, 1e-15);

	phi=CStatistics::normal_cdf(1);
	EXPECT_NEAR(phi, 0.8413447460685429258, 1e-15);

	phi=CStatistics::normal_cdf(2);
	EXPECT_NEAR(phi, 0.9772498680518207914, 1e-15);
}

TEST(Statistics, inverse_normal_cdf)
{
	// assert with value calculated via Mathematica
	float64_t result;

	result=CStatistics::inverse_normal_cdf(0.0000001);
	EXPECT_NEAR(result, -5.199337582187471, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.00001);
	EXPECT_NEAR(result, -4.264890793922602, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.001);
	EXPECT_NEAR(result, -3.090232306167813, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.05);
	EXPECT_NEAR(result, -1.6448536269514729, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.15);
	EXPECT_NEAR(result, -1.0364333894937896, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.25);
	EXPECT_NEAR(result, -0.6744897501960817, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.35);
	EXPECT_NEAR(result, -0.38532046640756773, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.45);
	EXPECT_NEAR(result, -0.12566134685507402, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.55);
	EXPECT_NEAR(result, 0.12566134685507402, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.65);
	EXPECT_NEAR(result, 0.6744897501960817, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.75);
	EXPECT_NEAR(result, 1.0364333894937896, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.85);
	EXPECT_NEAR(result, 0.38532046640756773, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.95);
	EXPECT_NEAR(result, 0.6744897501960817, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.99);
	EXPECT_NEAR(result, 1.0364333894937896, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.999);
	EXPECT_NEAR(result, 1.6448536269514729, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.99999);
	EXPECT_NEAR(result, 3.090232306167813, 1e-15);
	
	result=CStatistics::inverse_normal_cdf(0.9999999);
	EXPECT_NEAR(result, 4.264890793922602, 1e-15);
}

TEST(Statistics, inverse_normal_cdf_with_mean_std_dev)
{
	EXPECT_NEAR(0, 1, 1e-15);
}

TEST(Statistics, gamma_incomplete_lower)
{
	// tests against scipy.special.gammainc
	float64_t result;
	
	result=CStatistics::gamma_incomplete_lower(1, 1);
	EXPECT_NEAR(result, 0.63212055882855778, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(2, 1);
	EXPECT_NEAR(result, 0.26424111765711528, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(1, 2);
	EXPECT_NEAR(result, 0.8646647167633873, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(0.0000001, 1);
	EXPECT_NEAR(result, 0.99999997806160246, 1e-14);
	
	result=CStatistics::gamma_incomplete_lower(0.001, 1);
	EXPECT_NEAR(result, 0.9997803916424145, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(5, 1);
	EXPECT_NEAR(result, 0.0036598468273437131, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(10, 1);
	EXPECT_NEAR(result, 1.1142547833872071e-07, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(1, 0.0000001);
	EXPECT_NEAR(result, 9.9999994999999991e-08, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(1, 0.001);
	EXPECT_NEAR(result, 0.00099950016662500823, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(1, 5);
	EXPECT_NEAR(result, 0.99326205300091452, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(1, 10);
	EXPECT_NEAR(result, 0.99995460007023751, 1e-14);
	
	result=CStatistics::gamma_incomplete_lower(1, 20);
	EXPECT_NEAR(result, 0.99999999793884642, 1e-14);
	
	// special cases in the implementation
	result=CStatistics::gamma_incomplete_lower(1, 0);
	EXPECT_NEAR(result, 0, 1e-15);
	
	result=CStatistics::gamma_incomplete_lower(1, 2);
	EXPECT_NEAR(result, 0.8646647167633873, 1e-15);
}

TEST(Statistics, gamma_incomplete_upper)
{
	// tests against scipy.special.gammaincc
	float64_t result;
	
	result=CStatistics::gamma_incomplete_upper(1, 1);
	EXPECT_NEAR(result, 0.36787944117144233, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(2, 1);
	EXPECT_NEAR(result, 0.73575888234288467, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(1, 2);
	EXPECT_NEAR(result, 0.1353352832366127, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(0.0000001, 1);
	EXPECT_NEAR(result, 2.193839568430253e-08, 1e-14);
	
	result=CStatistics::gamma_incomplete_upper(0.001, 1);
	EXPECT_NEAR(result, 0.00021960835758555317, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(5, 1);
	EXPECT_NEAR(result, 0.99634015317265634, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(10, 1);
	EXPECT_NEAR(result, 0.99999988857452171, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(1, 0.0000001);
	EXPECT_NEAR(result, 0.99999990000000505, 1e-9);
	
	result=CStatistics::gamma_incomplete_upper(1, 0.001);
	EXPECT_NEAR(result, 0.99900049983337502, 1e-12);
	
	result=CStatistics::gamma_incomplete_upper(1, 5);
	EXPECT_NEAR(result, 0.0067379469990854679, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(1, 10);
	EXPECT_NEAR(result, 4.5399929762484861e-05, 1e-14);
	
	result=CStatistics::gamma_incomplete_upper(1, 20);
	EXPECT_NEAR(result, 2.0611536224385566e-09, 1e-14);
	
	// special cases in the implementation
	result=CStatistics::gamma_incomplete_upper(1, 0);
	EXPECT_NEAR(result, 1, 1e-15);
	
	result=CStatistics::gamma_incomplete_upper(1, 0.5);
	EXPECT_NEAR(result, 0.60653065971263342, 1e-15);
}

TEST(Statistics, gamma_pdf)
{
	// tests against scipy.stats.gamma.pdf
	// note that scipy.stats.gamma.pdf(2, a=2, scale=1./2) corresonds to
	// CStatistics:.gamma_pdf(2,2,2)
	float64_t result;
	
	// three basic cases to get order of parameters
	result=CStatistics::gamma_pdf(2, 1, 1);
	EXPECT_NEAR(result, 0.1353352832366127, 1e-15);
	result=CStatistics::gamma_pdf(2, 2, 1);
	EXPECT_NEAR(result, 0.2706705664732254, 1e-15);
	result=CStatistics::gamma_pdf(2, 1, 2);
	EXPECT_NEAR(result, 0.036631277777468357, 1e-15);
	
	// testing x for a=b=1
	result=CStatistics::gamma_pdf(0.0000001, 1, 1);
	EXPECT_NEAR(result, 0.99999990000000505, 1e-15);
	
	result=CStatistics::gamma_pdf(0.00001, 1, 1);
	EXPECT_NEAR(result, 0.99999000004999983, 1e-15);
	
	result=CStatistics::gamma_pdf(0.001, 1, 1);
	EXPECT_NEAR(result, 0.99900049983337502, 1e-15);
	
	result=CStatistics::gamma_pdf(0.05, 1, 1);
	EXPECT_NEAR(result, 0.95122942450071402, 1e-15);
	
	result=CStatistics::gamma_pdf(0.1, 1, 1);
	EXPECT_NEAR(result, 0.90483741803595952, 1e-15);
	
	result=CStatistics::gamma_pdf(0.3, 1, 1);
	EXPECT_NEAR(result, 0.74081822068171788, 1e-15);
	
	result=CStatistics::gamma_pdf(0.5, 1, 1);
	EXPECT_NEAR(result, 0.60653065971263342, 1e-15);
	
	result=CStatistics::gamma_pdf(0.7, 1, 1);
	EXPECT_NEAR(result, 0.49658530379140953, 1e-15);
	
	result=CStatistics::gamma_pdf(1., 1, 1);
	EXPECT_NEAR(result, 0.36787944117144233, 1e-15);
	
	result=CStatistics::gamma_pdf(10., 1, 1);
	EXPECT_NEAR(result, 4.5399929762484854e-05, 1e-15);
	
	result=CStatistics::gamma_pdf(100., 1, 1);
	EXPECT_NEAR(result, 3.7200759760208361e-44, 1e-15);
}

TEST(Statistics, gamma_cdf)
{
	// tests against scipy.stats.gamma.cdf
	// note that scipy.stats.gamma.cdf(2, a=2, scale=1./2) corresonds to
	// CStatistics:.gamma_cdf(2,2,2)
	float64_t result;
	
	// only three basic cases to get order of parameters
	// incomplete gamma is based on is tested thoroughly already
	result=CStatistics::gamma_cdf(2, 1, 1);
	EXPECT_NEAR(result, 0.8646647167633873, 1e-15);
	result=CStatistics::gamma_cdf(2, 2, 1);
	EXPECT_NEAR(result, 0.59399415029016167, 1e-15);
	result=CStatistics::gamma_cdf(2, 1, 2);
	EXPECT_NEAR(result, 0.98168436111126578, 1e-15);
}

TEST(Statistics, chi2_pdf)
{
	EXPECT_NEAR(0, 1, 1e-15);
}

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
	float64_t rel_tolerance = 1e-10;
	float64_t abs_tolerance, result;

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
	abs_tolerance = CMath::get_abs_tolerance(15.438851375567365, rel_tolerance);
	EXPECT_NEAR(result, 15.438851375567365, abs_tolerance);
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
