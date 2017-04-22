/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Wu Lin
 * Written (W) 2011-2016 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * Most cdf routines are wrappers for CDFLIB, part of the public domain NETLIB.
 * https://people.sc.fsu.edu/~jburkardt/f_src/cdflib/cdflib.html
 */

#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseMatrix.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/external/cdflib.hpp>
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

using namespace shogun;

float64_t CStatistics::variance(SGVector<float64_t> values)
{
	REQUIRE(values.vlen>1, "Number of observations (%d) needs to be at least 1.\n",
			values.vlen);

	float64_t mean=CStatistics::mean(values);

	float64_t sum_squared_diff=0;
	for (index_t i=0; i<values.vlen; ++i)
		sum_squared_diff+=CMath::pow(values.vector[i]-mean, 2);

	return sum_squared_diff/(values.vlen-1);
}

SGVector<float64_t> CStatistics::matrix_mean(SGMatrix<float64_t> values,
		bool col_wise)
{
	ASSERT(values.num_rows>0)
	ASSERT(values.num_cols>0)
	ASSERT(values.matrix)

	SGVector<float64_t> result;

	if (col_wise)
	{
		result=SGVector<float64_t>(values.num_cols);
		for (index_t j=0; j<values.num_cols; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_rows; ++i)
				result[j]+=values(i,j);

			result[j]/=values.num_rows;
		}
	}
	else
	{
		result=SGVector<float64_t>(values.num_rows);
		for (index_t j=0; j<values.num_rows; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_cols; ++i)
				result[j]+=values(j,i);

			result[j]/=values.num_cols;
		}
	}

	return result;
}

SGVector<float64_t> CStatistics::matrix_variance(SGMatrix<float64_t> values,
		bool col_wise)
{
	ASSERT(values.num_rows>0)
	ASSERT(values.num_cols>0)
	ASSERT(values.matrix)

	/* first compute mean */
	SGVector<float64_t> mean=CStatistics::matrix_mean(values, col_wise);

	SGVector<float64_t> result;

	if (col_wise)
	{
		result=SGVector<float64_t>(values.num_cols);
		for (index_t j=0; j<values.num_cols; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_rows; ++i)
				result[j]+=CMath::pow(values(i,j)-mean[j], 2);

			result[j]/=(values.num_rows-1);
		}
	}
	else
	{
		result=SGVector<float64_t>(values.num_rows);
		for (index_t j=0; j<values.num_rows; ++j)
		{
			result[j]=0;
			for (index_t i=0; i<values.num_cols; ++i)
				result[j]+=CMath::pow(values(j,i)-mean[j], 2);

			result[j]/=(values.num_cols-1);
		}
	}

	return result;
}

float64_t CStatistics::std_deviation(SGVector<float64_t> values)
{
	return CMath::sqrt(variance(values));
}

SGVector<float64_t> CStatistics::matrix_std_deviation(
		SGMatrix<float64_t> values, bool col_wise)
{
	SGVector<float64_t> var=CStatistics::matrix_variance(values, col_wise);
	for (index_t i=0; i<var.vlen; ++i)
		var[i]=CMath::sqrt(var[i]);

	return var;
}

SGMatrix<float64_t> CStatistics::covariance_matrix(
		SGMatrix<float64_t> observations, bool in_place)
{
	int32_t D = observations.num_rows;
	int32_t N = observations.num_cols;
	SG_SDEBUG("%d observations in %d dimensions\n", N, D)

	REQUIRE(N>1, "Number of observations (%d) must be at least 2.\n", N);
	REQUIRE(D>0, "Number of dimensions (%d) must be at least 1.\n", D);

	SGMatrix<float64_t> centered=in_place ? observations :
					SGMatrix<float64_t>(D, N);

	/* center observations, potentially in-place */
	if (!in_place)
	{
		int64_t num_elements = N*D;
		sg_memcpy(centered.matrix, observations.matrix,
				sizeof(float64_t)*num_elements);
	}
	SG_SDEBUG("Centering observations\n");
	Map<MatrixXd> eigen_centered(centered.matrix, D, N);
	eigen_centered.colwise() -= eigen_centered.rowwise().mean();

	/* compute and store 1/(N-1) * X * X.T */
	SG_SDEBUG("Computing squared differences\n");
	SGMatrix<float64_t> cov(D, D);
	Map<MatrixXd> eigen_cov(cov.matrix, D, D);
	eigen_cov = (eigen_centered * eigen_centered.adjoint()) / double(N - 1);

	return cov;
}

SGVector<float64_t> CStatistics::fishers_exact_test_for_multiple_2x3_tables(
		SGMatrix<float64_t> tables)
{
	SGMatrix<float64_t> table(NULL, 2, 3, false);
	int32_t len=tables.num_cols/3;

	SGVector<float64_t> v(len);
	for (int32_t i=0; i<len; i++)
	{
		table.matrix=&tables.matrix[2*3*i];
		v.vector[i]=fishers_exact_test_for_2x3_table(table);
	}
	return v;
}

float64_t CStatistics::fishers_exact_test_for_2x3_table(
		SGMatrix<float64_t> table)
{
	ASSERT(table.num_rows==2)
	ASSERT(table.num_cols==3)

	int32_t m_len=3+2;
	float64_t* m=SG_MALLOC(float64_t, 3+2);
	m[0]=table.matrix[0]+table.matrix[2]+table.matrix[4];
	m[1]=table.matrix[1]+table.matrix[3]+table.matrix[5];
	m[2]=table.matrix[0]+table.matrix[1];
	m[3]=table.matrix[2]+table.matrix[3];
	m[4]=table.matrix[4]+table.matrix[5];

	float64_t n=SGVector<float64_t>::sum(m, m_len)/2.0;
	int32_t x_len=2*3*CMath::sq(CMath::max(m, m_len));
	float64_t* x=SG_MALLOC(float64_t, x_len);
	SGVector<float64_t>::fill_vector(x, x_len, 0.0);

	float64_t log_nom=0.0;
	for (int32_t i=0; i<3+2; i++)
		log_nom+=lgamma(m[i]+1);
	log_nom-=lgamma(n+1.0);

	float64_t log_denomf=0;
	floatmax_t log_denom=0;

	for (int32_t i=0; i<3*2; i++)
	{
		log_denom+=lgammal((floatmax_t)table.matrix[i]+1);
		log_denomf+=lgammal((floatmax_t)table.matrix[i]+1);
	}

	floatmax_t prob_table_log=log_nom-log_denom;

	int32_t dim1=CMath::min(m[0], m[2]);

	//traverse all possible tables with given m
	int32_t counter=0;
	for (int32_t k=0; k<=dim1; k++)
	{
		for (int32_t l=CMath::max(0.0, m[0]-m[4]-k);
				l<=CMath::min(m[0]-k, m[3]); l++)
		{
			x[0+0*2+counter*2*3]=k;
			x[0+1*2+counter*2*3]=l;
			x[0+2*2+counter*2*3]=m[0]-x[0+0*2+counter*2*3]-x[0+1*2+counter*2*3];
			x[1+0*2+counter*2*3]=m[2]-x[0+0*2+counter*2*3];
			x[1+1*2+counter*2*3]=m[3]-x[0+1*2+counter*2*3];
			x[1+2*2+counter*2*3]=m[4]-x[0+2*2+counter*2*3];

			counter++;
		}
	}

//#define DEBUG_FISHER_TABLE
#ifdef DEBUG_FISHER_TABLE
	SG_SPRINT("counter=%d\n", counter)
	SG_SPRINT("dim1=%d\n", dim1)
	SG_SPRINT("l=%g...%g\n", CMath::max(0.0,m[0]-m[4]-0), CMath::min(m[0]-0, m[3]))
	SG_SPRINT("n=%g\n", n)
	SG_SPRINT("prob_table_log=%.18Lg\n", prob_table_log)
	SG_SPRINT("log_denomf=%.18g\n", log_denomf)
	SG_SPRINT("log_denom=%.18Lg\n", log_denom)
	SG_SPRINT("log_nom=%.18g\n", log_nom)
	display_vector(m, m_len, "marginals");
	display_vector(x, 2*3*counter, "x");
#endif // DEBUG_FISHER_TABLE

	floatmax_t* log_denom_vec=SG_MALLOC(floatmax_t, counter);
	SGVector<floatmax_t>::fill_vector(log_denom_vec, counter, (floatmax_t)0.0);

	for (int32_t k=0; k<counter; k++)
	{
		for (int32_t j=0; j<3; j++)
		{
			for (int32_t i=0; i<2; i++)
				log_denom_vec[k]+=lgammal(x[i+j*2+k*2*3]+1.0);
		}
	}

	for (int32_t i=0; i<counter; i++)
		log_denom_vec[i]=log_nom-log_denom_vec[i];

#ifdef DEBUG_FISHER_TABLE
	display_vector(log_denom_vec, counter, "log_denom_vec");
#endif // DEBUG_FISHER_TABLE

	float64_t nonrand_p=-CMath::INFTY;
	for (int32_t i=0; i<counter; i++)
	{
		if (log_denom_vec[i]<=prob_table_log)
			nonrand_p=CMath::logarithmic_sum(nonrand_p, log_denom_vec[i]);
	}

#ifdef DEBUG_FISHER_TABLE
	SG_SPRINT("nonrand_p=%.18g\n", nonrand_p)
	SG_SPRINT("exp_nonrand_p=%.18g\n", CMath::exp(nonrand_p))
#endif // DEBUG_FISHER_TABLE
	nonrand_p=CMath::exp(nonrand_p);

	SG_FREE(log_denom_vec);
	SG_FREE(x);
	SG_FREE(m);

	return nonrand_p;
}

float64_t CStatistics::mutual_info(float64_t* p1, float64_t* p2, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		for (int32_t j=0; j<len; j++)
			e+=exp(p2[j*len+i])*(p2[j*len+i]-p1[i]-p1[j]);

	return (float64_t)e;
}

float64_t CStatistics::relative_entropy(float64_t* p, float64_t* q, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e+=exp(p[i])*(p[i]-q[i]);

	return (float64_t)e;
}

float64_t CStatistics::entropy(float64_t* p, int32_t len)
{
	double e=0;

	for (int32_t i=0; i<len; i++)
		e-=exp(p[i])*p[i];

	return (float64_t)e;
}

SGVector<int32_t> CStatistics::sample_indices(int32_t sample_size, int32_t N)
{
	REQUIRE(sample_size<N,
			"sample size should be less than number of indices\n");
	int32_t* idxs=SG_MALLOC(int32_t,N);
	int32_t i, rnd;
	int32_t* permuted_idxs=SG_MALLOC(int32_t,sample_size);

	// reservoir sampling
	for (i=0; i<N; i++)
		idxs[i]=i;
	for (i=0; i<sample_size; i++)
		permuted_idxs[i]=idxs[i];
	for (i=sample_size; i<N; i++)
	{
		rnd=CMath::random(1, i);
		if (rnd<sample_size)
			permuted_idxs[rnd]=idxs[i];
	}
	SG_FREE(idxs);

	SGVector<int32_t> result=SGVector<int32_t>(permuted_idxs, sample_size);
	CMath::qsort(result);
	return result;
}

float64_t CStatistics::inverse_normal_cdf(float64_t p, float64_t mean,
		float64_t std_deviation)
{
	REQUIRE(p>=0, "p (%f); must be greater or equal to 0.\n", p);
	REQUIRE(p<=1, "p (%f); must be greater or equal to 1.\n", p);
	REQUIRE(std_deviation>0, "Standard deviation (%f); must be positive\n",
			std_deviation);

	// invserse normal cdf case, see cdflib.cpp for details
	int which=2;
	float64_t output_x;
	float64_t q=1-p;
	float64_t output_bound;
	int output_status;

	cdfnor(&which, &p, &q, &output_x, &mean, &std_deviation, &output_status, &output_bound);

	if (output_status!=0)
		SG_SERROR("Error %d while calling cdflib::cdfnor\n", output_status);

	return output_x;

	//void cdfnor ( int *which, double *p, double *q, double *x, double *mean,
	// double *sd, int *status, double *bound )
}

float64_t CStatistics::chi2_cdf(float64_t x, float64_t k)
{
	REQUIRE(x>=0, "x (%f) has to be greater or equal to 0.\n", x);
	REQUIRE(k>0, "Degrees of freedom (%f) has to be positive.\n", k);

	// chi2 cdf case, see cdflib.cpp for details
	int which=1;
	float64_t df=k;
	float64_t output_q;
	float64_t output_p;
	float64_t output_bound;
	int output_status;

	cdfchi(&which, &output_p, &output_q, &x, &df, &output_status, &output_bound);

	if (output_status!=0)
		SG_SERROR("Error %d while calling cdflib::cdfchi\n", output_status);

	return output_p;
}

float64_t CStatistics::gamma_cdf(float64_t x, float64_t a, float64_t b)
{
	REQUIRE(x>=0, "x (%f) has to be greater or equal to 0.\n", x);
	REQUIRE(a>=0, "a (%f) has to be greater or equal to 0.\n", a);
	REQUIRE(b>=0, "b (%f) has to be greater or equal to 0.\n", b);

	// inverse gamma cdf case, see cdflib.cpp for details
	float64_t shape=a;
	float64_t scale=b;
	int which=1;
	float64_t output_p;
	float64_t output_q;
	float64_t output_bound;
	int output_error_code;

	cdfgam(&which, &output_p, &output_q, &x, &shape, &scale, &output_error_code, &output_bound);

	if (output_error_code!=0)
		SG_SERROR("Error %d while calling cdflib::cdfgam\n", output_error_code);

	return output_p;
}

float64_t CStatistics::lnormal_cdf(float64_t x)
{
	/* Loosely based on logphi.m in
	 * Gaussian Process Machine Learning Toolbox file logphi.m
	 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
	 * Under FreeBSD license
	 */

	const float64_t sqrt_of_2=1.41421356237309514547;
	const float64_t log_of_2=0.69314718055994528623;
	const float64_t sqrt_of_pi=1.77245385090551588192;

	const index_t c_len=14;
	static float64_t c_array[c_len]=
	{
		0.00048204,
		-0.00142906,
		0.0013200243174,
		0.0009461589032,
		-0.0045563339802,
		0.00556964649138,
		0.00125993961762116,
		-0.01621575378835404,
		0.02629651521057465,
		-0.001829764677455021,
		2.0*(1.0-CMath::PI/3.0),
		(4.0-CMath::PI)/3.0,
		1.0,
		1.0
	};

	if (x*x<ERFC_CASE1)
	{
		float64_t f = 0.0;
		float64_t lp0 = -x/(sqrt_of_2*sqrt_of_pi);
		for (index_t i=0; i<c_len; i++)
			f=lp0*(c_array[i]+f);
		return -2.0*f-log_of_2;
	}
	else if (x<ERFC_CASE2)
		return CMath::log(erfc8_weighted_sum(x))-log_of_2-x*x*0.5;

	//id3 = ~id2 & ~id1; lp(id3) = log(erfc(-z(id3)/sqrt(2))/2);
	return CMath::log(normal_cdf(x));
}

float64_t CStatistics::erfc8_weighted_sum(float64_t x)
{
	/* This is based on index 5725 in Hart et al */

	const float64_t sqrt_of_2=1.41421356237309514547;

	static float64_t P[]=
	{
		0.5641895835477550741253201704,
		1.275366644729965952479585264,
		5.019049726784267463450058,
		6.1602098531096305440906,
		7.409740605964741794425,
		2.97886562639399288862
	};

	static float64_t Q[]=
	{
		1.0,
		2.260528520767326969591866945,
		9.396034016235054150430579648,
		12.0489519278551290360340491,
		17.08144074746600431571095,
		9.608965327192787870698,
		3.3690752069827527677
	};

	float64_t num=0.0, den=0.0;

	num = P[0];
	for (index_t i=1; i<6; i++)
	{
		num=-x*num/sqrt_of_2+P[i];
	}

	den = Q[0];
	for (index_t i=1; i<7; i++)
	{
		den=-x*den/sqrt_of_2+Q[i];
	}

	return num/den;
}

float64_t CStatistics::normal_cdf(float64_t x, float64_t std_dev)
{
	return 0.5*(erfc(-x*M_SQRT1_2/std_dev));
}

float64_t CStatistics::gamma_inverse_cdf(float64_t p, float64_t a,
		float64_t b)
{
	REQUIRE(p>=0, "p (%f) has to be greater or equal to 0.\n", p);
	REQUIRE(a>=0, "a (%f) has to be greater or equal to 0.\n", a);
	REQUIRE(b>=0, "b (%f) has to be greater or equal to 0.\n", b);

	float64_t shape=a;
	float64_t scale=b;
	float64_t q = 1-p;
	int which=2;
	float64_t output_x=0;
	float64_t output_bound;
	int output_error_code=0;

	// inverse gamma cdf case, see cdflib.cpp for details
	cdfgam(&which, &p, &q, &output_x, &shape, &scale, &output_error_code, &output_bound);

	if (output_error_code!=0)
		SG_SERROR("Error %d while calling cdflib::beta_inc\n", output_error_code);

	return output_x;
}

float64_t CStatistics::fdistribution_cdf(float64_t x, float64_t d1, float64_t d2)
{
	REQUIRE(x>=0, "x (%f) has to be greater or equal to 0.\n", x);
	REQUIRE(d1>0, "d1 (%f) has to be positive.\n", d1);
	REQUIRE(d2>0, "d2 (%f) has to be positive.\n", d2);

	// fcdf case, see cdflib.cpp for details
	int which=1;
	float64_t output_p;
	float64_t output_q;
	float64_t output_bound;
	int output_error_code;

	cdff(&which, &output_p, &output_q, &x, &d1, &d2, &output_error_code, &output_bound);

	if (output_error_code!=0)
		SG_SERROR("Error %d while calling cdflib::cdff\n", output_error_code);

	return output_p;
}

float64_t CStatistics::dlgamma(float64_t x)
{
	float64_t result=0.0;

	if (x<0.0)
	{
		// use reflection formula
		x=1.0-x;
		result=CMath::PI/CMath::tan(CMath::PI*x);
	}

	// make x>7 for approximation
	// (use reccurent formula: psi(x+1) = psi(x) + 1/x)
	while (x<=7.0)
	{
		result-=1.0/x;
		x++;
	}

	// perform approximation
	x-=0.5;
	result+=log(x);

	float64_t coeff[10]={
		0.04166666666666666667,
		-0.00729166666666666667,
		0.00384424603174603175,
		-0.00413411458333333333,
		0.00756096117424242424,
		-0.02108249687595390720,
		0.08332316080729166666,
		-0.44324627670587277880,
		3.05393103044765369366,
		-26.45616165999210241989};

	float64_t power=1.0;
	float64_t ix2=1.0/CMath::sq(x);

	// perform approximation
	for (index_t i=0; i<10; i++)
	{
		power*=ix2;
		result+=coeff[i]*power;
	}

	return result;
}

float64_t CStatistics::log_det_general(const SGMatrix<float64_t> A)
{
	Map<MatrixXd> eigen_A(A.matrix, A.num_rows, A.num_cols);
	REQUIRE(eigen_A.rows()==eigen_A.cols(),
		"Input matrix should be a sqaure matrix row(%d) col(%d)\n",
		eigen_A.rows(), eigen_A.cols());

	PartialPivLU<MatrixXd> lu(eigen_A);
	VectorXd tmp(eigen_A.rows());

	for (index_t idx=0; idx<tmp.rows(); idx++)
		tmp[idx]=idx+1;

	VectorXd p=lu.permutationP()*tmp;
	int32_t detP=1;

	for (index_t idx=0; idx<p.rows(); idx++)
	{
		if (p[idx]!=idx+1)
		{
			detP*=-1;
			index_t j=idx+1;
			while(j<p.rows())
			{
				if (p[j]==idx+1)
					break;
				j++;
			}
			p[j]=p[idx];
		}
	}

	VectorXd u=lu.matrixLU().diagonal();
	int32_t check_u=1;

	for (index_t idx=0; idx<u.rows(); idx++)
	{
		if (u[idx]<0)
			check_u*=-1;
		else if (u[idx]==0)
		{
			check_u=0;
			break;
		}
	}

	float64_t result=CMath::INFTY;

	if (check_u==detP)
		result=u.array().abs().log().sum();

	return result;
}

float64_t CStatistics::log_det(SGMatrix<float64_t> m)
{
	/* map the matrix to eigen3 to perform cholesky */
	Map<MatrixXd> M(m.matrix, m.num_rows, m.num_cols);

	/* computing the cholesky decomposition */
	LLT<MatrixXd> llt;
	llt.compute(M);

	/* the lower triangular matrix */
	MatrixXd l = llt.matrixL();

	/* calculate the log-determinant */
	VectorXd diag = l.diagonal();
	float64_t retval = 0.0;
	for( int32_t i = 0; i < diag.rows(); ++i ) {
		retval += log(diag(i));
	}
	retval *= 2;

	return retval;
}

float64_t CStatistics::log_det(const SGSparseMatrix<float64_t> m)
{
	typedef SparseMatrix<float64_t> MatrixType;
	const MatrixType &M=EigenSparseUtil<float64_t>::toEigenSparse(m);

	SimplicialLLT<MatrixType> llt;

	// factorize using cholesky with amd permutation
	llt.compute(M);
	MatrixType L=llt.matrixL();

	// calculate the log-determinant
	float64_t retval=0.0;
	for( index_t i=0; i<M.rows(); ++i )
		retval+=log(L.coeff(i,i));
	retval*=2;

	return retval;
}

SGMatrix<float64_t> CStatistics::sample_from_gaussian(SGVector<float64_t> mean,
	SGMatrix<float64_t> cov, int32_t N, bool precision_matrix)
{
	REQUIRE(cov.num_rows>0, "Number of covariance rows must be positive!\n");
	REQUIRE(cov.num_cols>0,"Number of covariance cols must be positive!\n");
	REQUIRE(cov.matrix, "Covariance is not initialized!\n");
	REQUIRE(cov.num_rows==cov.num_cols, "Covariance should be square matrix!\n");
	REQUIRE(mean.vlen==cov.num_rows, "Mean and covariance dimension mismatch!\n");

	int32_t dim=mean.vlen;
	Map<VectorXd> mu(mean.vector, mean.vlen);
	Map<MatrixXd> c(cov.matrix, cov.num_rows, cov.num_cols);

	// generate samples, z,  from N(0, I), DxN
	SGMatrix<float64_t> S(dim, N);
	for( int32_t j=0; j<N; ++j )
		for( int32_t i=0; i<dim; ++i )
			S(i,j)=CMath::randn_double();

	// the cholesky factorization c=L*U
	MatrixXd U=c.llt().matrixU();

	// generate samples, x, from N(mean, cov) or N(mean, cov^-1)
	// return samples of dimension NxD
	if( precision_matrix )
	{
		// here we have U*x=z, to solve this, we use cholesky again
		Map<MatrixXd> s(S.matrix, S.num_rows, S.num_cols);
		LDLT<MatrixXd> ldlt;
		ldlt.compute(U);
		s=ldlt.solve(s);
	}

	SGMatrix<float64_t>::transpose_matrix(S.matrix, S.num_rows, S.num_cols);

	if( !precision_matrix )
	{
		// here we need to find x=L*z, so, x'=z'*L' i.e. x'=z'*U
		Map<MatrixXd> s(S.matrix, S.num_rows, S.num_cols);
		s=s*U;
	}

	// add the mean
	Map<MatrixXd> x(S.matrix, S.num_rows, S.num_cols);
	for( int32_t i=0; i<N; ++i )
		x.row(i)+=mu;

	return S;
}

SGMatrix<float64_t> CStatistics::sample_from_gaussian(SGVector<float64_t> mean,
 SGSparseMatrix<float64_t> cov, int32_t N, bool precision_matrix)
{
	REQUIRE(cov.num_vectors>0,
		"CStatistics::sample_from_gaussian(): \
		Number of covariance rows must be positive!\n");
	REQUIRE(cov.num_features>0,
		"CStatistics::sample_from_gaussian(): \
		Number of covariance cols must be positive!\n");
	REQUIRE(cov.sparse_matrix,
		"CStatistics::sample_from_gaussian(): \
		Covariance is not initialized!\n");
	REQUIRE(cov.num_vectors==cov.num_features,
		"CStatistics::sample_from_gaussian(): \
		Covariance should be square matrix!\n");
	REQUIRE(mean.vlen==cov.num_vectors,
		"CStatistics::sample_from_gaussian(): \
		Mean and covariance dimension mismatch!\n");

	int32_t dim=mean.vlen;
	Map<VectorXd> mu(mean.vector, mean.vlen);

	typedef SparseMatrix<float64_t> MatrixType;
	const MatrixType &c=EigenSparseUtil<float64_t>::toEigenSparse(cov);

	SimplicialLLT<MatrixType> llt;

	// generate samples, z,  from N(0, I), DxN
	SGMatrix<float64_t> S(dim, N);
	for( int32_t j=0; j<N; ++j )
		for( int32_t i=0; i<dim; ++i )
			S(i,j)=CMath::randn_double();

	Map<MatrixXd> s(S.matrix, S.num_rows, S.num_cols);

	// the cholesky factorization P*c*P^-1 = LP*UP, with LP=P*L, UP=U*P^-1
	llt.compute(c);
	MatrixType LP=llt.matrixL();
	MatrixType UP=llt.matrixU();

	// generate samples, x, from N(mean, cov) or N(mean, cov^-1)
	// return samples of dimension NxD
	if( precision_matrix )
	{
		// here we have UP*xP=z, to solve this, we use cholesky again
		SimplicialLLT<MatrixType> lltUP;
		lltUP.compute(UP);
		s=lltUP.solve(s);
	}
	else
	{
		// here we need to find xP=LP*z
		s=LP*s;
	}

	// permute the samples back with x=P^-1*xP
	s=llt.permutationPinv()*s;

	SGMatrix<float64_t>::transpose_matrix(S.matrix, S.num_rows, S.num_cols);
	// add the mean
	Map<MatrixXd> x(S.matrix, S.num_rows, S.num_cols);
	for( int32_t i=0; i<N; ++i )
		x.row(i)+=mu;

	return S;
}


CStatistics::SigmoidParamters CStatistics::fit_sigmoid(SGVector<float64_t> scores)
{
	SG_SDEBUG("entering CStatistics::fit_sigmoid()\n")

	REQUIRE(scores.vector, "CStatistics::fit_sigmoid() requires "
			"scores vector!\n");

	/* count prior0 and prior1 if needed */
	int32_t prior0=0;
	int32_t prior1=0;
	SG_SDEBUG("counting number of positive and negative labels\n")
	{
		for (index_t i=0; i<scores.vlen; ++i)
		{
			if (scores[i]>0)
				prior1++;
			else
				prior0++;
		}
	}
	SG_SDEBUG("%d pos; %d neg\n", prior1, prior0)

	/* parameter setting */
	/* maximum number of iterations */
	index_t maxiter=100;

	/* minimum step taken in line search */
	float64_t minstep=1E-10;

	/* for numerically strict pd of hessian */
	float64_t sigma=1E-12;
	float64_t eps=1E-5;

	/* construct target support */
	float64_t hiTarget=(prior1+1.0)/(prior1+2.0);
	float64_t loTarget=1/(prior0+2.0);
	index_t length=prior1+prior0;

	SGVector<float64_t> t(length);
	for (index_t i=0; i<length; ++i)
	{
		if (scores[i]>0)
			t[i]=hiTarget;
		else
			t[i]=loTarget;
	}

	/* initial Point and Initial Fun Value */
	/* result parameters of sigmoid */
	float64_t a=0;
	float64_t b=CMath::log((prior0+1.0)/(prior1+1.0));
	float64_t fval=0.0;

	for (index_t i=0; i<length; ++i)
	{
		float64_t fApB=scores[i]*a+b;
		if (fApB>=0)
			fval+=t[i]*fApB+CMath::log(1+CMath::exp(-fApB));
		else
			fval+=(t[i]-1)*fApB+CMath::log(1+CMath::exp(fApB));
	}

	index_t it;
	float64_t g1;
	float64_t g2;
	for (it=0; it<maxiter; ++it)
	{
		SG_SDEBUG("Iteration %d, a=%f, b=%f, fval=%f\n", it, a, b, fval)

		/* Update Gradient and Hessian (use H' = H + sigma I) */
		float64_t h11=sigma; //Numerically ensures strict PD
		float64_t h22=h11;
		float64_t h21=0;
		g1=0;
		g2=0;

		for (index_t i=0; i<length; ++i)
		{
			float64_t fApB=scores[i]*a+b;
			float64_t p;
			float64_t q;
			if (fApB>=0)
			{
				p=CMath::exp(-fApB)/(1.0+CMath::exp(-fApB));
				q=1.0/(1.0+CMath::exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+CMath::exp(fApB));
				q=CMath::exp(fApB)/(1.0+CMath::exp(fApB));
			}

			float64_t d2=p*q;
			h11+=scores[i]*scores[i]*d2;
			h22+=d2;
			h21+=scores[i]*d2;
			float64_t d1=t[i]-p;
			g1+=scores[i]*d1;
			g2+=d1;
		}

		/* Stopping Criteria */
		if (CMath::abs(g1)<eps && CMath::abs(g2)<eps)
			break;

		/* Finding Newton direction: -inv(H') * g */
		float64_t det=h11*h22-h21*h21;
		float64_t dA=-(h22*g1-h21*g2)/det;
		float64_t dB=-(-h21*g1+h11*g2)/det;
		float64_t gd=g1*dA+g2*dB;

		/* Line Search */
		float64_t stepsize=1;

		while (stepsize>=minstep)
		{
			float64_t newA=a+stepsize*dA;
			float64_t newB=b+stepsize*dB;

			/* New function value */
			float64_t newf=0.0;
			for (index_t i=0; i<length; ++i)
			{
				float64_t fApB=scores[i]*newA+newB;
				if (fApB>=0)
					newf+=t[i]*fApB+CMath::log(1+CMath::exp(-fApB));
				else
					newf+=(t[i]-1)*fApB+CMath::log(1+CMath::exp(fApB));
			}

			/* Check sufficient decrease */
			if (newf<fval+0.0001*stepsize*gd)
			{
				a=newA;
				b=newB;
				fval=newf;
				break;
			}
			else
				stepsize=stepsize/2.0;
		}

		if (stepsize<minstep)
		{
			SG_SWARNING("CStatistics::fit_sigmoid(): line search fails, A=%f, "
					"B=%f, g1=%f, g2=%f, dA=%f, dB=%f, gd=%f\n",
					a, b, g1, g2, dA, dB, gd);
		}
	}

	if (it>=maxiter-1)
	{
		SG_SWARNING("CStatistics::fit_sigmoid(): reaching maximal iterations,"
				" g1=%f, g2=%f\n", g1, g2);
	}

	SG_SDEBUG("fitted sigmoid: a=%f, b=%f\n", a, b)

	CStatistics::SigmoidParamters result;
	result.a=a;
	result.b=b;

	SG_SDEBUG("leaving CStatistics::fit_sigmoid()\n")
	return result;
}

const float64_t CStatistics::ERFC_CASE1=0.0492;

const float64_t CStatistics::ERFC_CASE2=-11.3137;
