/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2016 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * Most cdf routines are wrappers for CDFLIB, part of the public domain NETLIB.
 * https://people.sc.fsu.edu/~jburkardt/f_src/cdflib/cdflib.html
 */

#ifndef __STATISTICS_H_
#define __STATISTICS_H_

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>

namespace shogun
{
template<class T> class SGMatrix;
template<class T> class SGSparseMatrix;

/** @brief Class that contains certain functions related to statistics, such as
 * probability/cumulative distribution functions, different statistics, etc.
 */
class CStatistics: public CSGObject
{

public:

	/** Calculates mean of given values. Given \f$\{x_1, ..., x_m\}\f$, this
	 * is \f$\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * @param vec vector of values
	 * @return mean of given values
	 */
	template<class T>
		static floatmax_t mean(SGVector<T> vec)
		{
			floatmax_t sum = 0;

			for ( index_t i = 0 ; i < vec.vlen ; ++i )
				sum += vec[i];

			return sum/vec.vlen;
		}

	/** Calculates unbiased empirical variance estimator of given values. Given
	 * \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * @param values vector of values
	 * @return variance of given values
	 */
	static float64_t variance(SGVector<float64_t> values);

	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * @param values vector of values
	 * @return variance of given values
	 */
	static float64_t std_deviation(SGVector<float64_t> values);

	/** Calculates mean of given values. Given \f$\{x_1, ..., x_m\}\f$, this
	 * is \f$\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * Computes the mean for each row/col of matrix
	 *
	 * @param values vector of values
	 * @param col_wise if true, every column vector will be used, row vectors
	 * otherwise
	 * @return mean of given values
	 */
	static SGVector<float64_t> matrix_mean(SGMatrix<float64_t> values,
			bool col_wise=true);

	/** Calculates unbiased empirical variance estimator of given values. Given
	 * \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * Computes the variance for each row/col of matrix
	 *
	 * @param values vector of values
	 * @param col_wise if true, every column vector will be used, row vectors
	 * otherwise
	 * @return variance of given values
	 */
	static SGVector<float64_t> matrix_variance(SGMatrix<float64_t> values,
			bool col_wise=true);

	/** Calculates unbiased empirical standard deviation estimator of given
	 * values. Given \f$\{x_1, ..., x_m\}\f$, this is
	 * \f$\sqrt{\frac{1}{m-1}\sum_{i=1}^m (x-\bar{x})^2}\f$ where
	 * \f$\bar x=\frac{1}{m}\sum_{i=1}^m x_i\f$
	 *
	 * Computes the variance for each row/col of matrix
	 *
	 * @param values vector of values
	 * @param col_wise if true, every column vector will be used, row vectors
	 * otherwise
	 * @return variance of given values
	 */
	static SGVector<float64_t> matrix_std_deviation(
			SGMatrix<float64_t> values, bool col_wise=true);

	/** Computes the empirical estimate of the DxD covariance matrix of the given
	 * data which is organized as num_cols variables with num_rows observations.
	 * Normalizes by N-1 for N observations
	 *
	 * Data is centered before matrix is computed. May be done in place.
	 * In this case, the observation matrix is changed (centered).
	 *
	 * @param observations Data matrix
	 * @param in_place Optional, if set to true, observations matrix will be
	 * centered, if false, a copy will be created an centered.
	 * @return DxD covariance matrix
	 */
	static SGMatrix<float64_t> covariance_matrix(
			SGMatrix<float64_t> observations, bool in_place=false);

	/** @return natural logarithm of the gamma function of input */
	static inline float64_t lgamma(float64_t x)
	{
		return ::lgamma((double) x);
	}

	/** @return natural logarithm of the gamma function of input for large
	 * numbers */
	static inline floatmax_t lgammal(floatmax_t x)
	{
#ifdef HAVE_LGAMMAL
		return ::lgammal((long double) x);
#else
		return ::lgamma((double) x);
#endif // HAVE_LGAMMAL
	}

	/** @return gamma function of input */
	static inline float64_t tgamma(float64_t x)
	{
		return ::tgamma((double) x);
	}

	/** Evaluates the CDF of the gamma distribution, parametrized with shape
	 * and rate parameters \f$\alpha, \beta\f$ at \f$x\f$.
	 *
	 * \f[
	 * \frac{\beta^\alpha}{\Gamma(\alpha)}\int _{-\infty}^x x^{\alpha-1}\exp(-t \beta)dt
	 * \f]
	 *
	 * @param x Argument \f$x\f$ to evaluate.
	 * @param a Shape parameter \f$\alpha\f$
	 * @param b Rate parameter \f$\beta\f$
	 * @return Gamma CDF at \f$x\f$
	 */
	static float64_t gamma_cdf(float64_t x, float64_t a, float64_t b);

	/** Inverse of Gamma cumulative distribution function, parametrized with shape
	 * and rate parameters \f$\alpha, \beta\f$, given by
	 *
	 * \f[
	 * \frac{\beta^\alpha}{\Gamma(\alpha)}\int _{-\infty}^x x^{\alpha-1}\exp(-t \beta)dt
	 * \f]
	 *
	 * Returns the argument \f$x\f$ for which the CDF is equal to \f$y\f$.
	 *
	 * @param p CDF value \f$y\f$.
	 * @param a Shape parameter \f$\alpha\f$
	 * @param b Rate parameter \f$\beta\f$
	 * @return Argument \f$x\f$ that produces \f$y\f$.
	 */
	static float64_t gamma_inverse_cdf(float64_t p, float64_t a, float64_t b);

	/** Evaluates the CDF of the Normal distribution, with mean \f$\mu\f$ and
	 * standard deviation \f$\sigma\f$, given by
	 *
	 * \f[
	 * y=\frac{1}{\sigma \sqrt {2\pi }} \int_{-\infty}^x \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)dt
	 * \f]
	 *
	 * @param x Argument \f$x\f$ to evaluate.
	 * @param std_dev Standard deviation \f$\sigma\f$. Default value is 1.
	 * @return Gamma CDF at \f$x\f$
	 */
	static float64_t normal_cdf(float64_t x, float64_t std_dev=1);

	/** Inverse of Normal cumulative distribution function with mean \f$\mu\f$ and
	 * standard deviation \f$\sigma\f$, given by
	 *
	 * \f[
	 * y=\frac{1}{\sigma \sqrt {2\pi }} \int_{-\infty}^x \exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)dt
	 * \f]
	 *
	 * Returns the argument \f$x\f$ for which CDF is equal to \f$y\f$.
	 *
	 * @param y0 CDF value \f$y\f$.
	 * @param mean Mean \f$\mu\f$. Default value is 0.
	 * @param std_dev Standard deviation \f$\sigma\f$. Default value is 1.
	 * @return Argument \f$x\f$ that produces \f$y\f$.
	 */
	static float64_t inverse_normal_cdf(float64_t y0, float64_t mean=0,
			float64_t std_dev=1);

	/** Returns logarithm of the cumulative distribution function
	 * (CDF) of Gaussian distribution \f$N(0, 1)\f$:
	 *
	 * \f[
	 * \text{lnormal\_cdf}(x)=log\left(\frac{1}{2}+
	 * \frac{1}{2}\text{error\_function}(\frac{x}{\sqrt{2}})\right)
	 * \f]
	 *
	 * @param x Evaluate CDF here
	 * @return \f$log(\text{normal\_cdf}(x))\f$
	 */
	static float64_t lnormal_cdf(float64_t x);

	/** Evaluates the CDF of the chi square distribution with
	 * parameter k at \f$x\f$.
	 *
	 * @param x position to evaluate
	 * @param k parameter
	 * @return chi square CDF at \f$x\f$
	 */
	static float64_t chi2_cdf(float64_t x, float64_t k);

	/** Evaluates the CDF of the F-distribution with parameters
	 * \f$d1,d2\f$ at \f$x\f$. Based on Wikipedia definition.
	 *
	 * @param x position to evaluate
	 * @param d1 parameter 1
	 * @param d2 parameter 2
	 * @return F-distribution CDF at \f$x\f$
	 */
	static float64_t fdistribution_cdf(float64_t x, float64_t d1, float64_t d2);

	/** Use to estimates erfc(x) valid for -100 < x < -8
	 *
	 * @param x real value
	 * @return weighted sum
	 */
	static float64_t erfc8_weighted_sum(float64_t x);

	/** @return mutual information of \f$p\f$ which is given in logspace
	 * where \f$p,q\f$ are given in logspace */
	static float64_t mutual_info(float64_t* p1, float64_t* p2, int32_t len);

	/** @return relative entropy \f$H(P||Q)\f$
	 * where \f$p,q\f$ are given in logspace */
	static float64_t relative_entropy(
			float64_t* p, float64_t* q, int32_t len);

	/** @return entropy of \f$p\f$ which is given in logspace */
	static float64_t entropy(float64_t* p, int32_t len);

	/** fisher's test for multiple 2x3 tables
	 * @param tables
	 */
	static SGVector<float64_t> fishers_exact_test_for_multiple_2x3_tables(SGMatrix<float64_t> tables);

	/** fisher's test for 2x3 table
	 * @param table
	 */
	static float64_t fishers_exact_test_for_2x3_table(SGMatrix<float64_t> table);

	/** sample indices
	 * @param sample_size size of sample to pick
	 * @param N total number of indices
	 */
	static SGVector<int32_t> sample_indices(int32_t sample_size, int32_t N);

	/** @return object name */
	virtual const char* get_name() const
	{
		return "Statistics";
	}

	 /** Derivative of the log gamma function.
	 *
	 * @param x input
	 * @return derivative of the log gamma input
	 */
	static float64_t dlgamma(float64_t x);

	/** Representation of a Sigmoid function for the fit_sigmoid function */
	struct SigmoidParamters
	{
		/** parameter a */
		float64_t a;

		/** parameter b */
		float64_t b;
	};

	/** Converts a given vector of scores to calibrated probabilities by fitting a
	 * sigmoid function using the method described in
	 * Lin, H., Lin, C., and Weng, R. (2007).
	 * A note on Platt's probabilistic outputs for support vector machines.
	 *
	 * This can be used to transform scores to probabilities as setting
	 * \f$pf=x*a+b\f$ for a given score \f$x\f$ and computing
	 * \f$\frac{\exp(-f)}{1+}exp(-f)}\f$ if \f$f\geq 0\f$ and
	 * \f$\frac{1}{(1+\exp(f)}\f$ otherwise
	 *
	 * @param scores scores to fit the sigmoid to
	 * @return struct containing the sigmoid's shape parameters a and b
	 */
	static SigmoidParamters fit_sigmoid(SGVector<float64_t> scores);

	/** The log determinant of a dense matrix
	 *
	 * If determinant of the input matrix is positive, it returns the logarithm of the value.
	 * If not, it returns CMath::INFTY
	 * Note that the input matrix is not required to be symmetric positive definite.
	 * This method is slower than log_det() if input matrix is known to be symmetric positive definite
	 *
	 * It is adapted from Gaussian Process Machine Learning Toolbox
	 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
	 *
	 * @param A input matrix
	 * @return the log determinant value
	 */

	static float64_t log_det_general(const SGMatrix<float64_t> A);

	/** The log determinant of a dense matrix
	 *
	 * The log determinant of a positive definite symmetric real valued
	 * matrix is calculated as
	 * \f[
	 * \text{log\_determinant}(M)
	 * = \text{log}(\text{determinant}(L)\times\text{determinant}(L'))
	 * = 2\times \sum_{i}\text{log}(L_{i,i})
	 * \f]
	 * Where, \f$M = L\times L'\f$ as per Cholesky decomposition.
	 *
	 * @param m input matrix
	 * @return the log determinant value
	 */
	static float64_t log_det(SGMatrix<float64_t> m);

	/** The log determinant of a sparse matrix
	 *
	 * The log determinant of symmetric positive definite sparse matrix
	 * is calculated in a similar way as the dense case. But using
	 * cholesky decomposition on sparse matrices may suffer from fill-in
	 * phenomenon, i.e. the factors may not be as sparse. The
	 * SimplicialCholesky module for sparse matrix in eigen3 library
	 * uses an approach called approximate minimum degree reordering,
	 * or amd, which permutes the matrix beforehand and results in much
	 * sparser factors. If \f$P\f$ is the permutation matrix, it computes
	 * \f$\text{LLT}(P\times M\times P^{-1}) = L\times L'\f$.
	 *
	 * @param m input sparse matrix
	 * @return the log determinant value
	 */
	static float64_t log_det(const SGSparseMatrix<float64_t> m);

	/** Sampling from a multivariate Gaussian distribution with
	 * dense covariance matrix
	 *
	 * Sampling is performed by taking samples from \f$N(0, I)\f$, then
	 * using cholesky factor of the covariance matrix, \f$\Sigma\f$ and
	 * performing
	 * \f[S_{N(\mu,\Sigma)}=S_{N(0,I)}*L^{T}+\mu\f]
	 * where \f$\Sigma=L*L^{T}\f$ and \f$\mu\f$ is the mean vector.
	 *
	 * @param mean the mean vector
	 * @param cov the covariance matrix
	 * @param N number of samples
	 * @param precision_matrix if true, sample from N(mu,C^-1)
	 * @return the sample matrix of size \f$N\times dim\f$
	 */
	static SGMatrix<float64_t> sample_from_gaussian(SGVector<float64_t> mean,
	SGMatrix<float64_t> cov, int32_t N=1, bool precision_matrix=false);

	/** Sampling from a multivariate Gaussian distribution with
	 * sparse covariance matrix
	 *
	 * Sampling is performed in similar way as of dense covariance
	 * matrix, but direct cholesky factorization of sparse matrices
	 * could be inefficient. So, this method uses permutation matrix
	 * for factorization and then permutes back the final samples
	 * before adding the mean.
	 *
	 * @param mean the mean vector
	 * @param cov the covariance matrix
	 * @param N number of samples
	 * @param precision_matrix if true, sample from N(mu,C^-1)
	 * @return the sample matrix of size \f$N\times dim\f$
	 */
	static SGMatrix<float64_t> sample_from_gaussian(SGVector<float64_t> mean,
	SGSparseMatrix<float64_t> cov, int32_t N=1, bool precision_matrix=false);

	/** Magic number for computing lnormal_cdf */
	static const float64_t ERFC_CASE1;

	/** Magic number for computing lnormal_cdf */
	static const float64_t ERFC_CASE2;


};

/// mean not implemented for complex128_t, returns 0.0 instead
template <>
	inline floatmax_t CStatistics::mean<complex128_t>(SGVector<complex128_t> vec)
	{
		SG_SNOTIMPLEMENTED
		return floatmax_t(0.0);
	}

}

#endif /* __STATISTICS_H_ */
