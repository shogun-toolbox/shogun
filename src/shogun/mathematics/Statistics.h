/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * ALGLIB Copyright 1984, 1987, 1995, 2000 by Stephen L. Moshier under GPL2+
 * http://www.alglib.net/
 * See method comments which functions are taken from ALGLIB (with adjustments
 * for shogun)
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

	/** Calculates median of given values. The median is the value that one
	 * gets when the input vector is sorted and then selects the middle value.
	 *
	 * QuickSelect method copyright:
	 * This Quickselect routine is based on the algorithm described in
	 * "Numerical recipes in C", Second Edition,
	 * Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
	 * This code by Nicolas Devillard - 1998. Public domain.
	 *
	 * Torben method copyright:
	 * The following code is public domain.
	 * Algorithm by Torben Mogensen, implementation by N. Devillard.
	 * Public domain.
	 *
	 * Both methods adapted to SHOGUN by Heiko Strathmann.
	 *
	 * @param values vector of values
	 * @param modify if false, array is modified while median is computed
	 * (Using QuickSelect).
	 * If true, median is computed without modifications, which is slower.
	 * There are two methods to choose from.
	 * @param in_place if set false, the vector is copied and then computed
	 * using QuickSelect. If set true, median is computed in-place using
	 * Torben method.
	 * @return median of given values
	 */
	static float64_t median(SGVector<float64_t> values, bool modify=false,
			bool in_place=false);

	/** Calculates median of given values. Matrix is seen as a long vector for
	 * this. The median is the value that one
	 * gets when the input vector is sorted and then selects the middle value.
	 *
	 * This method is just a wrapper for median(). See this method for license
	 * of QuickSelect and Torben.
	 *
	 * @param values vector of values
	 * @param modify if false, array is modified while median is computed
	 * (Using QuickSelect).
	 * If true, median is computed without modifications, which is slower.
	 * There are two methods to choose from.
	 * @param in_place if set false, the vector is copied and then computed
	 * using QuickSelect. If set true, median is computed in-place using
	 * Torben method.
	 * @return median of given values
	 */
	static float64_t matrix_median(SGMatrix<float64_t> values,
			bool modify=false, bool in_place=false);

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

#ifdef HAVE_LAPACK
	/** Computes the empirical estimate of the covariance matrix of the given
	 * data which is organized as num_cols variables with num_rows observations.
	 *
	 * Data is centered before matrix is computed. May be done in place.
	 * In this case, the observation matrix is changed (centered).
	 *
	 * Given sample matrix \f$X\f$, first, column mean is removed to create
	 * \f$\bar X\f$. Then \f$\text{cov}(X)=(X-\bar X)^T(X - \bar X)\f$ is
	 * returned.
	 *
	 * Needs SHOGUN to be compiled with LAPACK.
	 *
	 * @param observations data matrix organized as one variable per column
	 * @param in_place optional, if set to true, observations matrix will be
	 * centered, if false, a copy will be created an centered.
	 * @return covariance matrix empirical estimate
	 */
	static SGMatrix<float64_t> covariance_matrix(
			SGMatrix<float64_t> observations, bool in_place=false);
#endif //HAVE_LAPACK

	/** Calculates the sample mean of a given set of samples and also computes
	 * the confidence interval for the actual mean for a given p-value,
	 * assuming that the actual variance and mean are unknown (These are
	 * estimated by the samples). Based on Student's t-distribution.
	 *
	 * Only for normally distributed data
	 *
	 * @param values vector of values that are used for calculations
	 * @param alpha actual mean lies in confidence interval with (1-alpha)*100%
	 * @param conf_int_low lower confidence interval border is written here
	 * @param conf_int_up upper confidence interval border is written here
	 * @return sample mean
	 *
	 */
	static float64_t confidence_intervals_mean(SGVector<float64_t> values,
			float64_t alpha, float64_t& conf_int_low, float64_t& conf_int_up);

	/** Functional inverse of Student's t distribution
	 *
	 * Given probability \f$p\f$, finds the argument \f$t\f$ such that
	 * \f$\text{student\_t}(k,t)=p\f$
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t inverse_student_t(int32_t k, float64_t p);

	/** Inverse of incomplete beta integral
	 *
	 * Given \f$y\f$, the function finds \f$x\f$ such that
	 *
	 * \f$\text{inverse\_incomplete\_beta}( a, b, x ) = y .\f$
	 *
	 * The routine performs interval halving or Newton iterations to find the
	 * root of \f$\text{inverse\_incomplete\_beta}( a, b, x )-y=0.\f$
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t inverse_incomplete_beta(float64_t a, float64_t b,
			float64_t y);

	/** Incomplete beta integral
	 *
	 * Returns incomplete beta integral of the arguments, evaluated
	 * from zero to \f$x\f$.  The function is defined as
	 * \f[
	 * \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\int_0^x t^{a-1} (1-t)^{b-1} dt.
	 * \f]
	 *
	 * The domain of definition is \f$0 \leq x \leq 1\f$.  In this
	 * implementation \f$a\f$ and \f$b\f$ are restricted to positive values.
	 * The integral from \f$x\f$ to \f$1\f$ may be obtained by the symmetry
	 * relation
	 *
	 * \f[
	 * 1-\text{incomplete\_beta}(a,b,x)=\text{incomplete\_beta}(b,a,1-x).
	 * \f]
	 *
	 * The integral is evaluated by a continued fraction expansion
	 * or, when \f$b\cdot x\f$ is small, by a power series.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t incomplete_beta(float64_t a, float64_t b, float64_t x);

	/** Inverse of Normal distribution function
	 *
	 * Returns the argument, \f$x\f$, for which the area under the
	 * Gaussian probability density function (integrated from
	 * minus infinity to \f$x\f$) is equal to \f$y\f$.
	 *
	 *
	 * For small arguments \f$0 < y < \exp(-2)\f$, the program computes
	 * \f$z = \sqrt{ -2.0  \log(y) }\f$;  then the approximation is
	 * \f$x = z - \frac{log(z)}{z}  - \frac{1}{z} \frac{P(\frac{1}{z})}{ Q(\frac{1}{z}}\f$.
	 * There are two rational functions \f$\frac{P}{Q}\f$, one for \f$0 < y < \exp(-32)\f$
	 * and the other for \f$y\f$ up to \f$\exp(-2)\f$.  For larger arguments,
	 * \f$w = y - 0.5\f$, and  \f$\frac{x}{\sqrt{2\pi}} = w + w^3 R(\frac{w^2)}{S(w^2)})\f$.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t inverse_normal_cdf(float64_t y0);

	/** same as other version, but with custom mean and variance */
	static float64_t inverse_normal_cdf(float64_t y0, float64_t mean,
				float64_t std_dev);

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

	/** Incomplete gamma integral
	 *
	 * Given \f$p\f$, the function finds \f$x\f$ such that
	 *
	 * \f[
	 * \text{incomplete\_gamma}(a,x)=\frac{1}{\Gamma(a)}}\int_0^x e^{-t} t^{a-1} dt.
	 * \f]
	 *
	 *
	 * In this implementation both arguments must be positive.
	 * The integral is evaluated by either a power series or
	 * continued fraction expansion, depending on the relative
	 * values of \f$a\f$ and \f$x\f$.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t incomplete_gamma(float64_t a, float64_t x);

	/** Complemented incomplete gamma integral
	 *
	 * The function is defined by
	 *
	 * \f[
	 * \text{incomplete\_gamma\_completed}(a,x)=1-\text{incomplete\_gamma}(a,x) =
	 * \frac{1}{\Gamma (a)}\int_x^\infty e^{-t} t^{a-1} dt
	 * \f]
	 *
	 * In this implementation both arguments must be positive.
	 * The integral is evaluated by either a power series or
	 * continued fraction expansion, depending on the relative
	 * values of \f$a\f$ and \f$x\f$.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t incomplete_gamma_completed(float64_t a, float64_t x);

	/** Evaluates the CDF of the gamma distribution with given parameters \f$a, b\f$
	 * at \f$x\f$. Based on Wikipedia definition and ALGLIB routines.
	 *
	 * @param x position to evaluate
	 * @param a shape parameter
	 * @param b scale parameter
	 * @return gamma CDF at \f$x\f$
	 */
	static float64_t gamma_cdf(float64_t x, float64_t a, float64_t b);

	/** Evaluates the inverse CDF of the gamma distribution with given
	 * parameters \f$a\f$, \f$b\f$ at \f$x\f$, such that result equals
	 * \f$\text{gamma\_cdf}(x,a,b)\f$.
	 *
	 * @param p position to evaluate
	 * @param a shape parameter
	 * @param b scale parameter
	 * @return \f$x\f$ such that result equals \f$\text{gamma\_cdf}(x,a,b)\f$.
	 */
	static float64_t inverse_gamma_cdf(float64_t p, float64_t a, float64_t b);

	/** Inverse of complemented incomplete gamma integral
	 *
	 * Given \f$p\f$, the function finds \f$x\f$ such that
	 *
	 * \f$\text{inverse\_incomplete\_gamma\_completed}( a, x ) = p.\f$
	 *
	 * Starting with the approximate value \f$ x=a t^3\f$, where
	 * \f$ t = 1 - d - \text{ndtri}(p) \sqrt{d} \f$ and \f$ d = \frac{1}{9}a \f$
	 *
	 * The routine performs up to 10 Newton iterations to find the
	 * root of \f$ \text{inverse\_incomplete\_gamma\_completed}( a, x )-p=0\f$
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t inverse_incomplete_gamma_completed(float64_t a,
			float64_t y0);

	/** Normal distribution function
	 *
	 * Returns the area under the Gaussian probability density
	 * function, integrated from minus infinity to \f$x\f$:
	 *
	 * \f[
	 * \text{normal\_cdf}(x)=\frac{1}{\sqrt{2\pi}} \int_{-\infty}^x
	 * \exp \left( -\frac{t^2}{2} \right) dt = \frac{1+\text{error\_function}(z) }{2}
	 * \f]
	 *
	 * where \f$ z = \frac{x}{\sqrt{2} \sigma}\f$ and \f$ \sigma \f$ is the standard
	 * deviation. Computation is via the functions \f$\text{error\_function}\f$
	 * and \f$\text{error\_function\_completement}\f$.
	 *
	 * Taken from ALGLIB under gpl2+
	 * Custom variance added by Heiko Strathmann
	 */
	static float64_t normal_cdf(float64_t x, float64_t std_dev=1);

	/** returns logarithm of the cumulative distribution function
	 * (CDF) of Gaussian distribution \f$N(0, 1)\f$:
	 *
	 * \f[
	 * \text{lnormal\_cdf}(x)=log\left(\frac{1}{2}+
	 * \frac{1}{2}\text{error\_function}(\frac{x}{\sqrt{2}})\right)
	 * \f]
	 *
	 * This method uses asymptotic expansion for \f$x<-10.0\f$,
	 * otherwise it returns \f$log(\text{normal\_cdf}(x))\f$.
	 *
	 * @param x real value
	 *
	 * @return \f$log(\text{normal\_cdf}(x))\f$
	 */
	static float64_t lnormal_cdf(float64_t x);

	/** Evaluates the CDF of the chi square distribution with 
	 * parameter k at \f$x\f$. Based on Wikipedia definition.
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

	/** Error function
	 *
	 * The integral is
	 *
	 * \f[
	 * \text{error\_function}(x)=
	 * \frac{2}{\sqrt{pi}}\int_0^x \exp (-t^2) dt
	 * \f]
	 *
	 * For \f$0 \leq |x| < 1, \text{error\_function}(x) = x \frac{P4(x^2)}{Q5(x^2)}\f$
	 * otherwise
	 * \f$\text{error\_function}(x) = 1 - \text{error\_function\_complement}(x)\f$.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t error_function(float64_t x);

	/** Complementary error function
	 *
	 * \f[
	 * 1 - \text{error\_function}(x) =
	 * \text{error\_function\_complement}(x)=
	 * \frac{2}{\sqrt{\pi}}\int_x^\infty \exp\left(-t^2 \right)dt
	 * \f]
	 *
	 * For small \f$x\f$, \f$\text{error\_function\_complement}(x) =
	 *  1 - \text{error\_function}(x)\f$; otherwise rational
	 * approximations are computed.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t error_function_complement(float64_t x);

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

#ifdef HAVE_EIGEN3
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
#endif //HAVE_EIGEN3

	/** Magic number for computing lnormal_cdf */
	static const float64_t ERFC_CASE1;

	/** Magic number for computing lnormal_cdf */
	static const float64_t ERFC_CASE2;

protected:
	/** Power series for incomplete beta integral.
	 * Use when \f$bx\f$ is small and \f$x\f$ not too close to \f$1\f$.
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetaps(float64_t a, float64_t b,
			float64_t x, float64_t maxgam);

	/** Continued fraction expansion #1 for incomplete beta integral
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetafe(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);

	/** Continued fraction expansion #2 for incomplete beta integral
	 *
	 * Taken from ALGLIB under gpl2+
	 */
	static float64_t ibetaf_incompletebetafe2(float64_t a, float64_t b,
			float64_t x, float64_t big, float64_t biginv);

	/** method to make ALGLIB integration easier */
	static inline bool equal(float64_t a, float64_t b) { return a==b; }

	/** method to make ALGLIB integration easier */
	static inline bool not_equal(float64_t a, float64_t b) { return a!=b; }

	/** method to make ALGLIB integration easier */
	static inline bool less(float64_t a, float64_t b) { return a<b; }

	/** method to make ALGLIB integration easier */
	static inline bool less_equal(float64_t a, float64_t b) { return a<=b; }

	/** method to make ALGLIB integration easier */
	static inline bool greater(float64_t a, float64_t b) { return a>b; }

	/** method to make ALGLIB integration easier */
	static inline bool greater_equal(float64_t a, float64_t b) { return a>=b; }
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
