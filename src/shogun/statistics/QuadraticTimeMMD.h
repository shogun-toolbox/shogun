/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __QUADRACTIMEMMD_H_
#define __QUADRACTIMEMMD_H_

#include <shogun/statistics/KernelTwoSampleTestStatistic.h>

namespace shogun
{

class CFeatures;
class CKernel;
class CCustomKernel;

/** Enum to select which statistic type of quadratic time MMD should be computed */
enum EQuadraticMMDType
{
	BIASED, UNBIASED
};

/** @brief This class implements the quadratic time Maximum Mean Statistic as
 * described in [1].
 * The MMD is the distance of two probability distributions \f$p\f$ and \f$q\f$
 * in a RKHS
 * \f[
 * \text{MMD}[\mathcal{F},p,q]^2=\textbf{E}_{x,x'}\left[ k(x,x')\right]-
 * 2\textbf{E}_{x,y}\left[ k(x,y)\right]
 * +\textbf{E}_{y,y'}\left[ k(y,y')\right]=||\mu_p - \mu_q||^2_\mathcal{F}
 * \f]
 *
 * Given two sets of samples \f$\{x_i\}_{i=1}^m\sim p\f$ and
 * \f$\{y_i\}_{i=1}^n\sim q\f$
 * the (unbiased) statistic is computed as
 *
 * \f[
 * \text{MMD}_u^2[\mathcal{F},X,Y]=\frac{1}{m(m-1)}\sum_{i=1}^m\sum_{j\neq i}^m
 * k(x_i,x_j) + \frac{1}{n(n-1)}\sum_{i=1}^n\sum_{j\neq i}^nk(y_i,y_j) - \frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^nk(x_i,y_j)
 * \f]
 *
 * A biased version is
 *
 * \f[
 * \text{MMD}_b^2[\mathcal{F},X,Y]=\frac{1}{m^2}\sum_{i=1}^m\sum_{j=1}^m
 * k(x_i,x_j) + \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^nk(y_i,y_j) -
 * \frac{2}{mn}\sum_{i=1}^m\sum_{j=1}^nk(x_i,y_j)
 * \f]
 *
 * The type (biased/unbiased) can be selected via set_statistic_type().
 * Note that computing the statistic returns m*MMD; same holds for the null
 * distribution samples.
 *
 * Along with the statistic comes a method to compute a p-value based on
 * different methods. Bootstrapping, is also possible. If unsure which one to
 * use, bootstrapping with 250 iterations always is correct (but slow).
 *
 * To choose, use set_null_approximation_method() and choose from.
 *
 * If you do not know about your data, but want to use the MMD from a kernel
 * matrix, just use the custom kernel constructor. Everything else will work as
 * usual.
 *
 * MMD2_SPECTRUM: for a fast, consistent test based on the spectrum of the kernel
 * matrix, as described in [2]. Only supported if LAPACK is installed.
 *
 * MMD2_GAMMA: for a very fast, but not consistent test based on moment matching
 * of a Gamma distribution, as described in [2].
 *
 * BOOTSTRAPPING: For permuting available samples to sample null-distribution
 *
 * For kernel selection see CMMDKernelSelection.
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 *
 * [2]: Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
 * A fast, consistent kernel two-sample test.
 *
 */
class CQuadraticTimeMMD : public CKernelTwoSampleTestStatistic
{
	public:
		CQuadraticTimeMMD();

		/** Constructor
		 *
		 * @param p_and_q feature data. Is assumed to contain samples from both
		 * p and q. First all samples from p, then from index m all
		 * samples from q
		 *
		 * @param kernel kernel to use
		 * @param p_and_q samples from p and q, appended
		 * @param m index of first sample of q
		 */
		CQuadraticTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t m);

		/** Constructor.
		 * This is a convienience constructor which copies both features to one
		 * element and then calls the other constructor. Needs twice the memory
		 * for a short time
		 *
		 * @param kernel kernel for MMD
		 * @param p samples from distribution p, will be copied and NOT
		 * SG_REF'ed
		 * @param q samples from distribution q, will be copied and NOT
		 * SG_REF'ed
		 */
		CQuadraticTimeMMD(CKernel* kernel, CFeatures* p, CFeatures* q);

		/** Constructor.
		 * This is a convienience constructor which copies allows to only specify
		 * a custom kernel. In this case, the features are completely ignored
		 * and all computations will be done on the custom kernel
		 *
		 * @param custom_kernel custom kernel for MMD, which is a kernel between
		 * the appended features p and q
		 * @param m index of first sample of q
		 */
		CQuadraticTimeMMD(CCustomKernel* custom_kernel, index_t m);

		virtual ~CQuadraticTimeMMD();

		/** Computes the squared quadratic time MMD for the current data. Note
		 * that the type (biased/unbiased) can be specified with
		 * set_statistic_type() method. Note that it returns m*MMD.
		 *
		 * @return (biased or unbiased) squared quadratic time MMD
		 */
		virtual float64_t compute_statistic();

		/** Same as compute_statistic(), but with the possibility to perform on
		 * multiple kernels at once
		 *
		 * @param multiple_kernels if true, and underlying kernel is K_COMBINED,
		 * method will be executed on all subkernels on the same data
		 * @return vector of results for subkernels
		 */
		virtual SGVector<float64_t> compute_statistic(bool multiple_kernels);

		/** computes a p-value based on current method for approximating the
		 * null-distribution. The p-value is the 1-p quantile of the null-
		 * distribution where the given statistic lies in.
		 *
		 * Not all methods for computing the p-value are compatible with all
		 * methods of computing the statistic (biased/unbiased).
		 *
		 * @param statistic statistic value to compute the p-value for
		 * @return p-value parameter statistic is the (1-p) percentile of the
		 * null distribution
		 */
		virtual float64_t compute_p_value(float64_t statistic);

		/** computes a threshold based on current method for approximating the
		 * null-distribution. The threshold is the value that a statistic has
		 * to have in ordner to reject the null-hypothesis.
		 *
		 * Not all methods for computing the p-value are compatible with all
		 * methods of computing the statistic (biased/unbiased).
		 *
		 * @param alpha test level to reject null-hypothesis
		 * @return threshold for statistics to reject null-hypothesis
		 */
		virtual float64_t compute_threshold(float64_t alpha);

		virtual const char* get_name() const
		{
			return "QuadraticTimeMMD";
		};

		/** returns the statistic type of this test statistic */
		virtual EStatisticType get_statistic_type() const
		{
			return S_QUADRATIC_TIME_MMD;
		}

#ifdef HAVE_LAPACK
		/** Returns a set of samples of an estimate of the null distribution
		 * using the Eigen-spectrum of the centered kernel matrix of the merged
		 * samples of p and q. May be used to compute p_value (easy)
		 *
		 * kernel matrix needs to be stored in memory
		 *
		 * Note that the provided statistic HAS to be the biased version
		 * (see paper for details). Note that m*Null-distribution is returned,
		 * which is fine since the statistic is also m*MMD:
		 *
		 * Works well if the kernel matrix is NOT diagonal dominant.
		 * See Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
		 * A fast, consistent kernel two-sample test.
		 *
		 * @param num_samples number of samples to draw
		 * @param num_eigenvalues number of eigenvalues to use to draw samples
		 * Maximum number of 2m-1 where m is the size of both sets of samples.
		 * It is usually safe to use a smaller number since they decay very
		 * fast, however, a conservative approach would be to use all (-1 does
		 * this). See paper for details.
		 * @return samples from the estimated null distribution
		 */
		SGVector<float64_t> sample_null_spectrum(index_t num_samples,
				index_t num_eigenvalues);
#endif // HAVE_LAPACK

		/** setter for number of samples to use in spectrum based p-value
		 * computation.
		 *
		 * @param num_samples_spectrum number of samples to draw from
		 * approximate null-distributrion
		 */
		void set_num_samples_sepctrum(index_t num_samples_spectrum);

		/** setter for number of eigenvalues to use in spectrum based p-value
		 * computation. Maximum is 2*m_m-1
		 *
		 * @param num_eigenvalues_spectrum number of eigenvalues to use to
		 * approximate null-distributrion
		 */
		void set_num_eigenvalues_spectrum(index_t num_eigenvalues_spectrum);

		/** @param statistic_type statistic type (biased/unbiased) to use */
		void set_statistic_type(EQuadraticMMDType statistic_type);

		/** Approximates the null-distribution by the two parameter gamma
		 * distribution. It works in O(m^2) where m is the number of samples
		 * from each distribution. Its very fast, but may be inaccurate.
		 * However, there are cases where it performs very well.
		 * Returns parameters of gamma distribution that is fitted.
		 *
		 * Called by compute_p_value() if null approximation method is set to
		 * MMD2_GAMMA.
		 *
		 * Note that when being used for constructing a test, the provided
		 * statistic HAS to be the biased version
		 * (see paper for details). Note that m*Null-distribution is fitted,
		 * which is fine since the statistic is also m*MMD.
		 *
		 * See Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
		 * A fast, consistent kernel two-sample test.
		 *
		 * @return vector with two parameter for gamma distribution. To use:
		 * call gamma_cdf(statistic, a, b).
		 */
		SGVector<float64_t> fit_null_gamma();

	protected:
		/** helper method to compute m*unbiased squared quadratic time MMD */
		virtual float64_t compute_unbiased_statistic();

		/** helper method to compute m*biased squared quadratic time MMD */
		virtual float64_t compute_biased_statistic();

	private:
		void init();

	protected:
		/** number of samples for spectrum null-dstribution-approximation */
		index_t m_num_samples_spectrum;

		/** number of Eigenvalues for spectrum null-dstribution-approximation */
		index_t m_num_eigenvalues_spectrum;

		/** type of statistic (biased/unbiased) */
		EQuadraticMMDType m_statistic_type;
};

}

#endif /* __QUADRACTIMEMMD_H_ */
