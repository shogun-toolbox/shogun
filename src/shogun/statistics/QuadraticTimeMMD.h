/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __QUADRACTIMEMMD_H_
#define __QUADRACTIMEMMD_H_

#include <shogun/statistics/KernelTwoSampleTestStatistic.h>

namespace shogun
{

class CFeatures;
class CKernel;

enum EQuadraticMMDType
{
	BIASED, UNBIASED
};

/** @brief Class for the quadratic time MMD.
 *
 * Allows to perform a kernel based two-sample test using empirical estimates of
 * the quadratic time MMD, which is
 *
 * TODO when I have internet :)
 *
 * It is possible to use two types:
 * Biased, that is: TODO
 * Unbiased, that is: TODO
 *
 * See
 * Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 *
 * To choose, use set_statistic_type()
 *
 * To approximate the null-distribution in order to compute a p-value, currenlty,
 * in addition to bootstrapping (see CTwoSampleTestStatistic), two methods are
 * available (both based on the biased squared MMD):
 *
 * 1. A method that is based on the Eigenspectrum of the gram matrix of the
 * underlying data. (Only supported if LAPACK is installed)
 *
 * 2. A method that is based on moment matching of a Gamma distribution
 *
 * Both methods are described in
 * Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
 * A fast, consistent kernel two-sample test.
 *
 * To choose, use CTwoSampleTestStatistic::set_null_approximation_method()
 *
 */
class CQuadraticTimeMMD : public CKernelTwoSampleTestStatistic
{
	public:
		CQuadraticTimeMMD();

		/** Constructor
		 *
		 * @param p_and_q feature data. Is assumed to contain samples from both
		 * p and q. First all samples from p, then from index q_start all
		 * samples from q
		 *
		 * @param kernel kernel to use
		 * @param p_and_q samples from p and q, appended
		 * @param q_start index of first sample of q
		 */
		CQuadraticTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t q_start);

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

		virtual ~CQuadraticTimeMMD();

		/** Computes the squared quadratic time MMD for the current data. Note
		 * that the type (biased/unbiased) can be specified with
		 * set_statistic_type() method.
		 *
		 * @return (biased or unbiased) squared quadratic time MMD
		 */
		virtual float64_t compute_statistic();

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

		inline virtual const char* get_name() const
		{
			return "QuadraticTimeMMD";
		};

#ifdef HAVE_LAPACK
		/* returns a set of samples of an estimate of the null distribution
		 * using the Eigen-spectrum of the centered kernel matrix of the merged
		 * samples of p and q. May be used to compute p_value (easy)
		 *
		 * kernel matrix needs to be stored in memory
		 *
		 * Note that the provided statistic HAS to be the biased version
		 * (see paper for details)
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
		 * computation. Maximum is 2*m_q_start-1
		 *
		 * @param num_eigenvalues_spectrum number of eigenvalues to use to
		 * approximate null-distributrion
		 */
		void set_num_eigenvalues_spectrum(index_t num_eigenvalues_spectrum);

		/** @param statistic_type statistic type (biased/unboased) to use */
		void set_statistic_type(EQuadraticMMDType statistic_type);

	protected:
		/** Approximates the null-distribution by the two parameter gamma
		 * distribution. It works in O(m^2) where m is the number of samples
		 * from each distribution. Its very fast, but may be inaccurate.
		 * However, there are cases where it performs very well.
		 * Returns the p-value for a given statistic value in the
		 * null-distribution.
		 *
		 * Called by compute_p_value() if null approximation method is set to
		 * MMD2_GAMMA.
		 *
		 * Note that the provided statistic HAS to be the biased version
		 * (see paper for details)
		 *
		 * Works for arbritarily large kernel matrices (is not precomputed)
		 *
		 * See Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
		 * A fast, consistent kernel two-sample test.
		 *
		 * @param statistic MMD value to compute the p-value for.
		 * @return p-value of the given statistic
		 */
		float64_t compute_p_value_gamma(float64_t statistic);

		/** helper method to compute unbiased squared quadratic time MMD */
		virtual float64_t compute_unbiased_statistic();

		/** helper method to compute biased squared quadratic time MMD */
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
