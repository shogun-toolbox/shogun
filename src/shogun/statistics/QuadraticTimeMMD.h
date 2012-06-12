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
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;

class CQuadraticTimeMMD : public CKernelTwoSampleTestStatistic
{
	public:
		CQuadraticTimeMMD();
		CQuadraticTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t q_start);

		virtual ~CQuadraticTimeMMD();

		virtual float64_t compute_statistic();
		virtual float64_t compute_p_value(float64_t statistic);

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

		/** Approximates the null-distribution by the two parameter gamma
		 * distribution. It works in O(m^2) where m is the number of samples
		 * from each distribution. Its very fast, but may be inaccurate.
		 * However, there are cases where it performs very well.
		 * Returns the p-value for a given statistic value in the
		 * null-distribution.
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

		/** setter for number of samples to use in spectrum based p-value
		 * computation
		 *
		 * @param num_samples_spectrum number of samples to draw from
		 * approximate null-distributrion
		 */
		void set_num_samples_sepctrum(index_t num_samples_spectrum);

		/** setter for number of eigenvalues to use in spectrum based p-value
		 * computation
		 *
		 * @param num_eigenvalues_spectrum number of eigenvalues to use to
		 * approximate null-distributrion
		 */
		void set_num_eigenvalues_spectrum(index_t num_eigenvalues_spectrum);

	private:
		void init();

	protected:
		index_t m_num_samples_spectrum;
		index_t m_num_eigenvalues_spectrum;
};

}

#endif /* __QUADRACTIMEMMD_H_ */
