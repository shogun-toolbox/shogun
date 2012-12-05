/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __KERNELTWOSAMPLETESTSTATISTIC_H_
#define __KERNELTWOSAMPLETESTSTATISTIC_H_

#include <shogun/statistics/TwoDistributionsTestStatistic.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief Two sample test base class. Provides an interface for performing a
 * two-sample test, i.e. Given samples from two distributions \f$p\f$ and
 * \f$q\f$, the null-hypothesis is: \f$H_0: p=q\f$, the alternative hypothesis:
 * \f$H_1: p\neq q\f$.
 *
 * In this class, this is done using a single kernel for the data.
 *
 * The class also re-implements the bootstrap_null() method. If the underlying
 * kernel is a custom one (precomputed), the
 *
 * Abstract base class.
 */
class CKernelTwoSampleTestStatistic : public CTwoDistributionsTestStatistic
{
	public:
		CKernelTwoSampleTestStatistic();

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
		CKernelTwoSampleTestStatistic(CKernel* kernel, CFeatures* p_and_q,
				index_t q_start);

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
		CKernelTwoSampleTestStatistic(CKernel* kernel, CFeatures* p,
				CFeatures* q);

		virtual ~CKernelTwoSampleTestStatistic();

		/** Setter for the underlying kernel
		 * @param kernel new kernel to use
		 */
		void set_kernel(CKernel* kernel);

		/** merges both sets of samples and computes the test statistic
		 * m_bootstrap_iteration times. This version checks if a precomputed
		 * custom kernel is used, and, if so, just permutes it instead of re-
		 * computing it in every iteration.
		 *
		 * @return vector of all statistics
		 */
		virtual SGVector<float64_t> bootstrap_null();

		virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		/** underlying kernel */
		CKernel* m_kernel;
};

}

#endif /* __KERNELTWOSAMPLETESTSTATISTIC_H_ */
