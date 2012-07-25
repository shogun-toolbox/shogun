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
 * two-sample test, i.e. Given samples from two distributions p and q, the
 * null-hypothesis is: H0: p==q, the alternative hypothesis: H1: p!=q.
 *
 * In this class, this is done using a kernel for the data.
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

		inline virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		/** underlying kernel */
		CKernel* m_kernel;
};

}

#endif /* __KERNELTWOSAMPLETESTSTATISTIC_H_ */
