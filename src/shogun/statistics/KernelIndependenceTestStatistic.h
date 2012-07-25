/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __KERNELINDEPENDENCESTSTATISTIC_H_
#define __KERNELINDEPENDENCESTSTATISTIC_H_

#include <shogun/statistics/TwoDistributionsTestStatistic.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief Base class for kernel based independence testing. Extends the standard
 * independence class with a kernel for each sample.
 */
class CKernelIndependenceTestStatistic: public CTwoDistributionsTestStatistic
{
public:
	CKernelIndependenceTestStatistic();

	/** Constructor
	 *
	 * @param p_and_q feature data. Is assumed to contain samples from both
	 * p and q. First all samples from p, then from index q_start all
	 * samples from q
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p_and_q samples from p and q, appended
	 * @param q_start index of first sample of q
	 */
	CKernelIndependenceTestStatistic(CKernel* kernel_p, CKernel* kernel_q,
			CFeatures* p_and_q, index_t q_start);

	/** Constructor.
	 * This is a convienience constructor which copies both features to one
	 * element and then calls the other constructor. Needs twice the memory
	 * for a short time
	 *
	 * @param kernel_p kernel to use on samples from p
	 * @param kernel_q kernel to use on samples from q
	 * @param p samples from distribution p, will be copied and NOT
	 * SG_REF'ed
	 * @param q samples from distribution q, will be copied and NOT
	 * SG_REF'ed
	 */
	CKernelIndependenceTestStatistic(CKernel* kernel_p, CKernel* kernel_q,
			CFeatures* p, CFeatures* q);

	virtual ~CKernelIndependenceTestStatistic();

	inline virtual const char* get_name() const=0;

private:
	void init();

protected:
	/** underlying kernel for p */
	CKernel* m_kernel_p;

	/** underlying kernel for q */
	CKernel* m_kernel_q;
};

}

#endif /* __KERNELINDEPENDENCESTSTATISTIC_H_ */
