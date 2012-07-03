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

#include <shogun/statistics/IndependenceTestStatistic.h>

namespace shogun
{

class CFeatures;
class CKernel;

/** @brief Base class for kernel based independence testing. Extends the standard
 * independence class with a kernel for each sample.
 */
class CKernelIndependenceTestStatistic: public CIndependenceTestStatistic
{
public:
	CKernelIndependenceTestStatistic();

	/** Constructor.
	 *
	 * @param kernel_p kernel samples from p
	 * @param kernel_q kernel samples from q
	 * @param p samples from p
	 * @param q samples from q
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
