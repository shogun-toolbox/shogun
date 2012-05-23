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

#include <shogun/statistics/TwoSampleTestStatistic.h>

namespace shogun
{

class CFeatures;
class CKernel;

class CKernelTwoSampleTestStatistic : public CTwoSampleTestStatistic
{
	public:
		CKernelTwoSampleTestStatistic();
		CKernelTwoSampleTestStatistic(CKernel* kernel, CFeatures* p_and_q,
				index_t q_start);

		virtual ~CKernelTwoSampleTestStatistic();

		inline virtual const char* get_name() const=0;

	private:
		void init();

	protected:
		CKernel* m_kernel;
};

}

#endif /* __KERNELTWOSAMPLETESTSTATISTIC_H_ */
