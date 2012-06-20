/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __LINEARTIMEMMD_H_
#define __LINEARTIMEMMD_H_

#include <shogun/statistics/KernelTwoSampleTestStatistic.h>
#include <shogun/kernel/Kernel.h>

namespace shogun
{

class CFeatures;

class CLinearTimeMMD: public CKernelTwoSampleTestStatistic
{
public:
	CLinearTimeMMD();
	CLinearTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t q_start);

	virtual ~CLinearTimeMMD();

	virtual float64_t compute_statistic();
	virtual float64_t compute_p_value(float64_t statistic);

	inline virtual const char* get_name() const
	{
		return "LinearTimeMMD";
	}
	;

private:
	void init();
};

}

#endif /* __LINEARTIMEMMD_H_ */

