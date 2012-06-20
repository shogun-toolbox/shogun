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

	/** computes a linear time estimate of the variance of the squared linear
	 * time mmd, which may be used for an approximation of the null-distribution
	 * The value is the variance of the vector of which the linear time MMD is
	 * the mean.
	 *
	 * @return variance estimate
	 */
	virtual float64_t compute_variance_estimate();

	inline virtual const char* get_name() const
	{
		return "LinearTimeMMD";
	}

private:
	void init();
};

}

#endif /* __LINEARTIMEMMD_H_ */

