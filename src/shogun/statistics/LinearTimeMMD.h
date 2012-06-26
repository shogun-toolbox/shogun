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

/** @brief This class implements the linear time Maximum Mean Statistic as
 * described in
 * ﻿Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 *
 * Along with the statistic comes a method to compute a p-value based on a
 * Gaussian approximation of the null-distribution which is also possible in
 * linear time and constant space. Bootstrapping, of course, is also possible.
 *
 * To choose, use set_p_value_method(MMD1_GAUSSIAN). *
 */
class CLinearTimeMMD: public CKernelTwoSampleTestStatistic
{
public:
	CLinearTimeMMD();
	CLinearTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t q_start);

	virtual ~CLinearTimeMMD();

	/** Computes the squared linear time MMD for the current data. his is an
	 * unbiased estimate
	 *
	 * @return squared linear time MMD
	 */
	virtual float64_t compute_statistic();

	/** Computes the p-value for a given statistic. The method for computing
	 * the p-value can be set via set_p_value_method() method. Since the null-
	 * distribution is normal, a Gaussian approximation is available along with
	 * bootstrapping
	 *
	 * @param statistic statistic to compute the p-value for
	 *
	 * @return p-value of the given statistic
	 */
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

