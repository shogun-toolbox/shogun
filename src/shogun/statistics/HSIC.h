/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#ifndef __HSIC_H_
#define __HSIC_H_

#include <shogun/statistics/KernelIndependenceTestStatistic.h>

namespace shogun
{

/** TODO
 *
 */
class CHSIC : public CKernelIndependenceTestStatistic
{
public:
	/** TODO */
	CHSIC();

	/** Constructor.
	 *
	 * @param kernel_p kernel samples from p
	 * @param kernel_q kernel samples from q
	 * @param p samples from p
	 * @param q samples from q
	 */
	CHSIC(CKernel* kernel_p, CKernel* kernel_q, CFeatures* p, CFeatures* q);

	virtual ~CHSIC();

	/** Computes the biased HSIC TODO */
	virtual float64_t compute_statistic();

	/** computes a p-value based on current method for approximating the
	 * null-distribution. The p-value is the 1-p quantile of the null-
	 * distribution where the given statistic lies in.
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
	 * @param alpha test level to reject null-hypothesis
	 * @return threshold for statistics to reject null-hypothesis
	 */
	virtual float64_t compute_threshold(float64_t statistic);

	inline virtual const char* get_name() const
	{
		return "HSIC";
	}

	/** Approximates the null-distribution by the two parameter gamma
	 * distribution. TODO
	 *
	 * NOTE: the gamma distribution is fitted to m*HSIC_b. Therefore, the
	 * parameter statistic value is multiplied by m before anything is done.
	 * You can safely call this with values from compute_statistic().
	 * However, the attached features have to be the SAME size, as these, the
	 * statistic was computed on.
	 *
	 * Called by compute_p_value() if null approximation method is set to
	 * MMD2_GAMMA.
	 */
	float64_t compute_p_value_gamma(float64_t statistic);

private:
	void init();

};

}

#endif /* __HSIC_H_ */
