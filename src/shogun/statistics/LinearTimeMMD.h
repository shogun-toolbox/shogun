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
 * Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 *
 * Along with the statistic comes a method to compute a p-value based on a
 * Gaussian approximation of the null-distribution which is also possible in
 * linear time and constant space. Bootstrapping, of course, is also possible.
 *
 * To choose, use
 * CTwoSampleTestStatistic::set_null_approximation_method(MMD1_GAUSSIAN).
 *
 * IMPORTANT: In order to use the gaussian approximation, the p-value has to
 * be computed on other data than the statistic. Otherwise the null-distribution
 * is not normal.
 */
class CLinearTimeMMD: public CKernelTwoSampleTestStatistic
{
public:
	CLinearTimeMMD();

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
	CLinearTimeMMD(CKernel* kernel, CFeatures* p_and_q, index_t q_start);

	/** Constructor.
	 * This is a convienience constructor which copies both features to one
	 * element and then calls the other constructor. Needs twice the memory
	 * for a short time
	 *
	 * @param kernel kernel for MMD
	 * @param p samples from distribution p, will be copied and NOT
	 * SG_REF'ed
	 * @@param q samples from distribution q, will be copied and NOT
	 * SG_REF'ed
	 */
	CLinearTimeMMD(CKernel* kernel, CFeatures* p, CFeatures* q);

	virtual ~CLinearTimeMMD();

	/** Computes the squared linear time MMD for the current data. This is an
	 * unbiased estimate
	 *
	 * @return squared linear time MMD
	 */
	virtual float64_t compute_statistic();

	/** computes a p-value based on current method for approximating the
	 * null-distribution. The p-value is the 1-p quantile of the null-
	 * distribution where the given statistic lies in.
	 *
	 * The method for computing the p-value can be set via
	 * set_null_approximation_method().
	 * Since the null- distribution is normal, a Gaussian approximation
	 * is available. For Gaussian approximation, training and test data have
	 * to be DIFFERENT samples from same distribution
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
	 * The method for computing the p-value can be set via
	 * set_null_approximation_method().
	 * Since the null- distribution is normal, a Gaussian approximation
	 * is available. For Gaussian approximation, training and test data have
	 * to be DIFFERENT samples from same distribution
	 *
	 * @param alpha test level to reject null-hypothesis
	 * @return threshold for statistics to reject null-hypothesis
	 */
	virtual float64_t compute_threshold(float64_t alpha);

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

