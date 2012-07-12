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
#include <shogun/lib/external/libqp.h>

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
 * Comes with a method for selecting kernel weights, if a combined kernel on
 * combined features is used. See optimize_kernel_weights().
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
	 * is available.
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
	 * is available.
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

#ifdef HAVE_LAPACK
	/** TODO */
	/** Selects optimal kernel weights (if the underlying kernel and features)
	 * are combined ones) using the ratio of the squared MMD by its standard
	 * deviation as a criterion, i.e.
	 * TODO latex
	 * This comes down to solving a convex program which is quadratic in the
	 * number of kernels.
	 *
	 * SHOGUN has to be compiled with LAPACK to make this available. See
	 * set_opt* methods for optimization parameters.
	 *
	 * IMPORTANT: Kernel weights have to be learned on different data than is
	 * used for testing/evaluation!
	 */
	virtual void optimize_kernel_weights();

	/** Sets the max. number of iterations for optimizing kernel weights */
	void set_opt_max_iterations(index_t opt_max_iterations)
	{
		m_opt_max_iterations=opt_max_iterations;
	}

	/** Sets the stopping criterion epsilon for optimizing kernel weights */
	void set_opt_epsilon(float64_t opt_epsilon) {
		m_opt_epsilon=opt_epsilon;
	}

	/** Sets the low cut for optimizing kernel weights (weight below are set
	 * to zero */
	void set_opt_low_cut(float64_t opt_low_cut)
	{
		m_opt_low_cut=opt_low_cut;
	}

	/** Sets regularization constant. This value is added on diagonal of
	 * matrix for optimizing kernel weights */
	void set_opt_regularization_eps(float64_t opt_regularization_eps)
	{
		m_opt_regularization_eps=opt_regularization_eps;
	}

#endif //HAVE_LAPACK

	inline virtual const char* get_name() const
	{
		return "LinearTimeMMD";
	}

private:
	void init();

public:
	/** return pointer to i-th column of m_Q. Helper for libqp */
	static const float64_t* get_Q_col(uint32_t i);

	/** helper functions that prints current state */
	static void print_state(libqp_state_T state);

protected:
#ifdef HAVE_LAPACK
	/** maximum number of iterations of qp solver */
	index_t m_opt_max_iterations;

	/** stopping accuracy of qp solver */
	float64_t m_opt_epsilon;

	/** low cut for weights, if weights are under this value, are set to zero */
	float64_t m_opt_low_cut;

	/** regularization epsilon that is added to diagonal of Q matrix */
	float64_t m_opt_regularization_eps;

	/** matrix for selection of kernel weights (static because of libqp) */
	static SGMatrix<float64_t> m_Q;
#endif //HAVE_LAPACK
};

}

#endif /* __LINEARTIMEMMD_H_ */

