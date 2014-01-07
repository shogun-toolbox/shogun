/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#ifndef __LINEARTIMEMMD_H_
#define __LINEARTIMEMMD_H_

#include <statistics/KernelTwoSampleTestStatistic.h>
#include <kernel/Kernel.h>
#include <lib/external/libqp.h>

namespace shogun
{

class CStreamingFeatures;
class CFeatures;

/** @brief This class implements the linear time Maximum Mean Statistic as
 * described in [1]. This statistic is in particular suitable for streaming
 * data. Therefore, only streaming features may be passed. To process other
 * feature types, construct streaming features from these (see constructor
 * documentations). A blocksize has to be specified that determines how many
 * examples are processed at once. This should be set as large as available
 * memory allows to ensure faster computations.
 *
 * The MMD is the distance of two probability distributions \f$p\f$ and \f$q\f$
 * in a RKHS.
 * \f[
 * \text{MMD}}[\mathcal{F},p,q]^2=\textbf{E}_{x,x'}\left[ k(x,x')\right]-
 * 2\textbf{E}_{x,y}\left[ k(x,y)\right]
 * +\textbf{E}_{y,y'}\left[ k(y,y')\right]=||\mu_p - \mu_q||^2_\mathcal{F}
 * \f]
 *
 * Given two sets of samples \f$\{x_i\}_{i=1}^m\sim p\f$ and
 * \f$\{y_i\}_{i=1}^n\sim q\f$
 * the (unbiased) statistic is computed as
 * \f[
 * \text{MMD}_l^2[\mathcal{F},X,Y]=\frac{1}{m_2}\sum_{i=1}^{m_2}
 * h(z_{2i},z_{2i+1})
 * \f]
 * where
 * \f[
 * h(z_{2i},z_{2i+1})=k(x_{2i},x_{2i+1})+k(y_{2i},y_{2i+1})-k(x_{2i},y_{2i+1})-
 * k(x_{2i+1},y_{2i})
 * \f]
 * and \f$ m_2=\lfloor\frac{m}{2} \rfloor\f$.
 *
 * Along with the statistic comes a method to compute a p-value based on a
 * Gaussian approximation of the null-distribution which is also possible in
 * linear time and constant space. Bootstrapping, is also possible (no
 * permutations but new examples will be used here).
 * If unsure which one to use, bootstrapping with 250 iterations always is
 * correct (but slow). When the sample size is large (>1000) at least,
 * the Gaussian approximation is an accurate and much faster choice than
 * bootstrapping.
 *
 * To choose, use set_null_approximation_method() and choose from
 *
 * MMD1_GAUSSIAN: Approximates the null-distribution with a Gaussian. Only use
 * from at least 1000 samples. If using, check if type I error equals the
 * desired value.
 *
 * BOOTSTRAPPING: For permuting available samples to sample null-distribution
 *
 * For kernel selection see CMMDKernelSelection.
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 */
class CLinearTimeMMD: public CKernelTwoSampleTestStatistic
{
public:
	CLinearTimeMMD();

	/** Constructor.
	 * @param kernel kernel to use
	 * @param p streaming features p to use
	 * @param q streaming features q to use
	 * @param m index of first sample of q
	 * @param blocksize size of examples that are processed at once when
	 * computing statistic/threshold. If larger than m/2, all examples will be
	 * processed at once. Memory consumption increased linearly in the
	 * blocksize. Choose as large as possible regarding available memory.
	 */
	CLinearTimeMMD(CKernel* kernel, CStreamingFeatures* p,
			CStreamingFeatures* q, index_t m, index_t blocksize=10000);

	virtual ~CLinearTimeMMD();

	/** Computes the squared linear time MMD for the current data. This is an
	 * unbiased estimate.
	 *
	 * Note that the underlying streaming feature parser has to be started
	 * before this is called. Otherwise deadlock.
	 *
	 * @return squared linear time MMD
	 */
	virtual float64_t compute_statistic();

	/** Same as compute_statistic(), but with the possibility to perform on
	 * multiple kernels at once
	 *
	 * @param multiple_kernels if true, and underlying kernel is K_COMBINED,
	 * method will be executed on all subkernels on the same data
	 * @return vector of results for subkernels
	 */
	virtual SGVector<float64_t> compute_statistic(bool multiple_kernels);

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

	/** Performs the complete two-sample test on current data and returns a
	 * p-value.
	 *
	 * In case null distribution should be estimated with MMD1_GAUSSIAN,
	 * statistic and p-value are computed in the same loop, which is more
	 * efficient than first computing statistic and then computung p-values.
	 *
	 * In case of bootstrapping, superclass method is called.
	 *
	 * The method for computing the p-value can be set via
	 * set_null_approximation_method().
	 *
	 * @return p-value such that computed statistic is the (1-p) quantile
	 * of the estimated null distribution
	 */
	virtual float64_t perform_test();

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

	/** Computes MMD and a linear time variance estimate.
	 * If multiple_kernels is set to true, each subkernel is evaluated on the
	 * same data.
	 *
	 * @param statistic return parameter for statistic, vector with entry for
	 * each kernel. May be allocated before but doesn not have to be
	 *
	 * @param variance return parameter for statistic, vector with entry for
	 * each kernel. May be allocated before but doesn not have to be
	 *
	 * @param multiple_kernels optional flag, if set to true, it is assumed that
	 * the underlying kernel is of type K_COMBINED. Then, the MMD is computed on
	 * all subkernel separately rather than computing it on the combination.
	 * This is used by kernel selection strategies that need to evaluate
	 * multiple kernels on the same data. Since the linear time MMD works on
	 * streaming data, one cannot simply compute MMD, change kernel since data
	 * would be different for every kernel.
	 */
	virtual void compute_statistic_and_variance(
			SGVector<float64_t>& statistic, SGVector<float64_t>& variance,
			bool multiple_kernels=false);

	/** Same as compute_statistic_and_variance, but computes a linear time
	 * estimate of the covariance of the multiple-kernel-MMD.
	 * See [1] for details.
	 */
	virtual void compute_statistic_and_Q(
			SGVector<float64_t>& statistic, SGMatrix<float64_t>& Q);

	/** Mimics bootstrapping for the linear time MMD. However, samples are not
	 * permutated but constantly streamed and then merged. Usually, this is not
	 * necessary since there is the Gaussian approximation for the null
	 * distribution. However, in certain cases this may fail and sampling the
	 * null distribution might be numerically more stable.
	 * Ovewrite superclass method that merges samples.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> bootstrap_null();

	/** Setter for the blocksize of examples to be processed at once
	 * @param blocksize new blocksize to use
	 */
	void set_blocksize(index_t blocksize) { m_blocksize=blocksize; }

	/** Not implemented for linear time MMD since it uses streaming feautres */
	virtual void set_p_and_q(CFeatures* p_and_q);

	/** Not implemented for linear time MMD since it uses streaming feautres */
	virtual CFeatures* get_p_and_q();

	/** Getter for streaming features of p distribution.
	 * @return streaming features object for p distribution, SG_REF'ed
	 */
	virtual CStreamingFeatures* get_streaming_p();

	/** Getter for streaming features of q distribution.
	 * @return streaming features object for q distribution, SG_REF'ed
	 */
	virtual CStreamingFeatures* get_streaming_q();

	/** returns the statistic type of this test statistic */
	virtual EStatisticType get_statistic_type() const
	{
		return S_LINEAR_TIME_MMD;
	}

	/** @param simulate_h0 if true, samples from p and q will be mixed and
	 * permuted
	 */
	inline void set_simulate_h0(bool simulate_h0) { m_simulate_h0=simulate_h0; }


	virtual const char* get_name() const
	{
		return "LinearTimeMMD";
	}

private:
	void init();

protected:
	/** Streaming feature objects that are used instead of merged samples */
	CStreamingFeatures* m_streaming_p;

	/** Streaming feature objects that are used instead of merged samples*/
	CStreamingFeatures* m_streaming_q;

	/** Number of examples processed at once, i.e. in one burst */
	index_t m_blocksize;

	/** If this is true, samples will be mixed between p and q ind any method
	 * that computes the statistic */
	bool m_simulate_h0;
};

}

#endif /* __LINEARTIMEMMD_H_ */

