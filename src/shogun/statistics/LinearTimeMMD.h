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
 * from at least 1000 samples.
 *
 * BOOTSTRAPPING: For permuting available samples to sample null-distribution
 *
 * Comes with a method for selecting kernel weights, if a combined kernel on
 * combined features is used. See optimize_kernel_weights(). See [2]
 *
 * A very basic method for kernel selection when using CGaussianKernel is to
 * use the median distance of the underlying data. See examples how to do that.
 * More advanced methods will follow in the near future. However, the median
 * heuristic works in quite some cases. See [1].
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 *
 * [2]: Gretton, A., Sriperumbudur, B., Sejdinovic, D., Strathmann, H.,
 * Balakrishnan, S., Pontil, M., & Fukumizu, K. (2012).
 * Optimal kernel choice for large-scale two-sample tests.
 * Advances in Neural Information Processing Systems.
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

	/** Computes MMD and a linear time variance estimate using an in-place
	 * method.
	 *
	 * @param statistic return parameter for statistic
	 * @param variance return parameter for variance
	 */
	virtual void compute_statistic_and_variance(float64_t& statistic,
			float64_t& variance);

	virtual SGVector<float64_t> compute_h_terms();

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

#ifdef HAVE_LAPACK
	/** Selects optimal kernel weights (if the underlying kernel and features
	 * are combined ones) using the ratio of the squared MMD by its standard
	 * deviation as a criterion, i.e.
	 * \f[
	 * \frac{\text{MMD}_l^2[\mathcal{F},X,Y]}{\sigma_l}
	 * \f]
	 * where both expressions are estimated in linear time.
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

	virtual const char* get_name() const
	{
		return "LinearTimeMMD";
	}

private:
	void init();

#ifdef HAVE_LAPACK
	/** return pointer to i-th column of m_Q. Helper for libqp */
	static const float64_t* get_Q_col(uint32_t i);

	/** helper functions that prints current state */
	static void print_state(libqp_state_T state);
#endif //HAVE_LAPACK

protected:
	/** Streaming feature objects that are used instead of merged samples */
	CStreamingFeatures* m_streaming_p;

	/** Streaming feature objects that are used instead of merged samples*/
	CStreamingFeatures* m_streaming_q;

	/** Number of examples processed at once, i.e. in one burst */
	index_t m_blocksize;

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

