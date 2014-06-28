/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012-2013 Heiko Strathmann
 * Written (w) 2014 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#ifndef STREAMING_MMD_H_
#define STREAMING_MMD_H_

#include <shogun/lib/config.h>

#include <shogun/statistics/KernelTwoSampleTest.h>

namespace shogun
{

class CStreamingFeatures;
class CFeatures;
class CKernel;

/**
 * Null variance estimation methods for streaming MMD classes. See
 * CStreamingMMD class documentation for their description
 */
enum ENullVarianceEstimationMethod
{
	WITHIN_BURST_PERMUTATION,
	WITHIN_BLOCK_DIRECT,
	NO_PERMUTATION_DEPRECATED
};

/**
 * Statistic type for streaming MMD classes. See CStreamingMMD class
 * documentation for their description
 */
enum EStreamingStatisticType
{
	S_UNBIASED,
	S_INCOMPLETE,
	S_INCOMPLETE_DEPRECATED
};

/** @brief Abstract base class that provides an interface for performing kernel
 * two-sample test on streaming data using Maximum Mean Discrepancy (MMD) as
 * the test statistic. The MMD is the distance of two probability distributions
 * \f$p\f$ and \f$q\f$ in a RKHS (see [1] for formal description).
 *
 * \f[
 * 	\text{MMD}[\mathcal{F},p,q]^2=\textbf{E}_{x,x'}\left[ k(x,x')\right]-
 * 	2\textbf{E}_{x,y}\left[ k(x,y)\right]
 * 	+\textbf{E}_{y,y'}\left[ k(y,y')\right]=||\mu_p - \mu_q||^2_\mathcal{F}
 * \f]
 *
 * where \f$x,x'\sim p\f$ and \f$y,y'\sim q\f$. The data has to be provided as
 * an instance of CStreamingFeatures, which are processed in blocks for a given
 * blocksize, \f$B\f$. The blocksize determines how many samples are processed
 * at once from both distributions. The number of samples in each block for each
 * distribution depends on the total number of samples from both distribution
 * as well as their proportion in that total number. To be exact, if there are
 * \f$n_x\f$ samples from \f$p\f$, \f$\{x_i\}_{i=1}^{n_x}\sim p\f$ and \f$n_y\f$
 * samples from \f$q\f$, \f$\{y_i\}_{i=1}^{n_y}\sim q\f$, for a given blocksize,
 * \f$B\f$, the number of samples in each block from \f$p\f$, \f$B_x\f$ would
 * be equal to \f$\frac{n_x}{n}B\f$ (similarly, \f$B_y=\frac{n_x}{n}B\f$),
 * where \f$n=n_x+n_y\f$. The blocksize can be specified via set_blocksize()
 * method and then \f$B_x\f$ and \f$B_y\f$ are computed internally.
 *
 * Note that for the sake of simplicity, it is assumed that \f$n\f$ is divisible
 * by \f$B\f$ and \f$B_x\f$ and \f$B_y\f$ are integers. If provided blocksize
 * disagrees with this, an error message is displayed.
 *
 * Streaming data blocks is performed by stream_data_blocks() method. For faster
 * execution, a number of data blocks of size \f$B\f$ are streamed from both the
 * streaming features in each burst which is decided by #m_num_blocks_per_burst.
 * Using \f$s\f$ to denote this, it then returns a merged feature of \f$s\times
 * B_x\f$ samples from \f$p\f$ followed by \f$s\times B_y\f$ samples from
 * \f$q\f$ which are then processed blockwise. The total number of features
 * returned by this method is always equal to \f$s\times B\f$. \f$s\f$ by
 * default is set as 10000 but it's verified and adjusted internally based on
 * how many total samples are there and how many of them are left to be
 * processed. One can always set this number by set_num_blocks_per_burst().
 *
 * If ::PERMUTATION test is desired, stream_data_blocks() optionally shuffles
 * this merged features to accomplish the effect of randomly merging and
 * redistributing samples between these two distributions.
 *
 * This class provides a compute_statistic_and_variance() method for computing
 * statistic estimate \f$\hat{\eta}_k\f$ and variance estimate of the asymptotic
 * distribution of the statistic under null, \f$\hat{\sigma}_{k,0}^2\f$, for
 * multiple kernels at once, where \f$k\f$ is the current kernel. The statistic
 * is computed blockwise as \f$\hat{\eta}_{k,b}\f$, \f$b\f$ being the current
 * block, and finally
 * \f[
 *	\hat{\eta}_k=\frac{B}{n}\sum_{b=1}^{n/B}\hat{\eta}_{k,b}
 * \f]
 * The statistic returned is always \f$\theta_1\times\hat{\eta}_k\f$ where the
 * normalizing constant \f$\theta_1\f$ is computed in subclasses.
 *
 * Presently there are three methods for computing the statistic.
 *
 * - ::S_UNBIASED:
 * The default setup. Within block \f$b\f$, the statistic estimate is computed
 * as
 * \f[
 * 	\hat{\eta}_{k,b}=\frac{1}{B_x(B_x-1)}\sum_{i=1}^{B_x}\sum_{j\neq i}
 * 	k(x_i,x_j)+\frac{1}{B_y(B_y-1)}\sum_{i=1}^{B_y}\sum_{j\neq i}k(y_i,y_j)
 * 	-\frac{2}{B_xB_y}\sum_{i=1}^{B_x}\sum_{j=1}^{B_y}k(x_i,y_j)
 * \f]
 *
 * - ::S_INCOMPLETE:
 * Only applicable when \f$n_x=n_y=\frac{n}{2}\f$. Within block \f$b\f$,
 * the statistic estimate is computed as
 * \f[
 *	\hat{\eta}_{k,b^-}=\frac{1}{\frac{B}{2}(\frac{B}{2}-1)}\sum_{i\neq j}
 *	h(z_i^{(b)},z_j^{(b)})
 * \f]
 * where for each pair \f$z=(x,y)\f$, \f$h(z,z')=k(x,x')+k(y,y')-k(x,y')-
 * k(x',y)\f$.
 *
 * - ::S_INCOMPLETE_DEPRECATED:
 * This is similar to ::S_INCOMPLETE except that the normalizing constant
 * for statistic \f$\theta_1\f$ is 1 (See CLinearTimeMMD or CBTestMMD).
 *
 * The statistic type can be specified via set_statistic_type() method
 *
 * Variance estimate is also computed blockwise via two separate methods -
 *
 * - ::WITHIN_BURST_PERMUTATION:
 * In this approach, the samples within current burst are randomly split in the
 * same proportions and a statistic estimate \f$\hat{\eta}_{k,b}^*\f$ is
 * computed blockwise. The final variance estimate is computed as
 * \f[
 * 	\hat{\sigma}_{k,0}^2=\theta_2\times\text{var}\left[\{\hat{\eta}_{k,b}^*\}
 *	_{b=1}^{n/B} \right ]
 * \f]
 * where the normalizing constant \f$\theta_2\f$ is computed in the subclasses.
 *
 * - ::WITHIN_BLOCK_DIRECT:
 * A blockwise variance estimate is computed as
 * \f[
 * 	\left(\hat{\sigma}_{k,0}^2 \right )^{(b)}=\frac{2}{B(B-3)}\left[
 * 	\left(\bar{\mathbf{K}}^{(b)}\circ\bar{\mathbf{K}}^{(b)}\right )_{++}+
 * 	\frac{\left( \bar{\mathbf{K}}^{(b)}_{++}\right )^2}{(B-1)(B-2)}-
 * 	\frac{2}{B-2}\left((\bar{\mathbf{K}}^{(b)})^2 \right )_{++} \right ]
 * \f]
 * where \f$\bar{\mathbf{K}}^{(b)}=\mathbf{K}^{(b)}-\text{diag}(\mathbf{K}
 * ^{(b)})\f$ and \f$\mathbf{K}^{(b)}\f$ is the Gram matrix on of samples
 * from \f$p\f$ within block \f$b\f$, \f$A\circ A\f$ denotes element-wise
 * square operation and \f$A_{++}\f$ denotes the num of all elements for
 * matrix \f$A\f$. Finally, the variance estimate is computed as
 * \f[
 * 	\hat{\sigma}_{k,0}^2=\frac{B}{n}\sum_{b=1}^{n/B}\left(\hat{\sigma}_{k,0}^2
 *	\right )^{(b)}
 * \f]
 *
 * - ::NO_PERMUTATION_DEPRECATED:
 * This is an incorrest estimation of the variance under null. If specified,
 * this just computes a runtime variance based on the statistic computed for
 * each burst without permuting the samples.
 *
 * Use set_null_var_est_method() to set the variance estimation method. The
 * default is ::WITHIN_BURST_PERMUTATION. Both these methods take
 * \f$\mathcal{O}(B^2)\f$
 *
 * Blockwise computation of kernel functions for statistic/variance is done via
 *
 * - compute_blockwise_statistic_variance() method: used with
 * ::WITHIN_BLOCK_DIRECT variance estimation which returns a matrix of
 * statistic estimate \f$\hat{\eta}_{k,b}\f$ in its first column and variance
 * estimate \f$\left(\hat{\sigma}_{k,0}^2\right )^{(b)}\f$ in second column.
 *
 * - compute_blockwise_statistic() method : used with ::WITHIN_BURST_PERMUTATION
 * variance estimation which returns a vector of values \f$\hat{\eta}_{k,b}\f$,
 * one entry per block in the current burst
 *
 * For computing mean of statistic and variance, a linear time online algorithm
 * by D. Knuth (see wikipedia) is used. Computing covariance matrix (Q) for
 * multiple kernels is also possible following the approach described in [2].
 *
 * Along with the statistic comes a method to compute a p-value based on a
 * Gaussian approximation of the null-distribution which is possible in
 * linear time and constant space. Sampling from null is also possible (no
 * permutations but new examples will be used here).
 * If unsure which one to use, sampling with 250 iterations always is
 * correct (but slow). When the sample size is large (>1000) at least,
 * the Gaussian approximation is an accurate and much faster choice.
 *
 * To choose, use set_null_approximation_method() and choose from
 *
 * - ::MMD1_GAUSSIAN: Approximates the null-distribution with a Gaussian. Only use
 * from at least 1000 samples. If using, check if type I error equals the
 * desired value.
 *
 * - ::MMD1_GAUSSIAN_DEPRECATED: Used with ::S_INCOMPLETE_DEPRECATED statistic type.
 * The Gaussian approximation of the null-distribution uses variance
 * estimation computed by ::NO_PERMUTATION_DEPRECATED approximation method.
 *
 * - ::PERMUTATION: For permuting available samples to sample null-distribution.
 *
 * For kernel selection see CMMDKernelSelection.
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., &
 * Smola, A. (2012). A Kernel Two-Sample Test. Journal of Machine Learning
 * Research, 13, 671-721.
 * [2]: Gretton, A., B. Sriperumbudur, D. Sejdinovic, H, Strathmann,
 * S. Balakrishnan, M. Pontil, K. Fukumizu: Optimal kernel choice for large-
 * -scale two-sample tests. NIPS 2012.
 *
 * Please note that \f$n_x\f$, \f$n_x\f$, \f$B\f$, \f$B_x\f$, \f$B_y\f$, \f$s\f$
 * are denoted as #m_m, #m_n, #m_blocksize, #m_blocksize_p, #m_blocksize_q and
 * #m_num_blocks_per_burst, respectively, in the implementation.
 */
class CStreamingMMD: public CKernelTwoSampleTest
{
public:
	/** Default constructor */
	CStreamingMMD();

	/**
	 * Constructor.
	 *
	 * @param kernel kernel to use
	 * @param p streaming features p to use
	 * @param q streaming features q to use
	 * @param m number of samples from first distribution, p
	 * @param n number of samples from second distribution, q
	 */
	CStreamingMMD(CKernel* kernel, CStreamingFeatures* p, CStreamingFeatures* q,
			index_t m, index_t n);

	/** Destructor */
	virtual ~CStreamingMMD();

	/** Computes the squared MMD for the current data. This is an unbiased
	 * estimate.
	 *
	 * Note that the underlying streaming feature parser has to be started
	 * before this is called. Otherwise deadlock.
	 *
	 * @return squared MMD
	 */
	virtual float64_t compute_statistic();

	/** Same as compute_statistic(), but with the possibility to perform on
	 * multiple kernels at once
	 *
	 * @param multiple_kernels if true, and underlying kernel is ::K_COMBINED,
	 * method will be executed on all subkernels on the same data
	 * @return vector of results for subkernels
	 */
	SGVector<float64_t> compute_statistic(bool multiple_kernels);

	/** Computes a p-value based on current method for approximating the
	 * null-distribution. The p-value is the 1-p quantile of the null-
	 * distribution where the given statistic lies in.
	 *
	 * The method for computing the p-value can be set via
	 * set_null_approximation_method().
	 * Since the null-distribution is normal, a Gaussian approximation
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
	 * In case null distribution should be estimated with ::MMD1_GAUSSIAN,
	 * statistic and p-value are computed in the same loop, which is more
	 * efficient than first computing statistic and then computung p-values.
	 *
	 * In case of sampling null, superclass method is called.
	 *
	 * The method for computing the p-value can be set via
	 * set_null_approximation_method().
	 *
	 * @return p-value such that computed statistic is the (1-p) quantile
	 * of the estimated null distribution
	 */
	virtual float64_t perform_test();

	/** Computes a threshold based on current method for approximating the
	 * null-distribution. The threshold is the value that a statistic has
	 * to have in order to reject the null-hypothesis.
	 *
	 * The method for computing the p-value can be set via
	 * set_null_approximation_method().
	 * Since the null-distribution is normal, a Gaussian approximation
	 * is available.
	 *
	 * @param alpha test level to reject null-hypothesis
	 * @return threshold for statistics to reject null-hypothesis
	 */
	virtual float64_t compute_threshold(float64_t alpha);

	/** Computes a linear time estimate of the variance of the squared mmd,
	 * which may be used for an approximation of the null-distribution
	 * The value is the variance of the vector of which the MMD is the mean.
	 *
	 * @return variance estimate
	 */
	float64_t compute_variance_estimate();

	/** Computes MMD and a linear time variance estimate. If multiple_kernels
	 * is set to true, each subkernel is evaluated on the same data.
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
	void compute_statistic_and_variance(SGVector<float64_t>& statistic,
			SGVector<float64_t>& variance, bool multiple_kernels=false);

	/** Same as compute_statistic_and_variance, but computes a linear time
	 * estimate of the covariance of the multiple-kernel-MMD.
	 * See [2] for details.
	 */
	void compute_statistic_and_Q(SGVector<float64_t>& statistic,
			SGMatrix<float64_t>& Q);

	/** Mimics sampling null for MMD. However, samples are not permutated but
	 * constantly streamed and then merged. Usually, this is not necessary
	 * since there is the Gaussian approximation for the null distribution.
	 * However, in certain cases this may fail and sampling the null
	 * distribution might be numerically more stable. Ovewrite superclass
	 * method that merges samples.
	 *
	 * @return vector of all null samples
	 */
	virtual SGVector<float64_t> sample_null();

	/** Setter for the blocksize of examples to be processed at once. This
	 * method internally computes the number of samples to be processed from
	 * each of the distribution at once, blocksize_p and blocksize_q.
	 * (see class documentation for details). This method internally sets
	 * the number of blocks to be streamed in one burst as well and verifies
	 * whether it is valid (see set_num_blocks_per_burst() documentation)
	 *
	 * @param blocksize new blocksize to use
	 */
	void set_blocksize(index_t blocksize);

	/** Setter for the number of blocks to be streamed at once. This method
	 * internally verifies the number provided. For example, if total number
	 * of samples are 1000 from p and 1200 from q and the corresponding
	 * blocksizes are 10 and 12, at max 100 such blocks can be streamed. If
	 * given num_blocks_per_burst is greater than this, it sets
	 * num_blocks_per_burst as maximum value possible (i.e. 100 in this case).
	 * Can only be used once the blocksize is set.
	 *
	 * @param num_blocks_per_burst number of blocks to be streamed at once
	 */
	void set_num_blocks_per_burst(index_t num_blocks_per_burst);

	/** @return blocksize being used */
	index_t get_blocksize();

	/** @return the number of blocks to be streamed at once */
	index_t get_num_blocks_per_burst();

	/** Not implemented for streaming MMD since it uses streaming feautres */
	virtual void set_p_and_q(CFeatures* p_and_q);

	/** Not implemented for streaming MMD since it uses streaming feautres */
	virtual CFeatures* get_p_and_q();

	/** Setter for streaming features of p distribution.
	 * @param p streaming features object for p distribution
	 */
	void set_streaming_p(CStreamingFeatures* p);

	/** Setter for streaming features of q distribution.
	 * @param q streaming features object for q distribution
	 */
	void set_streaming_q(CStreamingFeatures* q);

	/** Getter for streaming features of p distribution.
	 * @return streaming features object for p distribution, SG_REF'ed
	 */
	CStreamingFeatures* get_streaming_p();

	/** Getter for streaming features of q distribution.
	 * @return streaming features object for q distribution, SG_REF'ed
	 */
	CStreamingFeatures* get_streaming_q();

	/** @param simulate_h0 if true, samples from p and q will be mixed and
	 * permuted
	 */
	inline void set_simulate_h0(bool simulate_h0)
	{
		m_simulate_h0=simulate_h0;
	}

	/** @param statistic_type type of statistic estimation */
	void set_statistic_type(EStreamingStatisticType statistic_type);

	/** @return current method of statistic estimation */
	EStreamingStatisticType get_statistic_type();

	/** @param null_var_est_method estimation method for variance under null */
	void set_null_var_est_method(ENullVarianceEstimationMethod
			null_var_est_method);

	/** @return estimation method for variance under null */
	ENullVarianceEstimationMethod get_null_var_est_method();

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "StreamingMMD";
	}

protected:
	/**
	 * Streams #m_num_blocks_per_burst blocks of data from each distribution
	 * with blocks of size #m_blocksize. If #m_simulate_h0 is set, it shuffles
	 * the samples (See class description).
	 *
	 * @return merged features for samples from both distribution with
	 * #m_num_blocks_per_burst*#m_blocksize_p samples from p followed by
	 * #m_num_blocks_per_burst*#m_blocksize_q samples from q.
	 */
	CFeatures* stream_data_blocks();

#ifdef HAVE_EIGEN3
	/**
	 * Method that computes blockwise statistic and variance estimate.
	 * Used with ::WITHIN_BLOCK_DIRECT null-var estimation method
	 *
	 * @param kernel the kernel to be used for computing MMD. This will be
	 * useful when multiple kernels are used
	 * @param p_and_q_current_burst the merged features within current burst,
	 * with #m_num_blocks_per_burst*#m_blocksize_p samples from p followed by
	 * #m_num_blocks_per_burst*#m_blocksize_q samples from q.
	 * @return a matrix of #m_num_blocks_per_burst rows with two entries in
	 * each column for each block, first entry is the statistic estimate in
	 * current block is \f$\hat{\eta}_{k,b}\f$, and second entry is the
	 * variance estimate \f$\left(\hat{\sigma}_{k,0}^2\right )^{(b)}\f$ for
	 * within-block direct estimation
	 */
	SGMatrix<float64_t> compute_blockwise_statistic_variance(CKernel*
			kernel, CFeatures* p_and_q_current_burst);
#endif // HAVE_EIGEN3

	/**
	 * Method that computes blockwise statistic estimate.
	 * Used with ::WITHIN_BURST_PERMUTATION null-var estimation method
	 *
	 * @param kernel the kernel to be used for computing MMD. This will be
	 * useful when multiple kernels are used
	 * @param p_and_q_current_burst the merged features within current burst,
	 * with m_num_blocks_per_burst*m_blocksize_p samples from p followed by
	 * m_num_blocks_per_burst*m_blocksize_q samples from q.
	 * @return a vector of statistic estimates for each block in the current
	 * burst, \f$\hat{\eta}_{k,b}\f$
	 */
	SGVector<float64_t> compute_blockwise_statistic(CKernel* kernel,
			CFeatures* p_and_q_current_burst);

	/** Abstract method that computes normalizing constant \f$theta_1\f$ for
	 * statistic estimate. This varies in the subclasses
	 *
	 * @return normalizing constant \f$theta_1\f$ for statistic estimate for
	 * computing p-value/threshold
	 */
	virtual float64_t compute_statistic_normalizing_constant()=0;

	/** Abstract method that computes normalizing constant \f$\theta_2\f$ for
	 * variance estimate under null for within burst permuation approach. This
	 * varies in the subclasses.
	 *
	 * @return normalizing constant \f$theta_2\f$ for variance estimate of the
	 * statistic under null when using within-burst permutation method
	 */
	virtual float64_t compute_variance_normalizing_constant()=0;

	/** Abstract method that computes the variance for Gaussian approximation of
	 * asymptotic distribution of the test-statistic under null
	 *
	 * @param variance the variance under null
	 * @return the variance for Gaussian approximation of asymptotic
	 * distribution
	 */
	virtual float64_t compute_gaussian_variance(float64_t variance)=0;

	/** Streaming feature objects that are used instead of merged samples */
	CStreamingFeatures* m_streaming_p;

	/** Streaming feature objects that are used instead of merged samples*/
	CStreamingFeatures* m_streaming_q;

	/** number of samples from the second distribution, q */
	index_t m_n;

	/** Number of examples processed at once, i.e. in one burst */
	index_t m_blocksize;

	/** Number of samples from p processed at once, i.e. in one burst */
	index_t m_blocksize_p;

	/** Number of samples from q processed at once, i.e. in one burst */
	index_t m_blocksize_q;

	/** Number of blocks to be streamed at one burst Use higher number for
	 * faster execution. Default is 10000 */
	index_t m_num_blocks_per_burst;

	/** If this is true, samples will be mixed between p and q in any method
	 * that computes the statistic */
	bool m_simulate_h0;

	/** statistic estimation type */
	EStreamingStatisticType m_statistic_type;

	/** Null variance estimation method */
	ENullVarianceEstimationMethod m_null_var_est_method;

private:
	/** register parameters and initialize with defaults */
	void init();
};

}

#endif /* STREAMING_MMD_H_ */
