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

#include <shogun/statistics/KernelTwoSampleTest.h>

namespace shogun
{

class CStreamingFeatures;
class CFeatures;

/** @brief Abstract base class that provides an interface for performing kernel
 * two-sample test on streaming data using Maximum Mean Discrepancy (MMD) as
 * the test statistic. The MMD is the distance of two probability distributions
 * \f$p\f$ and \f$q\f$ in a RKHS (see [1] for formal description).
 *
 * \f[
 * \text{MMD}[\mathcal{F},p,q]^2=\textbf{E}_{x,x'}\left[ k(x,x')\right]-
 * 2\textbf{E}_{x,y}\left[ k(x,y)\right]
 * +\textbf{E}_{y,y'}\left[ k(y,y')\right]=||\mu_p - \mu_q||^2_\mathcal{F}
 * \f]
 *
 * where \f$x,x'\sim p\f$ and \f$y,y'\sim q\f$. The data has to be provided as
 * streaming features, which are processed in blocks for a given blocksize.
 * The blocksize determines how many examples are processed at once. A method
 * for getting a specified number of blocks of data is provided which can
 * optionally merge and permute the data within the current burst. The exact
 * computation of kernel functions for MMD computation is abstract and has to
 * be defined by its subclasses, which should return a vector of function
 * values. Please note that for streaming MMD, the number of data points from
 * both the distributions has to be equal.
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
 * MMD1_GAUSSIAN: Approximates the null-distribution with a Gaussian. Only use
 * from at least 1000 samples. If using, check if type I error equals the
 * desired value.
 *
 * PERMUTATION: For permuting available samples to sample null-distribution.
 *
 * For kernel selection see CMMDKernelSelection.
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., &
 * Smola, A. (2012). A Kernel Two-Sample Test. Journal of Machine Learning
 * Research, 13, 671-721.
 */
class CStreamingMMD: public CKernelTwoSampleTest
{
public:
	/** default constructor */
	CStreamingMMD();

	/** constructor.
	 *
	 * @param kernel kernel to use
	 * @param p streaming features p to use
	 * @param q streaming features q to use
	 * @param m number of samples from each distribution
	 * @param blocksize size of examples that are processed at once when
	 * computing statistic/threshold.
	 */
	CStreamingMMD(CKernel* kernel, CStreamingFeatures* p,
			CStreamingFeatures* q, index_t m, index_t blocksize=10000);

	/** destructor */
	virtual ~CStreamingMMD();

	/** Computes the squared MMD for the current data. This is an unbiased
	 * estimate. This method relies on compute_statistic_and_variance which
	 * has to be defined in the subclasses
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
	 * In case of sampling null, superclass method is called.
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

	/** computes a linear time estimate of the variance of the squared mmd,
	 * which may be used for an approximation of the null-distribution
	 * The value is the variance of the vector of which the MMD is the mean.
	 *
	 * @return variance estimate
	 */
	virtual float64_t compute_variance_estimate();

	/** Abstract method that computes MMD and a linear time variance estimate.
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
			bool multiple_kernels=false)=0;

	/** Same as compute_statistic_and_variance, but computes a linear time
	 * estimate of the covariance of the multiple-kernel-MMD.
	 * See [1] for details.
	 */
	virtual void compute_statistic_and_Q(
			SGVector<float64_t>& statistic, SGMatrix<float64_t>& Q)=0;

	/** Mimics sampling null for MMD. However, samples are not permutated but
	 * constantly streamed and then merged. Usually, this is not necessary
	 * since there is the Gaussian approximation for the null distribution.
	 * However, in certain cases this may fail and sampling the null
	 * distribution might be numerically more stable. Ovewrite superclass
	 * method that merges samples.
	 *
	 * @return vector of all statistics
	 */
	virtual SGVector<float64_t> sample_null();

	/** Setter for the blocksize of examples to be processed at once
	 * @param blocksize new blocksize to use
	 */
	void set_blocksize(index_t blocksize)
	{
		m_blocksize=blocksize;
	}

	/** Streams num_blocks data from each distribution with blocks of size
	 * num_this_run. If m_simulate_h0 is set, it merges the blocks together,
	 * shuffles and redistributes between the blocks.
	 *
	 * @param num_blocks number of blocks to be streamed from each distribution
	 * @param num_this_run number of data points to be streamed for one block
	 * @return an ordered list of blocks of data. The order in the
	 * list is \f$x,x',\cdots\sim p\f$ followed by \f$y,y',\cdots\sim q\f$.
	 * The features inside the list are SG_REF'ed and delete_data is set in the
	 * list, which will destroy the at CList's destructor call
	 */
	CList* stream_data_blocks(index_t num_blocks, index_t num_this_run);

	/** Not implemented for streaming MMD since it uses streaming feautres */
	virtual void set_p_and_q(CFeatures* p_and_q);

	/** Not implemented for streaming MMD since it uses streaming feautres */
	virtual CFeatures* get_p_and_q();

	/** Getter for streaming features of p distribution.
	 * @return streaming features object for p distribution, SG_REF'ed
	 */
	virtual CStreamingFeatures* get_streaming_p();

	/** Getter for streaming features of q distribution.
	 * @return streaming features object for q distribution, SG_REF'ed
	 */
	virtual CStreamingFeatures* get_streaming_q();

	/** @param simulate_h0 if true, samples from p and q will be mixed and
	 * permuted
	 */
	inline void set_simulate_h0(bool simulate_h0)
	{
		m_simulate_h0=simulate_h0;
	}

	/** @return the class name */
	virtual const char* get_name() const
	{
		return "StreamingMMD";
	}

protected:
	/** abstract method that computes the squared MMD
	 *
	 * @param kernel the kernel to be used for computing MMD. This will be
	 * useful when multiple kernels are used
	 * @param data the list of data on which kernels are computed. The order
	 * of data in the list is \f$x,x',\cdots\sim p\f$ followed by
	 * \f$y,y',\cdots\sim q\f$. It is assumed that detele_data flag is set
	 * inside the list
	 * @param num_this_run number of data points in current blocks
	 * @return the MMD values
	 */
	virtual SGVector<float64_t> compute_squared_mmd(CKernel* kernel,
			CList* data, index_t num_this_run)=0;

	/** Streaming feature objects that are used instead of merged samples */
	CStreamingFeatures* m_streaming_p;

	/** Streaming feature objects that are used instead of merged samples*/
	CStreamingFeatures* m_streaming_q;

	/** Number of examples processed at once, i.e. in one burst */
	index_t m_blocksize;

	/** If this is true, samples will be mixed between p and q in any method
	 * that computes the statistic */
	bool m_simulate_h0;

private:
	/** register parameters and initialize with defaults */
	void init();
};

}

#endif /* STREAMING_MMD_H_ */

