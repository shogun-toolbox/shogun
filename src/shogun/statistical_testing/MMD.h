/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2017 Soumyajit De
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

#ifndef MMD_H_
#define MMD_H_

#include <utility>
#include <memory>
#include <functional>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/TwoSampleTest.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

class Kernel;
class KernelSelectionStrategy;
template <typename> class SGVector;

/** @brief Abstract base class that provides an interface for performing kernel
 * two-sample test using Maximum Mean Discrepancy (MMD) as the test statistic.
 * The MMD is the distance of two probability distributions \f$p\f$ and \f$q\f$
 * in a RKHS (see [1] for formal description).
 *
 * \f[
 * \text{MMD}[\mathcal{F},p,q]^2=||\mu_p - \mu_q||^2_\mathcal{F}=
 * \textbf{E}_{x,x'}\left[ k(x,x')\right]
 * -2\textbf{E}_{x,y}\left[ k(x,y)\right]
 * +\textbf{E}_{y,y'}\left[ k(y,y')\right]
 * \f]
 *
 * where \f$x,x'\sim p\f$ and \f$y,y'\sim q\f$.
 *
 * Given two sets of samples \f$\{x_i\}_{i=1}^{n_x}\sim p\f$ and
 * \f$\{y_i\}_{i=1}^{n_y}\sim q\f$, \f$n_x+n_y=n\f$,
 * the unbiased estimate of the above statistic is computed as
 * \f[
 * 	\hat{\eta}_{k,U}=\frac{1}{n_x(n_x-1)}\sum_{i=1}^{n_x}\sum_{j\neq i}
 * 	k(x_i,x_j)+\frac{1}{n_y(n_y-1)}\sum_{i=1}^{n_y}\sum_{j\neq i}k(y_i,y_j)
 * 	-\frac{2}{n_xn_y}\sum_{i=1}^{n_x}\sum_{j=1}^{n_y}k(x_i,y_j)
 * \f]
 *
 * A biased version is
 * \f[
 * 	\hat{\eta}_{k,V}=\frac{1}{n_x^2}\sum_{i=1}^{n_x}\sum_{j=1}^{n_x}
 * 	k(x_i,x_j)+\frac{1}{n_y^2}\sum_{i=1}^{n_y}\sum_{j=1}^{n_y}k(y_i,y_j)
 * 	-\frac{2}{n_xn_y}\sum_{i=1}^{n_x}\sum_{j=1}^{n_y}k(x_i,y_j)
 * \f]KernelSelectionStrategy
 *
 * When \f$n_x=n_y=\frac{n}{2}\f$, an incomplete version can also be computed
 * as the following
 * \f[
 * 	\hat{\eta}_{k,U^-}=\frac{1}{\frac{n}{2}(\frac{n}{2}-1)}\sum_{i\neq j}
 * 	h(z_i,z_j)
 * \f]
 * where for each pair \f$z=(x,y)\f$, \f$h(z,z')=k(x,x')+k(y,y')-k(x,y')-
 * k(x',y)\f$.
 *
 * The type (biased/unbiased/incomplete) can be selected via set_statistic_type()
 * via the enum values from EStatisticType, ST_BIASED, ST_UNBIASED and ST_INCOMPLETE,
 * respectively. The estimate returned by compute_statistic()
 * is \f$\frac{n_xn_y}{n_x+n_y}\hat{\eta}_k\f$.
 *
 * This class provides an interface for adding multiple kernels and then
 * selecting the best kernel based on specified strategies. To know more in details
 * about various learning algorithms for optimal kernel selection, please refer to [2].
 *
 * Along with the statistic comes a method to compute a p-value based on
 * different methods. Permutation test is possible. If unsure which one to
 * use, sampling with 250 permutation iterations usually always is correct.
 *
 * To choose, use set_null_approximation_method() and choose from.
 *
 * NAM_MMD2_SPECTRUM: For a fast, consistent test based on the spectrum of
 * the kernel matrix, as described in [2]. Only supported if Eigen3 is installed.
 * Only applicable for QuadraticTimeMMD.
 *
 * NAM_MMD2_GAMMA: for a very fast, but not consistent test based on moment matching
 * of a Gamma distribution, as described in [2].
 * Only applicable for QuadraticTimeMMD.
 *
 * NAM_PERMUTATION: For permuting available samples to sample null-distribution
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., &
 * Smola, A. (2012). A Kernel Two-Sample Test. Journal of Machine Learning
 * Research, 13, 671-721.
 *
 * [2] Arthur Gretton, Bharath K. Sriperumbudur, Dino Sejdinovic, Heiko Strathmann,
 * Sivaraman Balakrishnan, Massimiliano Pontil, Kenji Fukumizu: Optimal kernel choice
 * for large-scale two-sample tests. NIPS 2012: 1214-1222.
 */
class MMD : public RandomMixin<TwoSampleTest>
{
public:
	/** Default constructor */
	MMD();

	/** Destructor */
	~MMD() override;

	/**
	 * Method that sets the specific kernel selection strategy based on the
	 * specific parameters provided. Please see class documentation for details.
	 * Use this method for every other strategy other than KSM_CROSS_VALIDATION.
	 *
	 * @param method The kernel selection method as specified in EKernelSelectionMethod.
	 * @param weighted If true, then an weighted combination of the kernel is used after
	 * solving an optimization. If false, only a single kernel is selected among the
	 * provided ones.
	 */
	void set_kernel_selection_strategy(EKernelSelectionMethod method, bool weighted = false);

	/**
	 * Method that sets the specific kernel selection strategy based on the
	 * specific parameters provided. Please see class documentation for details.
	 * Use this method for KSM_CROSS_VALIDATION.
	 *
	 * @param method The kernel selection method as specified in EKernelSelectionMethod.
	 * @param num_runs The number of total runs of the cross-validation algorithm.
	 * @param num_folds The number of folds (k) to be used in k-fold stratified cross-validation.
	 * @param alpha The threshold to be used while performing test for the test-folds.
	 */
	void set_kernel_selection_strategy(EKernelSelectionMethod method, index_t num_runs, index_t num_folds, float64_t alpha);

	/**
	 * Method that adds a kernel instance to be used for kernel selection. Please
	 * note that the kernels added by this method are NOT set as the main test kernel
	 * unless select_kernel() method is executed.
	 *
	 * This method is NOT thread safe. Please DO NOT use this method from multiple threads.
	 *
	 * @param kernel One of the kernel instances with which learning algorithm will work.
	 */
	void add_kernel(std::shared_ptr<Kernel> kernel);

	/**
	 * Method that selects/learns the kernel based on the defined kernel selection strategy.
	 * If no explicit kernel selection strategy was set using set_kernel_selection_strategy()
	 * method, then a default strategy is used. Please see EKernelSelectionMethod for the
	 * default strategy.
	 *
	 * This method is NOT thread safe. It replaces the internel kernel set by set_kernel()
	 * method, if there was any. Please DO NOT use this method from multiple threads.
	 *
	 * The learned/selected kernel can be obtained from a subsequent get_kernel() call.
	 *
	 * This method expects train-test mode to be turned on at the time of invocation. Please
	 * see the class documentation of HypothesisTest.
	 */
	virtual void select_kernel();

	/**
	 * Method that returns the kernel selection strategy wrapper object that will be/
	 * was used in the last kernel learning algorithm. Use this method when results of
	 * intermediate steps taken by the kernel selection algorithms are of interest.
	 *
	 * @return The internal instance of KernelSelectionStrategy that holds intermediate
	 * measures computed at the time of the last kernel selection algorithm invocation.
	 */
	KernelSelectionStrategy const* get_kernel_selection_strategy() const;

	/**
	 * Interface for computing the test-statistic for the hypothesis test.
	 *
	 * @return test statistic for the given data/parameters/methods
	 */
	float64_t compute_statistic() override = 0;

	/**
	 * Interface for computing the samples under the null-hypothesis.
	 *
	 * @return vector of all statistics
	 */
	SGVector<float64_t> sample_null() override = 0;

	/** Method that releases the pre-computed kernel that is used in the computation. */
	void cleanup();

	/**
	 * Method that sets the number of null-samples used for computing p-value.
	 *
	 * @param null_samples Number of null-samples.
	 */
	void set_num_null_samples(index_t null_samples);

	/** @return Number of null-samples */
	index_t get_num_null_samples() const;

	/**
	 * Method that sets the type of the estimator for MMD^2
	 *
	 * @param stype The type of the estimator for MMD^2
	 */
	void set_statistic_type(EStatisticType stype);

	/** @return The type of the estimator for MMD^2 */
	EStatisticType get_statistic_type() const;

	/**
	 * Method that sets the approach to be taken while approximating the null-samples.
	 *
	 * @nmethod The null-approximation method
	 */
	void set_null_approximation_method(ENullApproximationMethod nmethod);

	/** @return The null-approximation method */
	ENullApproximationMethod get_null_approximation_method() const;

	/** @return The name of this class */
	const char* get_name() const override;
protected:
	virtual float64_t normalize_statistic(float64_t statistic) const = 0;
private:
	struct Self;
	std::unique_ptr<Self> self;
	void init();
};

}
#endif // MMD_H_
