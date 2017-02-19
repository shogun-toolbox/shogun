/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2016 Soumyajit De
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

#ifndef QUADRATIC_TIME_MMD_H_
#define QUADRATIC_TIME_MMD_H_

#include <memory>
#include <shogun/statistical_testing/MMD.h>

namespace shogun
{

class CMultiKernelQuadraticTimeMMD;
template <typename> class SGVector;

/**
 * @brief This class implements the quadratic time Maximum Mean Statistic as
 * described in [1].
 * The MMD is the distance of two probability distributions \f$p\f$ and \f$q\f$
 * in a RKHS which we denote by
 * \f[
 * 	\hat{\eta_k}=\text{MMD}[\mathcal{F},p,q]^2=\textbf{E}_{x,x'}
 * 	\left[ k(x,x')\right]-2\textbf{E}_{x,y}\left[ k(x,y)\right]
 * 	+\textbf{E}_{y,y'}\left[ k(y,y')\right]=||\mu_p - \mu_q||^2_\mathcal{F}
 * \f]
 *
 * Estimating variance of the asymptotic distribution of the statistic under
 * null and alternative hypothesis can be done using compute_variance_h0() and
 * compute_variance_h1() method.
 *
 * Note that all these operations can be done for multiple kernels
 * at once as well. To use this functionality, use multikernel() method to
 * obtain a CMultiKernelQuadraticTimeMMD instance and then call methods on that.
 *
 * If you do not know about your data, but want to use the MMD from a kernel
 * matrix, just use the custom kernel constructor and initialize the features as
 * CDummyFeatures. Everything else will work as usual.
 *
 * To make the computation faster, this class always pre-computes the kernel
 * and stores the Gram matrix using merged samples from p and q. It essentially
 * keeps a backup of the old kernel and rather uses this pre-computed one as
 * long as the present kernel is valid. Therefore, after a computation phase
 * is executed, upon calling get_kernel() we will obtain the pre-computed
 * kernel matrix as a CCustomKernel object. However, if subsequently the
 * features are updated or the underlying kernel itself is updated, it discards
 * the pre-computed kernel matrix (frees memory) and pulls the old kernel from
 * backup (or, simply replace that if a new kernel is provided) and then
 * pre-computes that in the next run.
 *
 * It is possible to turn off the above feature by turning it off. However,
 * it will affect the performance of the algorithms, since they are optimzied
 * for pre-computed kernel matrices. Therefore, this should only be turned off
 * if the storage of the kernel is a major concern. Please note that only
 * the lower triangular part of the Gram matrix is stored, in order to exploit
 * the symmetry.
 *
 * Since the methods modifies the object's state, using the methods of this
 * class from multiple threads may result in undesired/incorrect results/behavior.
 *
 * NOTE: \f$n_x\f$ and \f$n_y\f$ are represented by \f$m\f$ and \f$n\f$,
 * respectively in the implementation.
 *
 * [1]: Gretton, A., Borgwardt, K. M., Rasch, M. J., Schoelkopf, B., & Smola, A. (2012).
 * A Kernel Two-Sample Test. Journal of Machine Learning Research, 13, 671-721.
 *
 * [2]: Gretton, A., Fukumizu, K., & Harchaoui, Z. (2011).
 * A fast, consistent kernel two-sample test.
 */
class CQuadraticTimeMMD : public CMMD
{
	friend class CMultiKernelQuadraticTimeMMD;

public:
	/** Default constructor */
	CQuadraticTimeMMD();

	/**
	 * Convenience constructor. Initializes the features representing samples
	 * from both the distributions.
	 *
	 * @param samples_from_p Samples from p.
	 * @param samples_from_q Samples from q.
	 */
	CQuadraticTimeMMD(CFeatures* samples_from_p, CFeatures* samples_from_q);

	/** Destructor */
	virtual ~CQuadraticTimeMMD();

	/**
	 * Method that initializes/replaces samples from p. It will invalidate
	 * existing pre-computed kernel, if any, from previous run. However, if
	 * the underlying kernel, if set already by this point, is an instance of
	 * CCustomKernel itself, the supplied features will be ignored.
	 *
	 * @param samples_from_p Samples from p.
	 */
	virtual void set_p(CFeatures* samples_from_p);

	/**
	 * Method that initializes/replaces samples from q. It will invalidate
	 * existing pre-computed kernel, if any, from previous run. However, if
	 * the underlying kernel, if set already by this point, is an instance of
	 * CCustomKernel itself, the supplied features will be ignored.
	 *
	 * @param samples_from_p Samples from q.
	 */
	virtual void set_q(CFeatures* samples_from_q);

	/**
	 * Method that creates a merged copy of CFeatures instance from both
	 * the features, appending the samples from p and q. This method does not
	 * cache the merged copy from previous call. So, calling this method will
	 * create a new instance every time.
	 *
	 * @return The merged samples.
	 */
	CFeatures* get_p_and_q();

	/**
	 * Method that sets the kernel instance to be used. If a CCustomKernel is
	 * set, then the features passed would be effectively ignored. Therefore,
	 * if this is the intended behavior, simply passing two instances of
	 * CDummyFeatures would do (since they cannot be left null as of now).
	 *
	 * If a pre-computed instance already exists from previous runs, this will
	 * invalidate that one and free memory.
	 *
	 * @param kernel The kernel instance.
	 */
	virtual void set_kernel(CKernel* kernel);

	/**
	 * Method that learns/selects the kernel from a set of provided kernel
	 * instances added from the add_kernel() methods. Upon selection, it
	 * internally replaces the kernel instance, if any, that was already
	 * present.
	 *
	 * Please make sure to set the train-test mode on before using this method.
	 */
	virtual void select_kernel();

	/**
	 * Method that computes the estimator of MMD^2 (biased/unbiased/incomplete)
	 * as set from set_statistic_type() method. Default is unbiased.
	 *
	 * @return A normalized value of the MMD^2 estimator.
	 */
	virtual float64_t compute_statistic();

	/**
	 * Method that returns a number of null-samples, based on the null approximation
	 * method that was set using set_null_approximation_method(). Default is permutation.
	 *
	 * @return Normalized values of the MMD^2 estimates under null hypothesis.
	 */
	virtual SGVector<float64_t> sample_null();

	/**
	 * Method that computes the p-value from the provided statistic.
	 *
	 * @param statistic The test statistic
	 * @return The p-value computed using the null-appriximation method specified.
	 */
	virtual float64_t compute_p_value(float64_t statistic);

	/**
	 * Method that computes the threshold from the provided significance level (alpha).
	 *
	 * @param alpha The significance level (value should be between 0 and 1)
	 * @return The threshold computed using the null-approximation method specified.
	 */
	virtual float64_t compute_threshold(float64_t alpha);

	/**
	 * Method that computes an estimate of the variance of the unbiased MMD^2 estimator
	 * under the assumption that the null hypothesis was true.
	 *
	 * @return The variance estimate of the unbiased MMD^2 estimator under null.
	 */
	float64_t compute_variance_h0();

	/**
	 * Method that computes an estimate of the variance of the unbiased MMD^2 estimator
	 * under the assumption that the alternative hypothesis was true.
	 *
	 * @return The variance estimate of the unbiased MMD^2 estimator under alternative.
	 */
	float64_t compute_variance_h1();

	/**
	 * Method that returns the internal instance of CMultiKernelQuadraticTimeMMD which
	 * provides a similar API to this class to compute the estimates for multiple kernel
	 * all at once. This internal instance shares the same set of samples with this one
	 * but the kernel has to be added seperately using multikernel().add_kernel() method.
	 *
	 * @return An internal instance of CMultiKernelQuadraticTimeMMD.
	 */
	CMultiKernelQuadraticTimeMMD* multikernel();

	/**
	 * Method that sets the number of eigenvalues to be used when spectral estimation
	 * of the null samples is used. Will be ignored if null-approximation method was
	 * anything else.
	 *
	 * @param num_eigenvalues The number of eigenvalues to be used from the eigenspectrum
	 * of the Gram matrix.
	 */
	void spectrum_set_num_eigenvalues(index_t num_eigenvalues);

	/** @return The number of eigenvalues in use for the spectral test */
	index_t spectrum_get_num_eigenvalues() const;

	/**
	 * Use this method when pre-computation of the kernel matrix is NOT desired. By default
	 * this class always precomputes the Gram matrix. Please note that the performance will
	 * be slow if this option is turned off.
	 *
	 * @param precompute Flag to whether pre-compute the kernel matrix internally or not.
	 * If false, the kernel matrix is NOT pre-computed, otherwise it is. Default is true.
	 */
	void precompute_kernel_matrix(bool precompute);

	/**
	 * Method that saves the permutation indices that will be used while sampling from the
	 * null distribution in case permutation approach was adopted. The indices will be
	 * available only after a successful run of the permutation test. By default, the indices
	 * are never saved.
	 *
	 * @param save_inds Whether to save the permutation indices or not. If true, the indices
	 * are saved, otherwise not.
	 */
	void save_permutation_inds(bool save_inds);

	/**
	 * Method that returns the permutation indices, if that option was turned on by using
	 * the save_permutation_inds(true).
	 *
	 * @return The permutation indices, one column per null-sample.
	 */
	SGMatrix<index_t> get_permutation_inds() const;

	/** @return The name of the class */
	virtual const char* get_name() const;

protected:
	virtual float64_t normalize_statistic(float64_t statistic) const;

private:
	struct Self;
	std::unique_ptr<Self> self;
	void init();
};

}
#endif // QUADRATIC_TIME_MMD_H_
