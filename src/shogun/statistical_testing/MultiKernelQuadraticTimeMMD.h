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

#ifndef MULTI_KERNEL_QUADRATIC_TIME_MMD_H_
#define MULTI_KERNEL_QUADRATIC_TIME_MMD_H_

#include <memory>
#include <shogun/base/SGObject.h>

namespace shogun
{

class CFeatures;
class CQuadraticTimeMMD;
class CShiftInvariantKernel;
template <typename> class SGVector;

namespace internal
{
class KernelManager;
class MaxMeasure;
class MaxTestPower;
}

/**
 * @brief Class that performs quadratic time MMD test optimized for multiple
 * shift-invariant kernels. If the kernels are not shift-invariant, then the
 * class CQuadraticTimeMMD should be used multiple times instead of this one.
 *
 * If the features are updated, then (if any) existing precomputed distance
 * instance has to be invalidated by the owner (CQuadraticTimeMMD instance).
 * This is already taken care of internally. A separate instance of this class
 * should never be created by invoking the constructor. One should always
 * call the CQuadraticTimeMMD::multikernel() method to get an instance of this
 * class.
 */
class CMultiKernelQuadraticTimeMMD : public CSGObject
{
	friend class CQuadraticTimeMMD;
	friend class internal::MaxMeasure;
	friend class internal::MaxTestPower;
private:
	CMultiKernelQuadraticTimeMMD(CQuadraticTimeMMD* owner);
public:
	/**
	 * Default constructor. Should never be invoked by the user. Please use
	 * CQuadraticTimeMMD::multikernel() to obtain an instance of this class.
	 */
	CMultiKernelQuadraticTimeMMD();

	/** Destructor */
	virtual ~CMultiKernelQuadraticTimeMMD();

	/**
	 * Method that adds instances of shift-invariant kernels (e.g. CGaussianKernel).
	 * Invoke multiple times to add desired number of kernels. All the estimators
	 * obtianed from the computation will be in the same order the kernels were
	 * added.
	 *
	 * @param kernel The kernel instance.
	 */
	void add_kernel(CShiftInvariantKernel *kernel);

	/**
	 * Method that does internal cleanups (essentially releases memory from the
	 * internally stored pair-wise distance instance.
	 */
	void cleanup();

	/**
	 * Method that returns normalized estimates of the MMD^2 for all the kernels.
	 *
	 * @return A vector of values for normalized estimates of the MMD^2 for all
	 * the kernels.
	 */
	SGVector<float64_t> compute_statistic();

	/**
	 * Method that returns variance estimates of the unbiased MMD^2 estimator
	 * for all the kernels under the assumption that null-hypothesis was true.
	 *
	 * @return A vector of values for variance estimates of the unbiased MMD^2
	 * estimator for all the kernels under null.
	 */
	SGVector<float64_t> compute_variance_h0();

	/**
	 * Method that returns variance estimates of the unbiased MMD^2 estimator
	 * for all the kernels under the assumption that alternative-hypothesis was true.
	 *
	 * @return A vector of values for variance estimates of the unbiased MMD^2
	 * estimator for all the kernels under alternative.
	 */
	SGVector<float64_t> compute_variance_h1();

	/**
	 * Method that returns proxy measures of the test-power computed as the
	 * ratio of the unbiased MMD^2 estimator and sqrt of the variance estimate
	 * of it under alternative.
	 *
	 * @return A vector of values for proxy measures of test-power for all kernels.
	 */
	SGVector<float64_t> compute_test_power();

	/*
	 * Method that computes the null-samples for all the kernels, one column per kernel.
	 * This method uses permutation as the null-approximation technique.
	 *
	 * @return Null-samples for all the kernels.
	 */
	SGMatrix<float32_t> sample_null();

	/**
	 * Method that computes the p-values for all the kernels. The API is different
	 * here than CQuadraticTimeMMD since the test-statistics for the kernels are computed
	 * internally on the fly. This method uses permutation as the null-approximation
	 * technique.
	 *
	 * @return A vector of p-values for all the kernels.
	 */
	SGVector<float64_t> compute_p_value();

	/**
	 * Method that performs the test and returns whether the null hypothesis was
	 * accepted or rejected, based on the provided significance level.
	 *
	 * @param alpha The significance level of the hypothesis test. Should be between
	 * 0 and 1.
	 * @return A vector of values of the test results (true - null hypothesis was
	 * accepted, false - otherwise) for all the kernels.
	 */
	SGVector<bool> perform_test(float64_t alpha);

	/** @return The name of the class */
	virtual const char* get_name() const;
private:
	struct Self;
	std::unique_ptr<Self> self;
	void invalidate_precomputed_distance();
	SGVector<float64_t> statistic(const internal::KernelManager& kernel_mgr);
	SGVector<float64_t> variance_h1(const internal::KernelManager& kernel_mgr);
	SGVector<float64_t> test_power(const internal::KernelManager& kernel_mgr);
	SGMatrix<float32_t> sample_null(const internal::KernelManager& kernel_mgr);
	SGVector<float64_t> p_values(const internal::KernelManager& kernel_mgr);
};

}
#endif // MULTI_KERNEL_QUADRATIC_TIME_MMD_H_
