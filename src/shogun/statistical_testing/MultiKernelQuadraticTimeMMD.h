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
}

/**
 * @brief Class that performs quadratic time MMD test optimized for multiple
 * shift-invariant kernels. If the kernels are not shift-invariant, then the
 * class CQuadraticTimeMMD should be used multiple times instead of this one.
 *
 * If the features are updated, then (if any) existing precomputed distance
 * instance has to be invalidated by the owner (CQuadraticTimeMMD instance).
 */
class CMultiKernelQuadraticTimeMMD : public CSGObject
{
	friend class CQuadraticTimeMMD;
	friend class internal::MaxMeasure;
private:
	CMultiKernelQuadraticTimeMMD(CQuadraticTimeMMD* owner);
public:
	CMultiKernelQuadraticTimeMMD();
	virtual ~CMultiKernelQuadraticTimeMMD();
	void add_kernel(CShiftInvariantKernel *kernel);
	void cleanup();

	SGVector<float64_t> compute_statistic();
	SGVector<float64_t> compute_variance_h0();
	SGVector<float64_t> compute_variance_h1();
	SGMatrix<float32_t> sample_null();
	SGVector<float64_t> compute_p_value();
	SGVector<bool> perform_test(float64_t alpha);

	virtual const char* get_name() const;
private:
	struct Self;
	std::unique_ptr<Self> self;
	void invalidate_precomputed_distance();
	SGVector<float64_t> statistic(const internal::KernelManager& kernel_mgr);
	SGVector<float64_t> variance_h1(const internal::KernelManager& kernel_mgr);
	SGMatrix<float32_t> sample_null(const internal::KernelManager& kernel_mgr);
	SGVector<float64_t> p_values(const internal::KernelManager& kernel_mgr);
};

}
#endif // MULTI_KERNEL_QUADRATIC_TIME_MMD_H_
