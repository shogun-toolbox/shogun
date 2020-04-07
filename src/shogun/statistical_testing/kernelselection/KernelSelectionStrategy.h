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

#ifndef KERNEL_SELECTION_STRAGERY_H_
#define KERNEL_SELECTION_STRAGERY_H_

#include <memory>
#include <shogun/base/SGObject.h>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/mathematics/RandomMixin.h>

namespace shogun
{

class Kernel;
class MMD;
class QuadraticTimeMMD;
template <class> class SGVector;
template <class> class SGMatrix;

namespace internal
{

class KernelManager;

}
#ifndef DOXYGEN_SHOULD_SKIP_THIS
class KernelSelectionStrategy : public RandomMixin<SGObject>
{
	friend class MMD;
	friend class StreamingMMD;
	friend class QuadraticTimeMMD;
public:
	KernelSelectionStrategy();
	KernelSelectionStrategy(EKernelSelectionMethod method, bool weighted = false);
	KernelSelectionStrategy(EKernelSelectionMethod method, index_t num_runs, index_t num_folds, float64_t alpha);
	KernelSelectionStrategy(const KernelSelectionStrategy& other)=delete;
	KernelSelectionStrategy& operator=(const KernelSelectionStrategy& other)=delete;
	~KernelSelectionStrategy() override;

	KernelSelectionStrategy& use_method(EKernelSelectionMethod method);
	KernelSelectionStrategy& use_num_runs(index_t num_runs);
	KernelSelectionStrategy& use_num_folds(index_t num_folds);
	KernelSelectionStrategy& use_alpha(float64_t alpha);
	KernelSelectionStrategy& use_weighted(bool weighted);

	EKernelSelectionMethod get_method() const;
	index_t get_num_runs() const;
	index_t get_num_folds() const;
	float64_t get_alpha() const;
	bool get_weighted() const;

	void add_kernel(std::shared_ptr<Kernel> kernel);
	std::shared_ptr<Kernel> select_kernel(std::shared_ptr<MMD> estimator);
	const char* get_name() const override;
	void erase_intermediate_results();

	SGMatrix<float64_t> get_measure_matrix();
	SGVector<float64_t> get_measure_vector();
private:
	struct Self;
	std::unique_ptr<Self> self;
	void init();
	const internal::KernelManager& get_kernel_mgr() const;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS
}
#endif // KERNEL_SELECTION_STRAGERY_H_
