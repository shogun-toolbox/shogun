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

#ifndef MMD_H_
#define MMD_H_

#include <utility>
#include <memory>
#include <functional>
#include <shogun/statistical_testing/TestEnums.h>
#include <shogun/statistical_testing/TwoSampleTest.h>

namespace shogun
{

class CKernel;
class CKernelSelectionStrategy;
template <typename> class SGVector;

class CMMD : public CTwoSampleTest
{
public:
	CMMD();
	CMMD(CFeatures* samples_from_p, CFeatures* samples_from_q);
	virtual ~CMMD();

	void set_kernel_selection_strategy(EKernelSelectionMethod method, bool weighted = false);
	void set_kernel_selection_strategy(EKernelSelectionMethod method, index_t num_runs, index_t num_folds, float64_t alpha);

	void add_kernel(CKernel *kernel);
	virtual void select_kernel();

	CKernelSelectionStrategy const * get_kernel_selection_strategy() const;

	virtual float64_t compute_statistic() = 0;
	virtual SGVector<float64_t> sample_null() = 0;

	void cleanup();

	void set_num_null_samples(index_t null_samples);
	index_t get_num_null_samples() const;

	void set_statistic_type(EStatisticType stype);
	EStatisticType get_statistic_type() const;

	void set_null_approximation_method(ENullApproximationMethod nmethod);
	ENullApproximationMethod get_null_approximation_method() const;

	virtual const char* get_name() const;
protected:
	virtual float64_t normalize_statistic(float64_t statistic) const = 0;
private:
	struct Self;
	std::unique_ptr<Self> self;
	void init();
};

}
#endif // MMD_H_
