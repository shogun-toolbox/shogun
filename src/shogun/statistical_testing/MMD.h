/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
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
#include <shogun/statistical_testing/TwoSampleTest.h>

namespace shogun
{

class CKernel;
template <typename> class SGVector;
template <typename> class SGMatrix;

namespace internal
{

class MaxTestPower;

}

enum class EStatisticType
{
	UNBIASED_FULL,
	UNBIASED_INCOMPLETE,
	BIASED_FULL
};

enum class EVarianceEstimationMethod
{
	DIRECT,
	PERMUTATION
};

enum class ENullApproximationMethod
{
	PERMUTATION,
	MMD1_GAUSSIAN,
	MMD2_SPECTRUM,
	MMD2_GAMMA
};

enum class EKernelSelectionMethod
{
	MEDIAN_HEURISRIC,
	MAXIMIZE_MMD,
	MAXIMIZE_POWER
};

class CMMD : public CTwoSampleTest
{
	using operation=std::function<float32_t(SGMatrix<float32_t>)>;
	friend class internal::MaxTestPower;
public:
	CMMD();
	virtual ~CMMD();

	void add_kernel(CKernel *kernel);
	void select_kernel(EKernelSelectionMethod kmethod);

	virtual float64_t compute_statistic() override;
	virtual float64_t compute_variance();

	virtual SGVector<float64_t> sample_null() override;

	void use_gpu(bool gpu);
	void cleanup();

	void set_statistic_type(EStatisticType stype);
	const EStatisticType get_statistic_type() const;

	void set_variance_estimation_method(EVarianceEstimationMethod vmethod);
	const EVarianceEstimationMethod get_variance_estimation_method() const;

	void set_num_null_samples(index_t null_samples);
	const index_t get_num_null_samples() const;

	void set_null_approximation_method(ENullApproximationMethod nmethod);
	const ENullApproximationMethod get_null_approximation_method() const;

	virtual const char* get_name() const;
protected:
	virtual const operation get_direct_estimation_method() const = 0;
	virtual const float64_t normalize_statistic(float64_t statistic) const = 0;
	virtual const float64_t normalize_variance(float64_t variance) const = 0;
	bool use_gpu() const;
private:
	struct Self;
	std::unique_ptr<Self> self;
	virtual std::pair<float64_t, float64_t> compute_statistic_variance();
};

}
#endif // MMD_H_
