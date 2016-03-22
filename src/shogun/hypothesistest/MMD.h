/*
 * Restructuring Shogun's statistical hypothesis testing framework.
 * Copyright (C) 2016  Soumyajit De
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef MMD_H_
#define MMD_H_

#include <memory>
#include <shogun/hypothesistest/TwoSampleTest.h>

namespace shogun
{

class CKernel;
template <typename T> class SGVector;

enum class S_TYPE
{
	UNBIASED_FULL,
	UNBIASED_INCOMPLETE,
	BIASED_FULL
};

enum class V_METHOD
{
	DIRECT,
	PERMUTATION
};

enum class N_METHOD
{
	PERMUTATION,
	MMD1_GAUSSIAN,
	MMD2_SPECTRUM,
	MMD2_GAMMA
};

template <class Derived>
class CMMD : public CTwoSampleTest
{
public:
	CMMD();
	virtual ~CMMD();

	virtual void ugly_hack_for_class_list() = 0;

	virtual float64_t compute_statistic() override;
	SGVector<float64_t> compute_statistic(bool multiple_kernels);

	float64_t compute_variance();
	SGVector<float64_t> compute_variance(bool multiple_kernels);

	void set_statistic_type(S_TYPE stype);
	const S_TYPE get_statistic_type() const;

	void set_variance_estimation_method(V_METHOD vmethod);
	const V_METHOD get_variance_estimation_method() const;

	void set_simulate_null(bool null);
	void set_num_null_samples(index_t null_samples);
	const index_t get_num_null_samples() const;

	virtual SGVector<float64_t> sample_null() override;

	void set_null_approximation_method(N_METHOD nmethod);
	const N_METHOD get_null_approximation_method() const;

	void use_gpu(bool gpu);

	virtual const char* get_name() const;
private:
	struct Self;
	std::unique_ptr<Self> self;

};

} // namespace shogun
#endif // MMD_H_
