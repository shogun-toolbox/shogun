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

#ifndef HYPOTHESIS_TEST_EXP_H_
#define HYPOTHESIS_TEST_EXP_H_

#include <memory>
#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>

namespace shogun
{

class CFeatures;

namespace internal
{

class DataManager;
class KernelManager;

}

class CHypothesisTestExp : public CSGObject
{
public:
	CHypothesisTestExp(index_t num_distributions, index_t num_kernels);
	virtual ~CHypothesisTestExp();

	virtual float64_t compute_statistic() = 0;

	virtual float64_t compute_p_value(float64_t statistic);
	virtual float64_t compute_threshold(float64_t alpha);

	float64_t perform_test();
	bool perform_test(float64_t alpha);

	virtual SGVector<float64_t> sample_null() = 0;

	virtual const char* get_name() const;
private:
	struct Self;
	std::unique_ptr<Self> self;
protected:
	internal::DataManager& get_data_manager();
	const internal::DataManager& get_data_manager() const;

	internal::KernelManager& get_kernel_manager();
	const internal::KernelManager& get_kernel_manager() const;
};

}

#endif // HYPOTHESIS_TEST_EXP_H_
