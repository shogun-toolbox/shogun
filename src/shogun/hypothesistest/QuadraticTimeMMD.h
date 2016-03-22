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

#ifndef QUADRATIC_TIME_MMD_H_
#define QUADRATIC_TIME_MMD_H_

#include <memory>
#include <shogun/hypothesistest/MMD.h>
#include <shogun/hypothesistest/internals/mmd/FullDirect.h>

namespace shogun
{

template <typename> class SGVector;

class CQuadraticTimeMMD : public CMMD<CQuadraticTimeMMD>
{
	friend class CMMD;
public:
	CQuadraticTimeMMD();
	virtual ~CQuadraticTimeMMD();

	virtual void ugly_hack_for_class_list() override {}

	virtual SGVector<float64_t> sample_null() override;
	void set_num_eigenvalues(index_t num_eigenvalues);

	virtual float64_t compute_p_value(float64_t statistic) override;
	virtual float64_t compute_threshold(float64_t alpha) override;

	virtual const char* get_name() const;
private:
	struct Self;
	std::unique_ptr<Self> self;

	static internal::mmd::FullDirect get_direct_estimation_method();
	const float64_t normalize_statistic(float64_t statistic) const;
	const float64_t normalize_variance(float64_t variance) const;
	SGVector<float64_t> fit_null_gamma();
};

}
#endif // QUADRATIC_TIME_MMD_H_
