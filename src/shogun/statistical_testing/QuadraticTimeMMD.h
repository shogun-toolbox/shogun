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
#include <shogun/statistical_testing/MMD.h>

namespace shogun
{

template <typename> class SGVector;

class CQuadraticTimeMMD : public CMMD
{
	using operation = std::function<float64_t(SGMatrix<float64_t>)>;
public:
	CQuadraticTimeMMD();
	CQuadraticTimeMMD(CFeatures* samples_from_p, CFeatures* samples_from_q);

	virtual ~CQuadraticTimeMMD();

	virtual SGVector<float64_t> sample_null() override;
	void spectrum_set_num_eigenvalues(index_t num_eigenvalues);

	virtual float64_t compute_p_value(float64_t statistic) override;
	virtual float64_t compute_threshold(float64_t alpha) override;

	virtual const char* get_name() const;
private:
	struct Self;
	std::unique_ptr<Self> self;

	virtual const operation get_direct_estimation_method() const override;
	virtual const float64_t normalize_statistic(float64_t statistic) const override;
	virtual const float64_t normalize_variance(float64_t variance) const override;
	SGVector<float64_t> gamma_fit_null();
};

}
#endif // QUADRATIC_TIME_MMD_H_
