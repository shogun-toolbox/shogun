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

#ifndef TWO_DISTRIBUTION_TEST_H_
#define TWO_DISTRIBUTION_TEST_H_

#include <shogun/lib/common.h>
#include <shogun/statistical_testing/HypothesisTest.h>
#include <shogun/statistical_testing/internals/TestTypes.h>

namespace shogun
{

class CTwoDistributionTest : public CHypothesisTest
{
public:
	CTwoDistributionTest(index_t num_kernels);
	virtual ~CTwoDistributionTest();

	CFeatures* get_p() const;
	CFeatures* get_q() const;

	void set_num_samples_p(index_t num_samples_from_p);
	void set_num_samples_q(index_t num_samples_from_q);

	const index_t get_num_samples_p() const;
	const index_t get_num_samples_q() const;

	virtual float64_t compute_statistic()=0;
	virtual SGVector<float64_t> sample_null()=0;

	virtual const char* get_name() const;
protected:
	void set_p(CFeatures* samples_from_p);
	void set_q(CFeatures* samples_from_q);
};

}
#endif // TWO_DISTRIBUTION_TEST_H_
