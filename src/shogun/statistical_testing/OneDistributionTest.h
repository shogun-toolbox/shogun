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

#ifndef ONE_DISTRIBUTION_TEST_H_
#define ONE_DISTRIBUTION_TEST_H_

#include <shogun/statistical_testing/HypothesisTest.h>

namespace shogun
{

class COneDistributionTest : public CHypothesisTest
{
public:
	COneDistributionTest();
	virtual ~COneDistributionTest();

	void set_samples(CFeatures* samples);
	CFeatures* get_samples() const;

	void set_num_samples(index_t num_samples);
	index_t get_num_samples() const;

	virtual float64_t compute_statistic()=0;
	virtual SGVector<float64_t> sample_null()=0;

	virtual const char* get_name() const;
};

}
#endif // ONE_DISTRIBUTION_TEST_H_
