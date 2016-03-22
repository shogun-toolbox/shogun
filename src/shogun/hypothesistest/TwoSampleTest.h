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

#ifndef TWO_SAMPLE_TEST_H_
#define TWO_SAMPLE_TEST_H_

#include <shogun/hypothesistest/TwoDistributionTest.h>

namespace shogun
{

class CKernel;

class CTwoSampleTest : public CTwoDistributionTest
{
public:
	CTwoSampleTest();
	virtual ~CTwoSampleTest();

	void set_kernel(CKernel* kernel);
	CKernel* get_kernel() const;

	virtual float64_t compute_statistic() = 0;
	virtual SGVector<float64_t> sample_null() = 0;

	virtual const char* get_name() const;
};

}
#endif // TWO_SAMPLE_TEST_H_
