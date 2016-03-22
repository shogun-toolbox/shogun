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

#ifndef INDEPENDENCE_TEST_H_
#define INDEPENDENCE_TEST_H_

#include <shogun/statistical_testing/TwoDistributionTest.h>

namespace shogun
{

class CKernel;

class CIndependenceTest : public CTwoDistributionTest
{
public:
	CIndependenceTest();
	virtual ~CIndependenceTest();

	void set_kernel_p(CKernel* kernel_p);
	CKernel* get_kernel_p() const;

	void set_kernel_q(CKernel* kernel_q);
	CKernel* get_kernel_q() const;

	virtual float64_t compute_statistic() = 0;
	virtual SGVector<float64_t> sample_null() = 0;

	virtual const char* get_name() const;
};

}
#endif // INDEPENDENCE_TEST_H_
