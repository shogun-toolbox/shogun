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

#include <shogun/hypothesistest/TwoDistributionTest.h>
#include <shogun/hypothesistest/internals/DataManager.h>
#include <shogun/hypothesistest/internals/TestTypes.h>

using namespace shogun;
using namespace internal;

CTwoDistributionTest::CTwoDistributionTest(index_t num_kernels)
: CHypothesisTest(TwoDistributionTest::num_feats, num_kernels)
{
}

CTwoDistributionTest::~CTwoDistributionTest()
{
}

void CTwoDistributionTest::set_p(CFeatures* samples_from_p)
{
	auto& dm = get_data_manager();
	dm.samples_at(0) = samples_from_p;
}

CFeatures* CTwoDistributionTest::get_p() const
{
	const auto& dm = get_data_manager();
	return dm.samples_at(0);
}

void CTwoDistributionTest::set_q(CFeatures* samples_from_q)
{
	auto& dm = get_data_manager();
	dm.samples_at(1) = samples_from_q;
}

CFeatures* CTwoDistributionTest::get_q() const
{
	const auto& dm = get_data_manager();
	return dm.samples_at(1);
}

void CTwoDistributionTest::set_num_samples_p(index_t num_samples_from_p)
{
	auto& dm = get_data_manager();
	dm.num_samples_at(0) = num_samples_from_p;
}

const index_t CTwoDistributionTest::get_num_samples_p() const
{
	const auto& dm = get_data_manager();
	return dm.num_samples_at(0);
}

void CTwoDistributionTest::set_num_samples_q(index_t num_samples_from_q)
{
	auto& dm = get_data_manager();
	dm.num_samples_at(1) = num_samples_from_q;
}

const index_t CTwoDistributionTest::get_num_samples_q() const
{
	const auto& dm = get_data_manager();
	return dm.num_samples_at(1);
}

const char* CTwoDistributionTest::get_name() const
{
	return "TwoDistributionTest";
}
