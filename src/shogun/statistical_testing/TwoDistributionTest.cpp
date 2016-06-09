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

#include <shogun/distance/CustomDistance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/statistical_testing/TwoDistributionTest.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/TestTypes.h>
#include <shogun/statistical_testing/internals/NextSamples.h>
#include <shogun/statistical_testing/internals/FeaturesUtil.h>

using namespace shogun;
using namespace internal;

CTwoDistributionTest::CTwoDistributionTest() : CHypothesisTest(TwoDistributionTest::num_feats)
{
}

CTwoDistributionTest::~CTwoDistributionTest()
{
}

void CTwoDistributionTest::set_p(CFeatures* samples_from_p)
{
	auto& dm=get_data_mgr();
	dm.samples_at(0)=samples_from_p;
}

CFeatures* CTwoDistributionTest::get_p() const
{
	const auto& dm=get_data_mgr();
	return dm.samples_at(0);
}

void CTwoDistributionTest::set_q(CFeatures* samples_from_q)
{
	auto& dm=get_data_mgr();
	dm.samples_at(1)=samples_from_q;
}

CFeatures* CTwoDistributionTest::get_q() const
{
	const auto& dm=get_data_mgr();
	return dm.samples_at(1);
}

void CTwoDistributionTest::set_num_samples_p(index_t num_samples_from_p)
{
	auto& dm=get_data_mgr();
	dm.num_samples_at(0)=num_samples_from_p;
}

const index_t CTwoDistributionTest::get_num_samples_p() const
{
	const auto& dm=get_data_mgr();
	return dm.num_samples_at(0);
}

void CTwoDistributionTest::set_num_samples_q(index_t num_samples_from_q)
{
	auto& dm=get_data_mgr();
	dm.num_samples_at(1)=num_samples_from_q;
}

const index_t CTwoDistributionTest::get_num_samples_q() const
{
	const auto& dm=get_data_mgr();
	return dm.num_samples_at(1);
}

CCustomDistance* CTwoDistributionTest::compute_distance()
{
	auto distance=new CCustomDistance();
	auto& data_mgr=get_data_mgr();

	bool is_blockwise=data_mgr.is_blockwise();
	data_mgr.set_blockwise(false);

	data_mgr.start();
	auto samples=data_mgr.next();
	if (!samples.empty())
	{
		CFeatures *samples_p=samples[0][0].get();
		CFeatures *samples_q=samples[1][0].get();
		try
		{
			auto p_and_q=FeaturesUtil::create_merged_copy(samples_p, samples_q);
			samples.clear();
			auto euclidean_distance=std::unique_ptr<CEuclideanDistance>(new CEuclideanDistance());
			if (euclidean_distance->init(p_and_q, p_and_q))
			{
				euclidean_distance->set_disable_sqrt(true);
				auto dist_mat=euclidean_distance->get_distance_matrix<float32_t>();
				distance->set_triangle_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
			}
			else
			{
				SG_SERROR("Computing distance matrix was not possible! Please contact Shogun developers.\n");
			}
		}
		catch (ShogunException e)
		{
			SG_SERROR("%s, Data is too large! Computing distance matrix was not possible!\n", e.get_exception_string());
		}
	}
	else
		SG_SERROR("Could not fetch samples!\n");

	data_mgr.end();
	data_mgr.set_blockwise(is_blockwise);

	return distance;
}

const char* CTwoDistributionTest::get_name() const
{
	return "TwoDistributionTest";
}
