/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2012 - 2013 Heiko Strathmann
 * Written (w) 2014 - 2017 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/TwoDistributionTest.h>
#include <shogun/statistical_testing/internals/DataManager.h>
#include <shogun/statistical_testing/internals/TestTypes.h>
#include <shogun/statistical_testing/internals/NextSamples.h>

using namespace shogun;

TwoDistributionTest::TwoDistributionTest() : HypothesisTest(internal::TwoDistributionTest::num_feats)
{
}

TwoDistributionTest::~TwoDistributionTest()
{
}

void TwoDistributionTest::set_p(std::shared_ptr<Features> samples_from_p)
{
	REQUIRE(samples_from_p, "Samples from P cannot be NULL!\n");
	auto& dm=get_data_mgr();
	dm.samples_at(0)=samples_from_p;
}

std::shared_ptr<Features> TwoDistributionTest::get_p() const
{
	const auto& dm=get_data_mgr();
	return dm.samples_at(0);
}

void TwoDistributionTest::set_q(std::shared_ptr<Features> samples_from_q)
{
	REQUIRE(samples_from_q, "Samples from Q cannot be NULL!\n");
	auto& dm=get_data_mgr();
	dm.samples_at(1)=samples_from_q;
}

std::shared_ptr<Features> TwoDistributionTest::get_q() const
{
	const auto& dm=get_data_mgr();
	return dm.samples_at(1);
}

void TwoDistributionTest::set_num_samples_p(index_t num_samples_from_p)
{
	auto& dm=get_data_mgr();
	dm.num_samples_at(0)=num_samples_from_p;
}

const index_t TwoDistributionTest::get_num_samples_p() const
{
	const auto& dm=get_data_mgr();
	return dm.num_samples_at(0);
}

void TwoDistributionTest::set_num_samples_q(index_t num_samples_from_q)
{
	auto& dm=get_data_mgr();
	dm.num_samples_at(1)=num_samples_from_q;
}

const index_t TwoDistributionTest::get_num_samples_q() const
{
	const auto& dm=get_data_mgr();
	return dm.num_samples_at(1);
}

std::shared_ptr<CustomDistance> TwoDistributionTest::compute_distance(std::shared_ptr<Distance> distance)
{
	auto& data_mgr=get_data_mgr();
	bool is_blockwise=data_mgr.is_blockwise();
	data_mgr.set_blockwise(false);

	data_mgr.start();
	auto samples=data_mgr.next();
	REQUIRE(!samples.empty(), "Could not fetch samples!\n");

	std::shared_ptr<Features> samples_p=samples[0][0];
	std::shared_ptr<Features> samples_q=samples[1][0];

	distance->cleanup();
	distance->remove_lhs_and_rhs();
	REQUIRE(distance->init(samples_p, samples_q), "Could not initialize distance instance!\n");
	auto dist_mat=distance->get_distance_matrix<float32_t>();
	distance->remove_lhs_and_rhs();
	distance->cleanup();

	samples.clear();
	data_mgr.end();
	data_mgr.set_blockwise(is_blockwise);

	auto precomputed_distance=std::make_shared<CustomDistance>();
	precomputed_distance->set_full_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
	return precomputed_distance;
}

std::shared_ptr<CustomDistance> TwoDistributionTest::compute_joint_distance(std::shared_ptr<Distance> distance)
{
	REQUIRE(distance!=nullptr, "Distance instance cannot be NULL!\n");
	auto& data_mgr=get_data_mgr();
	bool is_blockwise=data_mgr.is_blockwise();
	data_mgr.set_blockwise(false);

	data_mgr.start();
	auto samples=data_mgr.next();
	REQUIRE(!samples.empty(), "Could not fetch samples!\n");

	std::shared_ptr<Features> samples_p=samples[0][0];
	std::shared_ptr<Features> samples_q=samples[1][0];
	auto p_and_q=samples_p->create_merged_copy(samples_q);

	samples.clear();
	data_mgr.end();
	data_mgr.set_blockwise(is_blockwise);

	distance->cleanup();
	distance->remove_lhs_and_rhs();
	REQUIRE(distance->init(p_and_q, p_and_q), "Could not initialize distance instance!\n");
	auto dist_mat=distance->get_distance_matrix<float32_t>();
	distance->remove_lhs_and_rhs();
	distance->cleanup();

	auto precomputed_distance=std::make_shared<CustomDistance>();
	precomputed_distance->set_triangle_distance_matrix_from_full(dist_mat.data(), dist_mat.num_rows, dist_mat.num_cols);
	return precomputed_distance;
}

const char* TwoDistributionTest::get_name() const
{
	return "TwoDistributionTest";
}
