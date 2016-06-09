/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
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

#include <shogun/base/some.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/statistical_testing/TwoDistributionTest.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace shogun
{

class CTwoDistributionTestMock : public CTwoDistributionTest
{
public:
	MOCK_METHOD0(compute_statistic, float64_t());
	MOCK_METHOD0(sample_null, SGVector<float64_t>());
};

}

using namespace shogun;

TEST(TwoDistributionTest, compute_distance_dense)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;

	auto gen_p=some<CMeanShiftDataGenerator>(0, dim, 0);
	auto gen_q=some<CMeanShiftDataGenerator>(difference, dim, 0);

	auto feats_p=static_cast<CDenseFeatures<float64_t>*>(gen_p->get_streamed_features(m));
	auto feats_q=static_cast<CDenseFeatures<float64_t>*>(gen_q->get_streamed_features(n));

	auto test=some<CTwoDistributionTestMock>();
	test->set_p(feats_p);
	test->set_q(feats_q);

	auto distance=test->compute_distance();
	auto distance_mat1=distance->get_distance_matrix();

	SGMatrix<float64_t> data_p_and_q(dim, m+n);
	auto data_p=feats_p->get_feature_matrix();
	auto data_q=feats_q->get_feature_matrix();
	std::copy(data_p.data(), data_p.data()+data_p.size(), data_p_and_q.data());
	std::copy(data_q.data(), data_q.data()+data_q.size(), data_p_and_q.data()+data_p.size());
	auto feats_p_and_q=new CDenseFeatures<float64_t>(data_p_and_q);

	auto euclidean_distance=some<CEuclideanDistance>();
	euclidean_distance->init(feats_p_and_q, feats_p_and_q);
	auto distance_mat2=euclidean_distance->get_distance_matrix();

	EXPECT_TRUE(distance_mat1.num_rows==distance_mat2.num_rows);
	EXPECT_TRUE(distance_mat1.num_cols==distance_mat2.num_cols);
	for (size_t i=0; i<distance_mat1.size(); ++i)
		EXPECT_NEAR(distance_mat1.data()[i], distance_mat2.data()[i], 1E-6);

	SG_UNREF(distance);
}

TEST(TwoDistributionTest, compute_distance_streaming)
{
	const index_t m=5;
	const index_t n=10;
	const index_t dim=1;
	const float64_t difference=0.5;

	auto gen_p=new CMeanShiftDataGenerator(0, dim, 0);
	auto gen_q=new CMeanShiftDataGenerator(difference, dim, 0);

	auto test=some<CTwoDistributionTestMock>();
	test->set_p(gen_p);
	test->set_q(gen_q);
	test->set_num_samples_p(m);
	test->set_num_samples_q(n);

	sg_rand->set_seed(12345);
	auto distance=test->compute_distance();
	auto distance_mat1=distance->get_distance_matrix();

	sg_rand->set_seed(12345);
	auto feats_p=static_cast<CDenseFeatures<float64_t>*>(gen_p->get_streamed_features(m));
	auto feats_q=static_cast<CDenseFeatures<float64_t>*>(gen_q->get_streamed_features(n));

	SGMatrix<float64_t> data_p_and_q(dim, m+n);
	auto data_p=feats_p->get_feature_matrix();
	auto data_q=feats_q->get_feature_matrix();
	std::copy(data_p.data(), data_p.data()+data_p.size(), data_p_and_q.data());
	std::copy(data_q.data(), data_q.data()+data_q.size(), data_p_and_q.data()+data_p.size());
	auto feats_p_and_q=new CDenseFeatures<float64_t>(data_p_and_q);
	SG_UNREF(feats_p);
	SG_UNREF(feats_q);

	auto euclidean_distance=some<CEuclideanDistance>();
	euclidean_distance->init(feats_p_and_q, feats_p_and_q);
	auto distance_mat2=euclidean_distance->get_distance_matrix();

	EXPECT_TRUE(distance_mat1.num_rows==distance_mat2.num_rows);
	EXPECT_TRUE(distance_mat1.num_cols==distance_mat2.num_cols);
	for (size_t i=0; i<distance_mat1.size(); ++i)
		EXPECT_NEAR(distance_mat1.data()[i], distance_mat2.data()[i], 1E-6);

	SG_UNREF(distance);
}
