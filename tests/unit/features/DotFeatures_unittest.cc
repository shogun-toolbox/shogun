/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Michele Mazzoni
 */

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>

using namespace shogun;

class DotFeaturesTest : public ::testing::Test
{
protected:
	virtual void SetUp()
	{
		SGMatrix<float64_t> data_a(dims, num_a);
		data_a(0, 0) = 1.01611997;
		data_a(1, 0) = 0.88935567;
		data_a(2, 0) = -0.53592717;
		data_a(0, 1) = 0.24132379;
		data_a(1, 1) = 0.50475675;
		data_a(2, 1) = 0.66029218;
		data_a(0, 2) = 0.776238;
		data_a(1, 2) = 0.19904003;
		data_a(2, 2) = -0.60085628;
		data_a(0, 3) = 0.86905328;
		data_a(1, 3) = -1.22505732;
		data_a(2, 3) = -1.12045593;
		data_a(0, 4) = -0.60848342;
		data_a(1, 4) = -1.45115708;
		data_a(2, 4) = 1.15711328;
		feats_a = std::make_shared<DenseFeatures<float64_t>>(data_a);


		SGMatrix<float64_t> data_b(dims, num_b);
		data_b(0, 0) = 0.14210129;
		data_b(1, 0) = -0.36770534;
		data_b(2, 0) = 0.80232687;
		data_b(0, 1) = -0.10386986;
		data_b(1, 1) = 0.3970658;
		data_b(2, 1) = 1.15765292;
		data_b(0, 2) = 1.22478326;
		data_b(1, 2) = 0.61167198;
		data_b(2, 2) = 0.49287339;
		data_b(0, 3) = 0.04932024;
		data_b(1, 3) = -1.0330936;
		data_b(2, 3) = -0.87217125;
		feats_b = std::make_shared<DenseFeatures<float64_t>>(data_b);


		ref_cov_a = SGMatrix<float64_t>(dims, dims);
		ref_cov_a(0, 0) = 0.353214;
		ref_cov_a(1, 0) = 0.29906652;
		ref_cov_a(2, 0) = -0.46552636;
		ref_cov_a(0, 1) = 0.29906652;
		ref_cov_a(1, 1) = 0.8914735;
		ref_cov_a(2, 1) = -0.13294825;
		ref_cov_a(0, 2) = -0.46552636;
		ref_cov_a(1, 2) = -0.13294825;
		ref_cov_a(2, 2) = 0.72797476;

		ref_cov_ab = SGMatrix<float64_t>(dims, dims);
		ref_cov_ab(0, 0) = 0.32300248;
		ref_cov_ab(1, 0) = 0.24380185;
		ref_cov_ab(2, 0) = -0.27024556;
		ref_cov_ab(0, 1) = 0.24380185;
		ref_cov_ab(1, 1) = 0.68716546;
		ref_cov_ab(2, 1) = 0.10940845;
		ref_cov_ab(0, 2) = -0.27024556;
		ref_cov_ab(1, 2) = 0.10940845;
		ref_cov_ab(2, 2) = 0.72460503;
	}

	virtual void TearDown()
	{


	}

	const index_t num_a = 5;
	const index_t num_b = 4;
	const index_t dims = 3;
	const float64_t eps = 1e-8;

	std::shared_ptr<DenseFeatures<float64_t>> feats_a;
	std::shared_ptr<DenseFeatures<float64_t>> feats_b;
	SGMatrix<float64_t> ref_cov_a;
	SGMatrix<float64_t> ref_cov_ab;
};

TEST_F(DotFeaturesTest, get_cov)
{
	auto cov = feats_a->DotFeatures::get_cov();

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_a[i], eps);
}

TEST_F(DotFeaturesTest, get_cov_nocopy)
{
	auto cov = feats_a->DotFeatures::get_cov(false);

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_a[i], eps);
}

TEST_F(DotFeaturesTest, compute_cov)
{
	auto cov = DotFeatures::compute_cov(feats_a, feats_b);

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_ab[i], eps);
}

TEST_F(DotFeaturesTest, compute_cov_nocopy)
{
	auto cov = DotFeatures::compute_cov(feats_a, feats_b, false);

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_ab[i], eps);
}

TEST_F(DotFeaturesTest, dense_dot_range)
{
	index_t num_feats = 2;
	index_t num_vectors = 4;
	float64_t bias = 25;

	SGMatrix<float64_t> data(num_feats, num_vectors);
	for (index_t i = 0; i < num_vectors; i++)
		data.get_column(i).set_const(i);
	auto feats = std::make_shared<DenseFeatures<float64_t>>(data);

	SGVector<float64_t> vec(num_feats);
	vec.range_fill(1);

	index_t start1 = 0;
	index_t stop1 = num_vectors - 1;
	index_t start2 = 1;
	index_t stop2 = num_vectors;

	SGVector<float64_t> alphas(num_vectors - 1);
	alphas.range_fill(0);

	SGVector<float64_t> output1(num_vectors - 1);
	SGVector<float64_t> output2(num_vectors - 1);

	feats->dense_dot_range(
	    output1.vector, start1, stop1, alphas.vector, vec.vector, num_feats,
	    bias);
	feats->dense_dot_range(
	    output2.vector, start2, stop2, alphas.vector, vec.vector, num_feats,
	    bias);

	for (index_t i = 0; i < num_vectors - 1; i++)
	{
		// output1[i] = alpha[i] * (data[i] dot vec) + bias
		ASSERT_EQ(
		    output1[i], i * i * num_feats * ((1 + num_feats) / 2.0) + bias);
		// output2[i] = alpha[i] * (data[i+1] dot vec) + bias
		ASSERT_EQ(
		    output2[i],
		    i * (i + 1) * num_feats * ((1 + num_feats) / 2.0) + bias);
	}
}
