/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
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
		feats_a = new CDenseFeatures<float64_t>(data_a);
		SG_REF(feats_a);

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
		feats_b = new CDenseFeatures<float64_t>(data_b);
		SG_REF(feats_b);

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
		SG_UNREF(feats_a);
		SG_UNREF(feats_b);
	}

	const index_t num_a = 5;
	const index_t num_b = 4;
	const index_t dims = 3;
	const float64_t eps = 1e-8;

	CDenseFeatures<float64_t>* feats_a;
	CDenseFeatures<float64_t>* feats_b;
	SGMatrix<float64_t> ref_cov_a;
	SGMatrix<float64_t> ref_cov_ab;
};

TEST_F(DotFeaturesTest, get_cov)
{
	auto cov = feats_a->CDotFeatures::get_cov();

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_a[i], eps);
}

TEST_F(DotFeaturesTest, get_cov_nocopy)
{
	auto cov = feats_a->CDotFeatures::get_cov(false);

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_a[i], eps);
}

TEST_F(DotFeaturesTest, compute_cov)
{
	auto cov = CDotFeatures::compute_cov(feats_a, feats_b);

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_ab[i], eps);
}

TEST_F(DotFeaturesTest, compute_cov_nocopy)
{
	auto cov = CDotFeatures::compute_cov(feats_a, feats_b, false);

	for (index_t i = 0; i < (index_t)cov.size(); ++i)
		EXPECT_NEAR(cov[i], ref_cov_ab[i], eps);
}
