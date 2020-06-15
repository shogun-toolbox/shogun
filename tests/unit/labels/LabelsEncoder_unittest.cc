/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */

#include <gtest/gtest.h>
#include <shogun/labels/BinaryLabelEncoder.h>
#include <shogun/labels/LabelEncoder.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/labels/MulticlassLabelsEncoder.h>
using namespace shogun;

TEST(BinaryLabelEncoder, fit_transform)
{
	auto label_encoder = std::make_shared<BinaryLabelEncoder>();
	SGVector<int32_t> vec{-1, -1, 1, -1, 1};
	auto origin_labels = std::make_shared<BinaryLabels>(vec);
	auto unique_vec = label_encoder->fit(origin_labels);
	EXPECT_EQ(-1, unique_vec[0]);
	EXPECT_EQ(1, unique_vec[1]);

	auto result_labels = label_encoder->transform(origin_labels);
	auto result_vec = result_labels->as<BinaryLabels>()->get_labels();
	EXPECT_EQ(-1, result_vec[0]);
	EXPECT_EQ(-1, result_vec[1]);
	EXPECT_EQ(1, result_vec[2]);
	EXPECT_EQ(-1, result_vec[3]);
	EXPECT_EQ(1, result_vec[4]);

	auto inv_result = label_encoder->inverse_transform(result_labels)
	                      ->as<BinaryLabels>()
	                      ->get_labels();

	for (int i = 0; i < 5; i++)
	{
		EXPECT_EQ(vec[i], inv_result[i]);
	}
}

TEST(MulticlassLabelsEncoder, fit_transform)
{
	auto label_encoder = std::make_shared<MulticlassLabelsEncoder>();
	SGVector<float64_t> vec{1, 2, 2, 6};
	auto origin_labels = std::make_shared<MulticlassLabels>(vec);
	auto unique_vec = label_encoder->fit(origin_labels);
	EXPECT_EQ(1, unique_vec[0]);
	EXPECT_EQ(2, unique_vec[1]);
	EXPECT_EQ(6, unique_vec[2]);

	auto result_labels = label_encoder->transform(origin_labels);
	auto result_vec = result_labels->as<MulticlassLabels>()->get_labels();
	EXPECT_EQ(0, result_vec[0]);
	EXPECT_EQ(1, result_vec[1]);
	EXPECT_EQ(1, result_vec[2]);
	EXPECT_EQ(2, result_vec[3]);

	auto inv_result = label_encoder->inverse_transform(result_labels)
	                      ->as<MulticlassLabels>()
	                      ->get_labels();

	for (int i = 0; i < 5; i++)
	{
		EXPECT_EQ(vec[i], inv_result[i]);
	}
}
