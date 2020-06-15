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

TEST(BinaryLabelEncoder, labels_not_neg1_or_1)
{
	auto label_encoder = std::make_shared<BinaryLabelEncoder>();
	SGVector<int32_t> vec{-100, 200, -100, 200, -100};
	auto origin_labels = std::make_shared<BinaryLabels>(vec);
	auto unique_vec = label_encoder->fit(origin_labels);
	EXPECT_EQ(-100, unique_vec[0]);
	EXPECT_EQ(200, unique_vec[1]);

	auto result_labels = label_encoder->transform(origin_labels);
	auto result_vec = result_labels->as<BinaryLabels>()->get_labels();
	EXPECT_EQ(-1, result_vec[0]);
	EXPECT_EQ(1, result_vec[1]);
	EXPECT_EQ(-1, result_vec[2]);
	EXPECT_EQ(1, result_vec[3]);
	EXPECT_EQ(-1, result_vec[4]);

	auto inv_result = label_encoder->inverse_transform(result_labels)
	                      ->as<DenseLabels>()
	                      ->get_labels();

	for (int i = 0; i < 5; i++)
	{
		EXPECT_EQ(vec[i], inv_result[i]);
	}

	SGVector<int32_t> test_vec{-1, -1, -1, -1, -1, 1};
	auto test_labels = std::make_shared<BinaryLabels>(test_vec);
	auto inv_test = label_encoder->inverse_transform(test_labels)
	                    ->as<BinaryLabels>()
	                    ->get_labels();
	EXPECT_EQ(-100, inv_test[0]);
	EXPECT_EQ(-100, inv_test[1]);
	EXPECT_EQ(-100, inv_test[2]);
	EXPECT_EQ(-100, inv_test[3]);
	EXPECT_EQ(-100, inv_test[4]);
	EXPECT_EQ(200, inv_test[5]);
}

TEST(BinaryLabelEncoder, more_than_two_labels)
{
	auto label_encoder = std::make_shared<BinaryLabelEncoder>();
	SGVector<int32_t> vec{-100, 200, -100, 200, -100, 42};
	auto origin_labels = std::make_shared<BinaryLabels>(vec);

	EXPECT_THROW(label_encoder->fit(origin_labels), ShogunException);

	EXPECT_THROW(label_encoder->transform(origin_labels), ShogunException);

	SGVector<int32_t> vec2{-1, -1, 1, 0};
	auto result_labels = std::make_shared<BinaryLabels>(vec2);
	EXPECT_THROW(
	    label_encoder->inverse_transform(result_labels), ShogunException);

	SGVector<int32_t> vec3{0, 1, 1, 0};
	auto result_labels2 = std::make_shared<BinaryLabels>(vec3);
	EXPECT_THROW(
	    label_encoder->inverse_transform(result_labels2), ShogunException);
}

TEST(MulticlassLabelsEncoder, fit_transform)
{
    auto eps = std::numeric_limits<float64_t>::epsilon();
	auto label_encoder = std::make_shared<MulticlassLabelsEncoder>();
	SGVector<float64_t> vec{1.0, 2.0, 2.0, 6.0};
	auto origin_labels = std::make_shared<MulticlassLabels>(vec);
	auto unique_vec = label_encoder->fit(origin_labels);
	EXPECT_NEAR(1, unique_vec[0], eps);
	EXPECT_NEAR(2, unique_vec[1], eps);
	EXPECT_NEAR(6, unique_vec[2], eps);

	auto result_labels = label_encoder->transform(origin_labels);
	auto result_vec = result_labels->as<MulticlassLabels>()->get_labels();
	EXPECT_NEAR(0, result_vec[0], eps);
	EXPECT_NEAR(1, result_vec[1], eps);
	EXPECT_NEAR(1, result_vec[2], eps);
	EXPECT_NEAR(2, result_vec[3], eps);

	auto inv_result = label_encoder->inverse_transform(result_labels)
	                      ->as<MulticlassLabels>()
	                      ->get_labels();

	for (int i = 0; i < 4; i++)
	{
		EXPECT_NEAR(vec[i], inv_result[i], eps);
	}
}

TEST(MulticlassLabelsEncoder, negative_labels)
{
    auto eps = std::numeric_limits<float64_t>::epsilon();
	auto label_encoder = std::make_shared<MulticlassLabelsEncoder>();
	SGVector<float64_t> vec{-100, 200, -2, 6, -2};
	auto origin_labels = std::make_shared<MulticlassLabels>(vec);
	auto unique_vec = label_encoder->fit(origin_labels);
	EXPECT_NEAR(-100, unique_vec[0], eps);
	EXPECT_NEAR(-2, unique_vec[1], eps);
	EXPECT_NEAR(6, unique_vec[2], eps);
	EXPECT_NEAR(200, unique_vec[3], eps);

	auto result_labels = label_encoder->transform(origin_labels);
	auto result_vec = result_labels->as<MulticlassLabels>()->get_labels();
	EXPECT_NEAR(0, result_vec[0], eps);
	EXPECT_NEAR(3, result_vec[1], eps);
	EXPECT_NEAR(1, result_vec[2], eps);
	EXPECT_NEAR(2, result_vec[3], eps);
	EXPECT_NEAR(1, result_vec[4], eps);

	auto inv_result = label_encoder->inverse_transform(result_labels)
	                      ->as<MulticlassLabels>()
	                      ->get_labels();

	for (int i = 0; i < 5; i++)
	{
		EXPECT_NEAR(vec[i], inv_result[i], eps);
	}

	SGVector<float64_t> test_vec{0, 1, 2, 3, 1, 3};
	auto test_labels = std::make_shared<MulticlassLabels>(test_vec);
	auto inv_test = label_encoder->inverse_transform(test_labels)
	                    ->as<MulticlassLabels>()
	                    ->get_labels();
	EXPECT_NEAR(-100, inv_test[0], eps);
	EXPECT_NEAR(-2, inv_test[1], eps);
	EXPECT_NEAR(6, inv_test[2], eps);
	EXPECT_NEAR(200, inv_test[3], eps);
	EXPECT_NEAR(-2, inv_test[4], eps);
	EXPECT_NEAR(200, inv_test[5], eps);
}
