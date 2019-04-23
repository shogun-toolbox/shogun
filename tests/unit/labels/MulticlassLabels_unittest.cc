/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Olivier NGuyen, Sergey Lisitsyn, Viktor Gal,
 *          Bjoern Esser, Thoralf Klein
 */

#include <gtest/gtest.h>
#include <shogun/base/range.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/View.h>

using namespace shogun;

class MulticlassLabelsTest : public ::testing::Test
{
public:
	SGMatrix<float64_t> probabilities;
	SGVector<float64_t> labels_true;
	const index_t n = 3;

	virtual void SetUp()
	{
		probabilities = SGMatrix<float64_t>(n, n);
		probabilities(0, 0) = 0.6;
		probabilities(0, 1) = 0.2;
		probabilities(0, 2) = 0.2;
		probabilities(1, 0) = 0.3;
		probabilities(1, 1) = 0.3;
		probabilities(1, 2) = 0.4;
		probabilities(2, 0) = 0.1;
		probabilities(2, 1) = 0.8;
		probabilities(2, 2) = 0.1;

		labels_true = {0, 2, 1};
	}

	virtual void TearDown()
	{
	}
};

TEST_F(MulticlassLabelsTest, confidences)
{
	const int n_labels = 3;
	const int n_classes = 4;

	auto labels = std::make_shared<MulticlassLabels>(n_labels);

	EXPECT_NO_THROW(labels->allocate_confidences_for(n_classes));

	for (int i=0; i<n_labels; i++)
		EXPECT_EQ(labels->get_multiclass_confidences(i).size(),n_classes);

	for (int i=0; i<n_labels; i++)
	{
		SGVector<float64_t> confs(n_classes);
		confs.zero();
		confs[i % n_classes] = 1.0;
		labels->set_multiclass_confidences(i, confs);

		SGVector<float64_t> obtained_confs = labels->get_multiclass_confidences(i);
		for (int j=0; j<n_classes; j++)
		{
			if (j==i%n_classes)
				EXPECT_NEAR(obtained_confs[j],1.0,1e-9);
			else
				EXPECT_NEAR(obtained_confs[j],0.0,1e-9);
		}
	}
	
}

TEST_F(MulticlassLabelsTest, multiclass_labels_from_multiclass)
{
	auto labels = std::make_shared<MulticlassLabels>(labels_true);
	auto labels2 = multiclass_labels(labels);
	EXPECT_EQ(labels.get(), labels2.get());
}

TEST_F(MulticlassLabelsTest, multiclass_labels_from_binary_not_contiguous)
{
	// delete this test once multiclass labels dont need to be contiguous,
	// i.e. [0,1,2,3,4,...], anymore
	auto labels = std::make_shared<BinaryLabels>(labels_true.size());
	labels->set_labels({0, 1, 3});
	auto converted = multiclass_labels(labels);
	ASSERT_NE(converted.get(), nullptr);
	EXPECT_TRUE(converted->get_labels().equals({0, 1, 2}));

	labels->set_labels({-1, 1, 1});
	auto converted2 = multiclass_labels(labels);
	ASSERT_NE(converted2.get(), nullptr);
	EXPECT_TRUE(converted2->get_labels().equals({0, 1, 1}));
}

TEST_F(MulticlassLabelsTest, view)
{
	auto labels = std::make_shared<MulticlassLabels>(labels_true);
	SGVector<index_t> subset{0, 2};
	auto labels_subset = view(labels, subset);

	ASSERT_EQ(labels_subset->get_num_labels(), subset.vlen);
	for (auto i : range(subset.vlen))
	{
		EXPECT_EQ(labels_subset->get_int_label(i), labels_true[subset[i]]);
	}
}
