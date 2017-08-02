/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Sergey Lisitsyn
 */

#include <shogun/labels/MulticlassLabels.h>
#include <gtest/gtest.h>

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

		SGVector<float64_t> labels_A(3);
		labels_A[0] = 0;
		labels_A[1] = 2;
		labels_A[2] = 1;

		labels_true = labels_A;
	}

	virtual void TearDown()
	{
	}
};

TEST_F(MulticlassLabelsTest, confidences)
{
	const int n_labels = 3;
	const int n_classes = 4;

	CMulticlassLabels* labels = new CMulticlassLabels(n_labels);

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
	SG_UNREF(labels);
}

TEST_F(MulticlassLabelsTest, prob_matrix_label_initialization)
{
	CMulticlassLabels* labels = new CMulticlassLabels(probabilities);
	int32_t n_classes = probabilities.num_cols;
	int32_t n_labels = probabilities.num_rows;

	auto labels_vector = labels->get_labels();
	for (int i = 0; i < n_labels; i++)
	{
		SGVector<float64_t> obtained_confs =
		    labels->get_multiclass_confidences(i);
		for (int j = 0; j < n_classes; j++)
		{
			EXPECT_FLOAT_EQ(probabilities(i, j), obtained_confs[j]);
		}
		EXPECT_FLOAT_EQ(labels_true[i], labels_vector[i]);
	}

	SG_UNREF(labels);
}
