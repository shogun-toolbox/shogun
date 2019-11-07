/*
 * This software is distributed under BSD Clause 3 license (see LICENSE file).
 *
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2014 Abinash Panda
 */

#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MultilabelSOLabels, constructor_zero_args)
{
	auto ml = std::make_shared<MultilabelSOLabels>();

	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 0);
	EXPECT_EQ(ml->get_num_classes(), 0);


}

TEST(MultilabelSOLabels, constructor_one_arg)
{
	auto ml = std::make_shared<MultilabelSOLabels>(2);

	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 0);
	EXPECT_EQ(ml->get_num_classes(), 2);


}

TEST(MultilabelSOLabels, constructor_two_args)
{
	auto ml = std::make_shared<MultilabelSOLabels>(5, 6);

	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 5);
	EXPECT_EQ(ml->get_num_classes(), 6);


}

TEST(MultilabelSOLabels, set_sparse_label)
{
	auto ml = std::make_shared<MultilabelSOLabels>(1, 2);

	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 1);
	EXPECT_EQ(ml->get_num_classes(), 2);

	SGVector<int32_t> lab(2);
	lab[0] = 0;
	lab[1] = 1;
	ml->set_sparse_label(0, lab);

	auto  slabel = ml->get_label(0)->as<SparseMultilabel>();
	SGVector<int32_t> slabel_data = slabel->get_data();

	for (int i = 0; i < slabel_data.vlen; i++)
	{
		EXPECT_EQ(lab[i], slabel_data[i]);
	}

}

TEST(MultilabelSOLabels, set_label)
{
	auto ml = std::make_shared<MultilabelSOLabels>(1, 2);

	ASSERT(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 1);
	EXPECT_EQ(ml->get_num_classes(), 2);

	SGVector<int32_t> lab(2);
	lab[0] = 0;
	lab[1] = 1;
	auto  slabel = std::make_shared<SparseMultilabel>(lab);

	ml->set_label(0, slabel);

	auto  slabel_out = ml->get_label(0)->as<SparseMultilabel>();
	SGVector<int32_t> slabel_data = slabel_out->get_data();

	for (index_t i = 0; i < slabel_data.vlen; i++)
	{
		EXPECT_EQ(lab[i], slabel_data[i]);
	}



}

TEST(MultilabelSOLabels, set_sparse_labels)
{
	auto ml = std::make_shared<MultilabelSOLabels>(2, 3);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 2);
	EXPECT_EQ(ml->get_num_classes(), 3);

	SGVector<int32_t> * labels = new SGVector<int32_t>[2];
	SGVector<int32_t> lab1(2);
	lab1[0] = 0;
	lab1[1] = 2;
	SGVector<int32_t> lab2(3);
	lab2[0] = 0;
	lab2[1] = 1;
	lab2[2] = 2;
	labels[0] = lab1;
	labels[1] = lab2;

	ml->set_sparse_labels(labels);

	for (int i = 0; i < ml->get_num_labels(); i++)
	{
		auto  slabel = ml->get_label(i)->as<SparseMultilabel>();
		SGVector<int32_t> slabel_data = slabel->get_data();
		SGVector<int32_t> lab = labels[i];
		EXPECT_EQ(slabel_data.vlen, lab.vlen);

		for (int j = 0; i < slabel_data.vlen; i++)
		{
			EXPECT_EQ(lab[j], slabel_data[j]);
		}

	}

	delete [] labels;

}

TEST(MultilabelSOLabels, to_dense)
{
	auto ml = std::make_shared<MultilabelSOLabels>(2, 3);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 2);
	EXPECT_EQ(ml->get_num_classes(), 3);

	SGVector<int32_t> * labels = new SGVector<int32_t>[2];
	SGVector<int32_t> lab1(2);
	lab1[0] = 0;
	lab1[1] = 2;
	SGVector<int32_t> lab2(3);
	lab2[0] = 0;
	lab2[1] = 1;
	lab2[2] = 2;
	labels[0] = lab1;
	labels[1] = lab2;

	ml->set_sparse_labels(labels);

	for (int i = 0; i < ml->get_num_labels(); i++)
	{
		auto  slabel = ml->get_label(0)->as<SparseMultilabel>();
		SGVector<float64_t> slabel_dense_data = MultilabelSOLabels::to_dense(slabel, ml->get_num_classes(), 1, 0);
		EXPECT_EQ(slabel_dense_data.vlen, ml->get_num_classes());
	}

	delete[] labels;

}

