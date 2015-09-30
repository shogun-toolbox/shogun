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
	CMultilabelSOLabels * ml = new CMultilabelSOLabels();
	SG_REF(ml);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 0);
	EXPECT_EQ(ml->get_num_classes(), 0);

	SG_UNREF(ml);
}

TEST(MultilabelSOLabels, constructor_one_arg)
{
	CMultilabelSOLabels * ml = new CMultilabelSOLabels(2);
	SG_REF(ml);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 0);
	EXPECT_EQ(ml->get_num_classes(), 2);

	SG_UNREF(ml);
}

TEST(MultilabelSOLabels, constructor_two_args)
{
	CMultilabelSOLabels * ml = new CMultilabelSOLabels(5, 6);
	SG_REF(ml);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 5);
	EXPECT_EQ(ml->get_num_classes(), 6);

	SG_UNREF(ml);
}

TEST(MultilabelSOLabels, set_sparse_label)
{
	CMultilabelSOLabels * ml = new CMultilabelSOLabels(1, 2);
	SG_REF(ml);
	ASSERT_TRUE(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 1);
	EXPECT_EQ(ml->get_num_classes(), 2);

	SGVector<int32_t> lab(2);
	lab[0] = 0;
	lab[1] = 1;
	ml->set_sparse_label(0, lab);

	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(
	                                     ml->get_label(0));
	SGVector<int32_t> slabel_data = slabel->get_data();

	for (int i = 0; i < slabel_data.vlen; i++)
	{
		EXPECT_EQ(lab[i], slabel_data[i]);
	}

	SG_UNREF(slabel);
	SG_UNREF(ml);
}

TEST(MultilabelSOLabels, set_label)
{
	CMultilabelSOLabels * ml = new CMultilabelSOLabels(1, 2);
	SG_REF(ml);
	ASSERT(ml != NULL);

	EXPECT_EQ(ml->get_num_labels(), 1);
	EXPECT_EQ(ml->get_num_classes(), 2);

	SGVector<int32_t> lab(2);
	lab[0] = 0;
	lab[1] = 1;
	CSparseMultilabel * slabel = new CSparseMultilabel(lab);
	SG_REF(slabel);
	ml->set_label(0, slabel);

	CSparseMultilabel * slabel_out = CSparseMultilabel::obtain_from_generic(
	                ml->get_label(0));
	SGVector<int32_t> slabel_data = slabel_out->get_data();

	for (index_t i = 0; i < slabel_data.vlen; i++)
	{
		EXPECT_EQ(lab[i], slabel_data[i]);
	}

	SG_UNREF(slabel);
	SG_UNREF(slabel_out);
	SG_UNREF(ml);
}

TEST(MultilabelSOLabels, set_sparse_labels)
{
	CMultilabelSOLabels * ml = new CMultilabelSOLabels(2, 3);
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
		CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(ml->get_label(i));
		SGVector<int32_t> slabel_data = slabel->get_data();
		SGVector<int32_t> lab = labels[i];
		EXPECT_EQ(slabel_data.vlen, lab.vlen);

		for (int j = 0; i < slabel_data.vlen; i++)
		{
			EXPECT_EQ(lab[j], slabel_data[j]);
		}

		SG_UNREF(slabel);
	}

	delete [] labels;
	SG_UNREF(ml);
}

TEST(MultilabelSOLabels, to_dense)
{
	CMultilabelSOLabels * ml = new CMultilabelSOLabels(2, 3);
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
		CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(ml->get_label(0));
		SGVector<float64_t> slabel_dense_data = CMultilabelSOLabels::to_dense(slabel, ml->get_num_classes(), 1, 0);
		SG_UNREF(slabel);
		EXPECT_EQ(slabel_dense_data.vlen, ml->get_num_classes());
	}

	delete[] labels;
	SG_UNREF(ml);
}

