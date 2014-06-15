/*
 * This software is distributed under BSD Clause 3 license (see LICENSE file).
 *
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2014 Abinash Panda
 */

#include <shogun/structure/MultilabelModel.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <gtest/gtest.h>

#define DIMS 3
#define NUM_SAMPLES 2

using namespace shogun;

TEST(MultilabelModel, get_joint_feature_vector)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);
	feats[0] = 1;
	feats[1] = 0;
	feats[2] = 2;
	feats[3] = 4;
	feats[4] = 5;
	feats[5] = 6;

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(2, 3);
	SG_REF(labels);
	SGVector<int32_t> lab_1(1);
	lab_1[0] = 1;
	SGVector<int32_t> lab_2(3);
	lab_2[0] = 0;
	lab_2[1] = 1;
	lab_2[2] = 2;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_2);

	CMultilabelModel * model = new CMultilabelModel(features, labels);
	SG_REF(model);

	CSparseMultilabel * slabel_1 = new CSparseMultilabel(lab_1);
	SG_REF(slabel_1);
	CSparseMultilabel * slabel_2 = new CSparseMultilabel(lab_2);
	SG_REF(slabel_2);
	SGVector<float64_t> psi_1 = model->get_joint_feature_vector(0,
	                            slabel_1);
	SGVector<float64_t> psi_2 = model->get_joint_feature_vector(1,
	                            slabel_2);

	for (index_t i = 0; i < psi_1.vlen; i++)
	{
		if (i < 3 || i > 5)
		{
			EXPECT_EQ(psi_1[i], 0);
		}
		else
		{
			EXPECT_EQ(psi_1[i], feats[i - 3]);
		}
	}

	for (index_t i = 0; i < psi_2.vlen; i++)
	{
		EXPECT_EQ(psi_2[i], feats[(i % 3) + 3]);
	}

	SG_UNREF(slabel_1);
	SG_UNREF(slabel_2);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(MultilabelModel, delta_loss)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);
	feats.zero();

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(2, 3);
	SG_REF(labels);

	CMultilabelModel * model = new CMultilabelModel(features, labels);
	SG_REF(model);

	SGVector<int32_t> lab_3(3);
	lab_3[0] = 0;
	lab_3[1] = 1;
	lab_3[2] = 2;
	SGVector<int32_t> lab_4(3);
	lab_4[0] = 0;
	lab_4[1] = 1;
	lab_4[2] = 2;

	CSparseMultilabel * slabel_3 = new CSparseMultilabel(lab_3);
	SG_REF(slabel_3);
	CSparseMultilabel * slabel_4 = new CSparseMultilabel(lab_4);
	SG_REF(slabel_4);
	float64_t delta_loss_1 = model->delta_loss(slabel_3, slabel_4);
	EXPECT_EQ(delta_loss_1, 0);

	SG_UNREF(slabel_3);
	SG_UNREF(slabel_4);

	float64_t false_pos = 1;
	float64_t false_neg = 2;
	model->set_misclass_cost(false_pos, false_neg);

	SGVector<int32_t> lab_5(3);
	lab_5[0] = 0;
	lab_5[1] = 1;
	lab_5[2] = 2;
	SGVector<int32_t> lab_6(2);
	lab_6[0] = 0;
	lab_6[1] = 1;

	CSparseMultilabel * slabel_5 = new CSparseMultilabel(lab_5);
	SG_REF(slabel_5);
	CSparseMultilabel * slabel_6 = new CSparseMultilabel(lab_6);
	SG_REF(slabel_6);
	float64_t delta_loss_2 = model->delta_loss(slabel_5, slabel_6);
	EXPECT_EQ(delta_loss_2, false_neg);

	float64_t delta_loss_3 = model->delta_loss(slabel_6, slabel_5);
	EXPECT_EQ(delta_loss_3, false_pos);

	SG_UNREF(slabel_5);
	SG_UNREF(slabel_6);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(MultilabelModel, argmax)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);
	feats[0] = 1;
	feats[1] = 0;
	feats[2] = 2;
	feats[3] = 6;
	feats[4] = 5;
	feats[5] = 4;

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(2, 3);
	SG_REF(labels);
	SGVector<int32_t> lab_1(1);
	lab_1[0] = 2;
	SGVector<int32_t> lab_2(2);
	lab_2[0] = 0;
	lab_2[1] = 1;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_2);

	CMultilabelModel * model = new CMultilabelModel(features, labels);
	SG_REF(model);

	SGVector<float64_t> w(model->get_dim());
	w[0] = 1;
	w[1] = -1;
	w[2] = 0;
	w[3] = 1;
	w[4] = 1;
	w[5] = -1;
	w[6] = 0;
	w[7] = -1;
	w[8] = 1;

	CResultSet * ret_1 = model->argmax(w, 0, true);
	CResultSet * ret_2 = model->argmax(w, 1, true);

	SGVector<int32_t>y_1_expected(2);
	y_1_expected[0] = 0;
	y_1_expected[1] = 2;
	SGVector<int32_t>y_2_expected(2);
	y_2_expected[0] = 0;
	y_2_expected[1] = 1;

	CSparseMultilabel * y_1 = CSparseMultilabel::obtain_from_generic(
	                                  ret_1->argmax);
	SGVector<int32_t> slabel_1 = y_1->get_data();

	EXPECT_EQ(ret_1->delta, 1);

	CSparseMultilabel * y_2 = CSparseMultilabel::obtain_from_generic(
	                                  ret_2->argmax);
	SGVector<int32_t> slabel_2 = y_2->get_data();

	EXPECT_EQ(ret_2->delta, 0);

	CResultSet * ret_3 = model->argmax(w, 0, false);
	CResultSet * ret_4 = model->argmax(w, 1, false);

	CSparseMultilabel * y_3 = CSparseMultilabel::obtain_from_generic(
	                                  ret_3->argmax);
	SGVector<int32_t> slabel_3 = y_3->get_data();

	CSparseMultilabel * y_4 = CSparseMultilabel::obtain_from_generic(
	                                  ret_4->argmax);
	SGVector<int32_t> slabel_4 = y_4->get_data();

	SG_UNREF(ret_1);
	SG_UNREF(ret_2);

	SG_UNREF(ret_3);
	SG_UNREF(ret_4);

	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(model);
}

