/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Written(W) 2014 Abinash Panda
 * Copyright(C) 2014 Abinash Panda
 */

#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/MultilabelCLRModel.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <gtest/gtest.h>

#define DIMS 3
#define NUM_SAMPLES 2

using namespace shogun;

TEST(MultilabelCLRModel, get_joint_feature_vector)
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

	CMultilabelCLRModel * model = new CMultilabelCLRModel(features, labels);
	SG_REF(model);

	CSparseMultilabel * slabel_1 = new CSparseMultilabel(lab_1);
	SG_REF(slabel_1);
	CSparseMultilabel * slabel_2 = new CSparseMultilabel(lab_2);
	SG_REF(slabel_2);
	SGVector<float64_t> psi_1 = model->get_joint_feature_vector(0,
	                            slabel_1);
	SGVector<float64_t> psi_2 = model->get_joint_feature_vector(1,
	                            slabel_2);

	SGVector<float64_t> label_coeffs_1(4);
	label_coeffs_1[0] = -1;
	label_coeffs_1[1] = 1;
	label_coeffs_1[2] = -1;
	label_coeffs_1[3] = 1;

	for (index_t i = 0; i < psi_1.vlen; i++)
	{
		EXPECT_EQ(psi_1[i], label_coeffs_1[i / 3] * feats[i % 3]);
	}

	SGVector<float64_t> label_coeffs_2(4);
	label_coeffs_2[0] = 1;
	label_coeffs_2[1] = 1;
	label_coeffs_2[2] = 1;
	label_coeffs_2[3] = -3;

	for (index_t i = 0; i < psi_2.vlen; i++)
	{
		EXPECT_EQ(psi_2[i], label_coeffs_2[i / 3] * feats[i % 3 + 3]);
	}

	SG_UNREF(slabel_1);
	SG_UNREF(slabel_2);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(MultilabelCLRModel, delta_loss)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);
	feats.zero();

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(2, 3);
	SG_REF(labels);

	CMultilabelCLRModel * model = new CMultilabelCLRModel(features, labels);
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
	EXPECT_EQ(delta_loss_2, 1);

	SG_UNREF(slabel_6);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(MultilabelCLRModel, argmax)
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

	CMultilabelCLRModel * model = new CMultilabelCLRModel(features, labels);
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
	w[9] = 1;
	w[10] = -1;
	w[11] = 0;

	CResultSet * ret_1 = model->argmax(w, 0, true);

	CSparseMultilabel * y_1 = CSparseMultilabel::obtain_from_generic(
	                                  ret_1->argmax);
	SGVector<int32_t> slabel_1 = y_1->get_data();

	float64_t calibrated_score = ((CDotFeatures *)features)->dense_dot(0,
	                             w.vector + 9, 3);

	for (index_t i = 0; i < slabel_1.vlen; i++)
	{
		int32_t label = slabel_1[i];
		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + label * 3, 3) - calibrated_score;

		if (label != 2)
		{
			score += 1;
		}
		else
		{
			score -= 1;
		}

		EXPECT_GE(score, 0);
	}

	CResultSet * ret_2 = model->argmax(w, 0, false);

	CSparseMultilabel * y_2 = CSparseMultilabel::obtain_from_generic(
	                                  ret_2->argmax);
	SGVector<int32_t> slabel_2 = y_2->get_data();

	for (index_t i = 0; i < slabel_2.vlen; i++)
	{
		int32_t label = slabel_1[i];
		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + label * 3, 3) - calibrated_score;
		EXPECT_GE(score, 0);
	}

	SG_UNREF(ret_1);
	SG_UNREF(ret_2);

	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(model);
}

