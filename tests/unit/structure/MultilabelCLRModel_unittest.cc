/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/MultilabelCLRModel.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

#define DIMS 3
#define NUM_SAMPLES 2
#define NUM_CLASSES 3

using namespace shogun;

TEST(MultilabelCLRModel, get_joint_feature_vector_1)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);

	for (index_t i = 0; i < DIMS * NUM_SAMPLES; i++)
	{
		feats[i] = CMath::random(-100, 100);
	}

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(NUM_SAMPLES,
	                NUM_CLASSES);
	SG_REF(labels);
	SGVector<int32_t> lab_1(1);
	lab_1[0] = 1;
	SGVector<int32_t> lab_012(3);
	lab_012[0] = 0;
	lab_012[1] = 1;
	lab_012[2] = 2;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_012);

	CMultilabelCLRModel * model = new CMultilabelCLRModel(features, labels);
	SG_REF(model);

	CSparseMultilabel * slabel_1 = new CSparseMultilabel(lab_1);
	SG_REF(slabel_1);
	SGVector<float64_t> psi_1 = model->get_joint_feature_vector(0,
	                            slabel_1);

	SGVector<float64_t> label_coeffs_1(4);
	label_coeffs_1[0] = -1;
	label_coeffs_1[1] = 1;
	label_coeffs_1[2] = -1;
	label_coeffs_1[3] = 1;

	for (index_t i = 0; i < psi_1.vlen; i++)
	{
		EXPECT_EQ(psi_1[i], label_coeffs_1[i / DIMS] * feats[i % DIMS]);
	}

	SG_UNREF(slabel_1);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(MultilabelCLRModel, get_joint_feature_vector_2)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);

	for (index_t i = 0; i < DIMS * NUM_SAMPLES; i++)
	{
		feats[i] = CMath::random(-100, 100);
	}

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(NUM_SAMPLES,
	                NUM_CLASSES);
	SG_REF(labels);
	SGVector<int32_t> lab_1(1);
	lab_1[0] = 1;
	SGVector<int32_t> lab_012(3);
	lab_012[0] = 0;
	lab_012[1] = 1;
	lab_012[2] = 2;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_012);

	CMultilabelCLRModel * model = new CMultilabelCLRModel(features, labels);
	SG_REF(model);

	CSparseMultilabel * slabel_012 = new CSparseMultilabel(lab_012);
	SG_REF(slabel_012);

	SGVector<float64_t> psi_2 = model->get_joint_feature_vector(1,
	                            slabel_012);

	SGVector<float64_t> label_coeffs_2(4);
	label_coeffs_2[0] = 1;
	label_coeffs_2[1] = 1;
	label_coeffs_2[2] = 1;
	label_coeffs_2[3] = -3;

	for (index_t i = 0; i < psi_2.vlen; i++)
	{
		EXPECT_EQ(psi_2[i], label_coeffs_2[i / DIMS] * feats[i % DIMS + DIMS]);
	}

	SG_UNREF(slabel_012);
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

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(NUM_SAMPLES,
	                NUM_CLASSES);
	SG_REF(labels);

	CMultilabelCLRModel * model = new CMultilabelCLRModel(features, labels);
	SG_REF(model);

	SGVector<int32_t> lab_012(3);
	lab_012[0] = 0;
	lab_012[1] = 1;
	lab_012[2] = 2;
	SGVector<int32_t> lab_01(2);
	lab_01[0] = 0;
	lab_01[1] = 1;
	SGVector<int32_t> lab_0(1);
	lab_0[0] = 0;
	SGVector<int32_t> lab_nill(0);

	CSparseMultilabel * slabel_012 = new CSparseMultilabel(lab_012);
	SG_REF(slabel_012);
	CSparseMultilabel * slabel_01 = new CSparseMultilabel(lab_01);
	SG_REF(slabel_01);
	CSparseMultilabel * slabel_0 = new CSparseMultilabel(lab_0);
	SG_REF(slabel_0);
	CSparseMultilabel * slabel_nill = new CSparseMultilabel(lab_nill);
	SG_REF(slabel_nill);

	float64_t delta_loss_1 = model->delta_loss(slabel_012, slabel_012);
	EXPECT_EQ(delta_loss_1, 0);

	float64_t delta_loss_2 = model->delta_loss(slabel_012, slabel_01);
	EXPECT_EQ(delta_loss_2, 1);

	float64_t delta_loss_3 = model->delta_loss(slabel_012, slabel_0);
	EXPECT_EQ(delta_loss_3, 2);

	float64_t delta_loss_4 = model->delta_loss(slabel_012, slabel_nill);
	EXPECT_EQ(delta_loss_4, 3);

	SG_UNREF(slabel_012);
	SG_UNREF(slabel_01);
	SG_UNREF(slabel_0);
	SG_UNREF(slabel_nill);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(MultilabelCLRModel, argmax)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);

	for (index_t i = 0; i < DIMS * NUM_SAMPLES; i++)
	{
		feats[i] = CMath::random(-100, 100);
	}

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(NUM_SAMPLES,
	                NUM_CLASSES);
	SG_REF(labels);
	SGVector<int32_t> lab_2(1);
	lab_2[0] = 2;
	SGVector<int32_t> lab_01(2);
	lab_01[0] = 0;
	lab_01[1] = 1;
	labels->set_sparse_label(0, lab_2);
	labels->set_sparse_label(1, lab_01);

	CMultilabelCLRModel * model = new CMultilabelCLRModel(features, labels);
	SG_REF(model);

	SGVector<float64_t> w(model->get_dim());

	for (index_t i = 0; i < w.vlen; i++)
	{
		w[i] = CMath::random(-1, 1);
	}

	CResultSet * ret_1 = model->argmax(w, 0, true);

	CSparseMultilabel * y_1 = CSparseMultilabel::obtain_from_generic(
	                                  ret_1->argmax);
	SGVector<int32_t> slabel_1 = y_1->get_data();

	// calibrated/virtual label is considered to be last label
	float64_t calibrated_score = ((CDotFeatures *)features)->dense_dot(0,
	                             w.vector + labels->get_num_classes() * DIMS, DIMS);

	for (index_t i = 0; i < slabel_1.vlen; i++)
	{
		int32_t label = slabel_1[i];
		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + label * DIMS, DIMS) - calibrated_score;

		// true label in this case is lab_2
		if (label != lab_2[0])
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

	for (index_t i = 0; i < labels->get_num_classes(); i++)
	{
		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + i * DIMS, DIMS);

		bool present = false;

		for (index_t j = 0; j < slabel_2.vlen; j++)
		{
			if (i == slabel_2[j])
			{
				present = true;
				break;
			}
		}

		if (present)
		{
			EXPECT_GE(score, calibrated_score);
		}
		else
		{
			EXPECT_GT(calibrated_score, score);
		}
	}

	SG_UNREF(ret_1);
	SG_UNREF(ret_2);

	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(model);
}

