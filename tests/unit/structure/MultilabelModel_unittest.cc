/*
 * This software is distributed under BSD Clause 3 license (see LICENSE file).
 *
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2014 Abinash Panda
 */

#include <gtest/gtest.h>
#include <shogun/structure/MultilabelModel.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/structure/MultilabelSOLabels.h>

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

	auto features = std::make_shared<SparseFeatures<float64_t>>(feats);


	auto labels = std::make_shared<MultilabelSOLabels>(2, 3);

	SGVector<int32_t> lab_1(1);
	lab_1[0] = 1;
	SGVector<int32_t> lab_2(3);
	lab_2[0] = 0;
	lab_2[1] = 1;
	lab_2[2] = 2;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_2);

	auto model = std::make_shared<MultilabelModel>(features, labels);


	auto slabel_1 = std::make_shared<SparseMultilabel>(lab_1);

	auto slabel_2 = std::make_shared<SparseMultilabel>(lab_2);

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






}

TEST(MultilabelModel, delta_loss)
{
	SGMatrix<float64_t> feats(DIMS, NUM_SAMPLES);
	feats.zero();

	auto features = std::make_shared<SparseFeatures<float64_t>>(feats);


	auto labels = std::make_shared<MultilabelSOLabels>(2, 3);


	auto model = std::make_shared<MultilabelModel>(features, labels);


	SGVector<int32_t> lab_3(3);
	lab_3[0] = 0;
	lab_3[1] = 1;
	lab_3[2] = 2;
	SGVector<int32_t> lab_4(3);
	lab_4[0] = 0;
	lab_4[1] = 1;
	lab_4[2] = 2;

	auto slabel_3 = std::make_shared<SparseMultilabel>(lab_3);

	auto slabel_4 = std::make_shared<SparseMultilabel>(lab_4);

	float64_t delta_loss_1 = model->delta_loss(slabel_3, slabel_4);
	EXPECT_EQ(delta_loss_1, 0);




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

	auto slabel_5 = std::make_shared<SparseMultilabel>(lab_5);

	auto slabel_6 = std::make_shared<SparseMultilabel>(lab_6);

	float64_t delta_loss_2 = model->delta_loss(slabel_5, slabel_6);
	EXPECT_EQ(delta_loss_2, false_neg);

	float64_t delta_loss_3 = model->delta_loss(slabel_6, slabel_5);
	EXPECT_EQ(delta_loss_3, false_pos);






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

	auto features = std::make_shared<SparseFeatures<float64_t>>(feats);


	auto labels = std::make_shared<MultilabelSOLabels>(2, 3);

	SGVector<int32_t> lab_1(1);
	lab_1[0] = 2;
	SGVector<int32_t> lab_2(2);
	lab_2[0] = 0;
	lab_2[1] = 1;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_2);

	auto model = std::make_shared<MultilabelModel>(features, labels);


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

	auto ret_1 = model->argmax(w, 0, true);
	auto ret_2 = model->argmax(w, 1, true);

	SGVector<int32_t>y_1_expected(2);
	y_1_expected[0] = 0;
	y_1_expected[1] = 2;
	SGVector<int32_t>y_2_expected(2);
	y_2_expected[0] = 0;
	y_2_expected[1] = 1;

	auto y_1 = ret_1->argmax->as<SparseMultilabel>();
	SGVector<int32_t> slabel_1 = y_1->get_data();
	SGVector<float64_t> psi_truth_1 = ret_1->psi_truth;

	for (index_t i = 0; i < slabel_1.vlen; i++)
	{
		EXPECT_EQ(slabel_1[i], y_1_expected[i]);
	}

	for (index_t i = 0; i < psi_truth_1.vlen; i++)
	{
		if (i < 6)
		{
			EXPECT_EQ(psi_truth_1[i], 0);
		}

		else
		{
			EXPECT_EQ(psi_truth_1[i], feats[i % 3]);
		}
	}

	EXPECT_EQ(ret_1->delta, 1);

	auto y_2 = ret_2->argmax->as<SparseMultilabel>();
	SGVector<int32_t> slabel_2 = y_2->get_data();
	SGVector<float64_t> psi_truth_2 = ret_2->psi_truth;

	for (index_t i = 0; i < slabel_2.vlen; i++)
	{
		EXPECT_EQ(slabel_2[i], y_2_expected[i]);
	}

	for (index_t i = 0; i < psi_truth_2.vlen; i++)
	{
		if (i >= 6)
		{
			EXPECT_EQ(psi_truth_2[i], 0);
		}

		else
		{
			EXPECT_EQ(psi_truth_2[i], feats[(i % 3) + 3]);
		}
	}

	EXPECT_EQ(ret_2->delta, 0);

	auto ret_3 = model->argmax(w, 0, false);
	auto ret_4 = model->argmax(w, 1, false);

	auto y_3 = ret_3->argmax->as<SparseMultilabel>();
	SGVector<int32_t> slabel_3 = y_3->get_data();
	SGVector<float64_t> psi_pred_3 = ret_3->psi_pred;

	for (index_t i = 0; i < slabel_3.vlen; i++)
	{
		EXPECT_EQ(slabel_3[i], y_1_expected[i]);
	}

	for (index_t i = 0; i < psi_pred_3.vlen; i++)
	{
		if (i > 2 && i < 6)
		{
			EXPECT_EQ(psi_pred_3[i], 0);
		}

		else
		{
			EXPECT_EQ(psi_pred_3[i], feats[i % 3]);
		}
	}

	auto y_4 = ret_4->argmax->as<SparseMultilabel>();
	SGVector<int32_t> slabel_4 = y_4->get_data();
	SGVector<float64_t> psi_pred_4 = ret_4->psi_pred;

	for (index_t i = 0; i < slabel_4.vlen; i++)
	{
		EXPECT_EQ(slabel_4[i], y_2_expected[i]);
	}

	for (index_t i = 0; i < psi_pred_4.vlen; i++)
	{
		if (i >= 6)
		{
			EXPECT_EQ(psi_pred_4[i], 0);
		}

		else
		{
			EXPECT_EQ(psi_pred_4[i], feats[(i % 3) + 3]);
		}
	}



}

