/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <gtest/gtest.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/structure/HashedMultilabelModel.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/lib/Hash.h>
#include <shogun/io/SGIO.h>

#define DIMS 3
#define NUM_EXAMPLES 2
#define NUM_CLASSES 3
#define HASH_DIMS 2

using namespace shogun;

TEST(HashedMultilabelModel, get_joint_feature_vector)
{
	SGMatrix<float64_t> feats(DIMS, NUM_EXAMPLES);
	feats[0] = 1;
	feats[1] = 0;
	feats[2] = 2;
	feats[3] = 4;
	feats[4] = 5;
	feats[5] = 6;

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(NUM_EXAMPLES,
	                NUM_CLASSES);
	SG_REF(labels);
	SGVector<int32_t> lab_1(1);
	lab_1[0] = 1;
	SGVector<int32_t> lab_2(3);
	lab_2[0] = 0;
	lab_2[1] = 1;
	lab_2[2] = 2;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_2);

	CHashedMultilabelModel * model = new CHashedMultilabelModel(features,
	                labels, HASH_DIMS);
	SG_REF(model);

	CSparseMultilabel * slabel_1 = new CSparseMultilabel(lab_1);
	SG_REF(slabel_1);
	CSparseMultilabel * slabel_2 = new CSparseMultilabel(lab_2);
	SG_REF(slabel_2);
	SGSparseVector<float64_t> psi_1 = model->get_sparse_joint_feature_vector(0,
	                                  slabel_1);
	SGSparseVector<float64_t> psi_2 = model->get_sparse_joint_feature_vector(1,
	                                  slabel_2);

	SGSparseVector<float64_t> expected_psi_1(lab_1.vlen * HASH_DIMS);
	index_t k = 0;

	SGSparseVector<float64_t> feat_1 = features->get_sparse_feature_vector(0);

	for (index_t i = 0; i < lab_1.vlen; i++)
	{
		uint32_t seed = (uint32_t)lab_1[i];

		for (int32_t j = 0; j < feat_1.num_feat_entries; j++)
		{
			uint32_t hash = CHash::MurmurHash3(
			                        (uint8_t *)&feat_1.features[j].feat_index,
			                        sizeof(index_t), seed);
			expected_psi_1.features[k].feat_index = (hash >> 1) % HASH_DIMS;
			expected_psi_1.features[k++].entry =
			        (hash % 2 == 1 ? -1.0 : 1.0) * feat_1.features[j].entry;
		}
	}

	expected_psi_1.sort_features(true);

	EXPECT_EQ(psi_1.num_feat_entries, expected_psi_1.num_feat_entries);

	for (index_t i = 0; i < psi_1.num_feat_entries; i++)
	{
		EXPECT_EQ(psi_1.features[i].feat_index, expected_psi_1.features[i].feat_index);
		EXPECT_NEAR(psi_1.features[i].entry, expected_psi_1.features[i].entry,
		            1E-7);
	}

	SG_UNREF(slabel_1);
	SG_UNREF(slabel_2);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(HashedMultilabelModel, delta_loss)
{
	SGMatrix<float64_t> feats(DIMS, NUM_EXAMPLES);
	feats.zero();

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(2, 3);
	SG_REF(labels);

	CHashedMultilabelModel * model = new CHashedMultilabelModel(features,
	                labels, HASH_DIMS);
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

TEST(HashedMultilabelModel, argmax)
{
	SGMatrix<float64_t> feats(DIMS, NUM_EXAMPLES);
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

	CHashedMultilabelModel * model = new CHashedMultilabelModel(features,
	                labels, HASH_DIMS);
	SG_REF(model);

	SGVector<float64_t> w(model->get_dim());
	w[0] = 1;
	w[1] = -1;

	CResultSet * ret_1 = model->argmax(w, 0, true);

	CSparseMultilabel * y_1 = CSparseMultilabel::obtain_from_generic(
	                                  ret_1->argmax);
	SGVector<int32_t> slabel_1 = y_1->get_data();


	SGSparseVector<float64_t> feat_1 = features->get_sparse_feature_vector(0);

	for (index_t i = 0; i < slabel_1.vlen; i++)
	{
		uint32_t seed = (uint32_t)slabel_1[i];
		SGSparseVector<float64_t> h_vec(feat_1.num_feat_entries);

		for (int32_t j = 0; j < feat_1.num_feat_entries; j++)
		{
			uint32_t hash = CHash::MurmurHash3(
			                        (uint8_t *)&feat_1.features[j].feat_index,
			                        sizeof(index_t), seed);
			h_vec.features[j].feat_index = (hash >> 1) % HASH_DIMS;
			h_vec.features[j].entry =
			        (hash % 2 == 1 ? -1.0 : 1.0) * feat_1.features[j].entry;
		}

		h_vec.sort_features(true);

		float64_t score = h_vec.dense_dot(1, w.vector, w.vlen, 0);

		if (slabel_1[i] != lab_1[0])
		{
			score += 1;
		}

		EXPECT_GT(score, 0);
	}

	CResultSet * ret_2 = model->argmax(w, 0, false);

	CSparseMultilabel * y_2 = CSparseMultilabel::obtain_from_generic(
	                                  ret_2->argmax);
	SGVector<int32_t> slabel_2 = y_2->get_data();

	for (index_t i = 0; i < slabel_2.vlen; i++)
	{
		uint32_t seed = (uint32_t)slabel_1[i];
		SGSparseVector<float64_t> h_vec(feat_1.num_feat_entries);

		for (int32_t j = 0; j < feat_1.num_feat_entries; j++)
		{
			uint32_t hash = CHash::MurmurHash3(
			                        (uint8_t *)&feat_1.features[j].feat_index,
			                        sizeof(index_t), seed);
			h_vec.features[j].feat_index = (hash >> 1) % HASH_DIMS;
			h_vec.features[j].entry =
			        (hash % 2 == 1 ? -1.0 : 1.0) * feat_1.features[j].entry;
		}

		h_vec.sort_features(true);

		float64_t score = h_vec.dense_dot(1, w.vector, w.vlen, 0);

		EXPECT_GT(score, 0);
	}


	SG_UNREF(ret_1);
	SG_UNREF(ret_2);

	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(model);
}

