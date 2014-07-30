/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/HierarchicalMultilabelModel.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(HierarchicalMultilabelModel, get_joint_feature_vector_1)
{
	int32_t dim_features = 3;
	int32_t num_samples = 2;

	SGMatrix<float64_t> feats(dim_features, num_samples);

	for (index_t i = 0; i < dim_features * num_samples; i++)
	{
		feats[i] = CMath::random(-100, 100);
	}

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(num_samples, 3);
	SG_REF(labels);
	SGVector<int32_t> lab_1(1);
	lab_1[0] = 1;
	SGVector<int32_t> lab_012(3);
	lab_012[0] = 0;
	lab_012[1] = 1;
	lab_012[2] = 2;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_012);

	// assuming the hierarchy to be
	//       0
	//     1   2
	SGVector<int32_t> taxonomy(3);
	taxonomy[0] = -1;
	taxonomy[1] = 0;
	taxonomy[2] = 0;

	CHierarchicalMultilabelModel * model = new CHierarchicalMultilabelModel(
	        features, labels, taxonomy);
	SG_REF(model);

	CSparseMultilabel * slabel_1 = new CSparseMultilabel(lab_1);
	SG_REF(slabel_1);
	SGVector<float64_t> psi_1 = model->get_joint_feature_vector(0,
	                            slabel_1);

	SGVector<float64_t> attr_vector(3);
	attr_vector[0] = 1;
	attr_vector[1] = 1;
	attr_vector[2] = 0;

	for (index_t i = 0; i < psi_1.vlen; i++)
	{
		EXPECT_EQ(psi_1[i], attr_vector[i / dim_features] * feats[i % dim_features]);
	}

	SG_UNREF(slabel_1);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(HierarchicalMultilabelModel, get_joint_feature_vector_2)
{
	int32_t dim_features = 3;
	int32_t num_samples = 2;

	SGMatrix<float64_t> feats(dim_features, num_samples);

	for (index_t i = 0; i < dim_features * num_samples; i++)
	{
		feats[i] = CMath::random(-100, 100);
	}

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(num_samples, 3);
	SG_REF(labels);
	SGVector<int32_t> lab_1(1);
	lab_1[0] = 1;
	SGVector<int32_t> lab_012(3);
	lab_012[0] = 0;
	lab_012[1] = 1;
	lab_012[2] = 2;
	labels->set_sparse_label(0, lab_1);
	labels->set_sparse_label(1, lab_012);

	// assuming the hierarchy to be
	//       0
	//     1   2
	SGVector<int32_t> taxonomy(3);
	taxonomy[0] = -1;
	taxonomy[1] = 0;
	taxonomy[2] = 0;

	CHierarchicalMultilabelModel * model = new CHierarchicalMultilabelModel(
	        features, labels, taxonomy);
	SG_REF(model);

	CSparseMultilabel * slabel_012 = new CSparseMultilabel(lab_012);
	SG_REF(slabel_012);
	SGVector<float64_t> psi_2 = model->get_joint_feature_vector(0,
	                            slabel_012);

	SGVector<float64_t> attr_vector(3);
	attr_vector[0] = 1;
	attr_vector[1] = 1;
	attr_vector[2] = 1;

	for (index_t i = 0; i < psi_2.vlen; i++)
	{
		EXPECT_EQ(psi_2[i], attr_vector[i / dim_features] * feats[i % dim_features]);
	}

	SG_UNREF(slabel_012);
	SG_UNREF(model);
	SG_UNREF(features);
	SG_UNREF(labels);
}

TEST(HierarchicalMultilabelModel, delta_loss)
{
	int32_t dim_features = 3;
	int32_t num_samples = 2;

	SGMatrix<float64_t> feats(dim_features, num_samples);
	feats.zero();

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(num_samples, 3);
	SG_REF(labels);

	// assuming the hierarchy to be
	//       0
	//     1   2
	SGVector<int32_t> taxonomy(3);
	taxonomy[0] = -1;
	taxonomy[1] = 0;
	taxonomy[2] = 0;

	CHierarchicalMultilabelModel * model = new CHierarchicalMultilabelModel(
	        features, labels, taxonomy);
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

TEST(HierarchicalMultilabelModel, argmax)
{
	int32_t dim_features = 3;
	int32_t num_samples = 2;

	SGMatrix<float64_t> feats(dim_features, num_samples);

	for (index_t i = 0; i < dim_features * num_samples; i++)
	{
		feats[i] = CMath::random(-100, 100);
	}

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(num_samples, 3);
	SG_REF(labels);

	SGVector<int32_t> lab_2(1);
	lab_2[0] = 2;
	SGVector<int32_t> lab_01(2);
	lab_01[0] = 0;
	lab_01[1] = 1;
	labels->set_sparse_label(0, lab_2);
	labels->set_sparse_label(1, lab_01);

	// assuming the hierarchy to be
	//       0
	//     1   2
	SGVector<int32_t> taxonomy(3);
	taxonomy[0] = -1;
	taxonomy[1] = 0;
	taxonomy[2] = 0;

	CHierarchicalMultilabelModel * model = new CHierarchicalMultilabelModel(
	        features, labels, taxonomy);
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

	for (index_t i = 0; i < slabel_1.vlen; i++)
	{
		int32_t label = slabel_1[i];
		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + label * dim_features, dim_features);

		// score for ROOT should be greater than 0
		if (label == 0)
		{
			EXPECT_GE(score, 0);
		}
		// if node is not the ROOT, then along with the score
		// of that node, the score for its parent should also be greater than
		// zero
		else
		{
			float64_t score_0 = ((CDotFeatures *)features)->dense_dot(0,
			                    w.vector, dim_features);
			EXPECT_GE(score_0, 0);
			EXPECT_GE(score, 0);
		}
	}

	CResultSet * ret_2 = model->argmax(w, 0, false);

	CSparseMultilabel * y_2 = CSparseMultilabel::obtain_from_generic(
	                                  ret_2->argmax);
	SGVector<int32_t> slabel_2 = y_2->get_data();

	for (index_t i = 0; i < slabel_2.vlen; i++)
	{
		int32_t label = slabel_2[i];
		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + label * dim_features, dim_features);

		// score for ROOT should be greater than 0
		if (label == 0)
		{
			EXPECT_GE(score, 0);
		}
		// if node is not the ROOT, then along with the score
		// of that node, the score for its parent should also be greater than
		// zero
		else
		{
			float64_t score_0 = ((CDotFeatures *)features)->dense_dot(0,
			                    w.vector, dim_features);
			EXPECT_GE(score_0, 0);
			EXPECT_GE(score, 0);
		}
	}

	SG_UNREF(ret_1);
	SG_UNREF(ret_2);

	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(model);
}

TEST(HierarchicalMultilabelModel, argmax_leaf_nodes_mandatory)
{
	int32_t dim_features = 3;
	int32_t num_samples = 2;

	SGMatrix<float64_t> feats(dim_features, num_samples);

	for (index_t i = 0; i < dim_features * num_samples; i++)
	{
		feats[i] = CMath::random(-100, 100);
	}

	CSparseFeatures<float64_t> * features = new CSparseFeatures<float64_t>(feats);
	SG_REF(features);

	CMultilabelSOLabels * labels = new CMultilabelSOLabels(num_samples, 3);
	SG_REF(labels);

	SGVector<int32_t> lab_2(1);
	lab_2[0] = 2;
	SGVector<int32_t> lab_01(2);
	lab_01[0] = 0;
	lab_01[1] = 1;
	labels->set_sparse_label(0, lab_2);
	labels->set_sparse_label(1, lab_01);

	// assuming the hierarchy to be
	//       0
	//     1   2
	SGVector<int32_t> taxonomy(3);
	taxonomy[0] = -1;
	taxonomy[1] = 0;
	taxonomy[2] = 0;

	CHierarchicalMultilabelModel * model = new CHierarchicalMultilabelModel(
	        features, labels, taxonomy, true);
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

	for (index_t i = 0; i < slabel_1.vlen; i++)
	{
		int32_t label = slabel_1[i];
		// as leaf_node_mandatory flag is set, the output cannot be an internal
		// node (it must be a terminal node)
		EXPECT_NE(label, 0);

		// if node is not the ROOT, then along with the score
		// of that node, the score for its parent should also be greater than
		// zero
		float64_t score_0 = ((CDotFeatures *)features)->dense_dot(0,
		                    w.vector, dim_features);
		EXPECT_GE(score_0, 0);

		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + label * dim_features, dim_features);
		EXPECT_GE(score, 0);
	}

	CResultSet * ret_2 = model->argmax(w, 0, false);

	CSparseMultilabel * y_2 = CSparseMultilabel::obtain_from_generic(
	                                  ret_2->argmax);
	SGVector<int32_t> slabel_2 = y_2->get_data();

	for (index_t i = 0; i < slabel_2.vlen; i++)
	{
		int32_t label = slabel_2[i];
		// as leaf_node_mandatory flag is set, the output cannot be an internal
		// node (it must be a terminal node)
		EXPECT_NE(label, 0);

		// if node is not the ROOT, then along with the score
		// of that node, the score for its parent should also be greater than
		// zero
		float64_t score_0 = ((CDotFeatures *)features)->dense_dot(0,
		                    w.vector, dim_features);
		EXPECT_GE(score_0, 0);

		float64_t score = ((CDotFeatures *)features)->dense_dot(0,
		                  w.vector + label * dim_features, dim_features);
		EXPECT_GE(score, 0);
	}

	SG_UNREF(ret_1);
	SG_UNREF(ret_2);

	SG_UNREF(features);
	SG_UNREF(labels);
	SG_UNREF(model);
}

