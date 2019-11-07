/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written (W) 2014 Abinash Panda
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/structure/MultilabelModel.h>
#include <shogun/structure/MultilabelSOLabels.h>

#include <utility>

using namespace shogun;

MultilabelModel::MultilabelModel()
	: StructuredModel()
{
	init();
}

MultilabelModel::MultilabelModel(std::shared_ptr<Features > features, std::shared_ptr<StructuredLabels > labels)
	: StructuredModel(std::move(features), std::move(labels))
{
	init();
}

MultilabelModel::~MultilabelModel()
{
}

std::shared_ptr<StructuredLabels > MultilabelModel::structured_labels_factory(int32_t num_labels)
{
	return std::make_shared<MultilabelSOLabels>(num_labels, m_num_classes);
}

void MultilabelModel::init()
{
	SG_ADD(&m_false_positive, "false_positive", "Misclassification cost for false positive");
	SG_ADD(&m_false_negative, "false_negative", "Misclassification cost for false negative");
	SG_ADD(&m_num_classes, "num_classes", "Number of (binary) class assignment per label");
	m_false_positive = 1;
	m_false_negative = 1;
	m_num_classes = 0;
}

int32_t MultilabelModel::get_dim() const
{
	int32_t num_classes = m_labels->as<MultilabelSOLabels>()->get_num_classes();
	int32_t feats_dim = m_features->as<DotFeatures>()->get_dim_feature_space();

	return feats_dim * num_classes;
}

void MultilabelModel::set_misclass_cost(float64_t false_positive, float64_t false_negative)
{
	m_false_positive = false_positive;
	m_false_negative = false_negative;
}

SGVector<float64_t> MultilabelModel::get_joint_feature_vector(int32_t feat_idx,
                std::shared_ptr<StructuredData > y)
{
	SGVector<float64_t> psi(get_dim());
	psi.zero();

	SGVector<float64_t> x = m_features->as<DotFeatures>()->
	                        get_computed_dot_feature_vector(feat_idx);
	auto slabel = y->as<SparseMultilabel>();
	ASSERT(slabel != NULL);
	SGVector<int32_t> slabel_data = slabel->get_data();

	for (index_t i = 0; i < slabel_data.vlen; i++)
	{
		for (index_t j = 0, k = slabel_data[i] * x.vlen; j < x.vlen; j++, k++)
		{
			psi[k] = x[j];
		}
	}

	return psi;
}

float64_t MultilabelModel::delta_loss(std::shared_ptr<StructuredData > y1, std::shared_ptr<StructuredData > y2)
{
	auto y1_slabel = y1->as<SparseMultilabel>();
	auto y2_slabel = y2->as<SparseMultilabel>();

	ASSERT(y1_slabel != NULL);
	ASSERT(y2_slabel != NULL);

	auto multi_labels = m_labels->as<MultilabelSOLabels>();
	return delta_loss(
	               MultilabelSOLabels::to_dense(y1_slabel,
	                               multi_labels->get_num_classes(), 1, 0),
	               MultilabelSOLabels::to_dense(y2_slabel,
	                               multi_labels->get_num_classes(), 1, 0));
}

float64_t MultilabelModel::delta_loss(SGVector<float64_t> y1, SGVector<float64_t> y2)
{
	require(y1.vlen == y2.vlen, "Size of both the vectors should be same");

	float64_t loss = 0;

	for (index_t i = 0; i < y1.vlen; i++)
	{
		loss += delta_loss(y1[i], y2[i]);
	}

	return loss;
}

float64_t MultilabelModel::delta_loss(float64_t y1, float64_t y2)
{
	return y1 > y2 ? m_false_negative : y1 < y2 ? m_false_positive : 0;
}

SGVector<int32_t> MultilabelModel::to_sparse(SGVector<float64_t> dense_vec,
                float64_t d_true, float64_t d_false)
{
	int32_t size = 0;

	for (index_t i = 0; i < dense_vec.vlen; i++)
	{
		require(dense_vec[i] == d_true || dense_vec[i] == d_false,
		        "The values of dense vector should be either ({}) or ({}).",
		        d_true, d_false);

		if (dense_vec[i] == d_true)
		{
			size++;
		}
	}

	SGVector<int32_t> sparse_vec(size);
	index_t j = 0;

	for (index_t i = 0; i < dense_vec.vlen; i++)
	{
		if (dense_vec[i] == d_true)
		{
			sparse_vec[j] = i;
			j++;
		}
	}

	return sparse_vec;
}

std::shared_ptr<ResultSet > MultilabelModel::argmax(SGVector<float64_t> w, int32_t feat_idx,
                                      bool const training)
{
	auto dot_feats = m_features->as<DotFeatures>();
	int32_t feats_dim = dot_feats->get_dim_feature_space();

	auto multi_labs = m_labels->as<MultilabelSOLabels>();

	if (training)
	{
		m_num_classes = multi_labs->get_num_classes();
	}
	else
	{
		require(m_num_classes > 0, "The model needs to be trained before using "
		        "it for prediction");
	}

	int32_t dim = get_dim();
	ASSERT(dim == w.vlen);

	float64_t score = 0, total_score = 0;
	SGVector<float64_t> y_pred_dense(m_num_classes);
	y_pred_dense.zero();

	for (int32_t c = 0; c < m_num_classes; c++)
	{
		score = dot_feats->dot(feat_idx, w.slice(c * feats_dim, c * feats_dim + feats_dim));

		if (score > 0)
		{
			y_pred_dense[c] = 1;
			total_score += score;
		}

	}

	SGVector<int32_t> y_pred_sparse = to_sparse(y_pred_dense, 1, 0);

	auto ret = std::make_shared<ResultSet>();

	ret->psi_computed = true;

	auto y_pred = std::make_shared<SparseMultilabel>(y_pred_sparse);


	ret->psi_pred = get_joint_feature_vector(feat_idx, y_pred);
	ret->score = total_score;
	ret->argmax = y_pred;

	if (training)
	{
		ret->delta = StructuredModel::delta_loss(feat_idx, y_pred);
		ret->psi_truth = StructuredModel::get_joint_feature_vector(
		                         feat_idx, feat_idx);
		ret->score += (ret->delta - linalg::dot(w, ret->psi_truth));
	}

	return ret;
}

void MultilabelModel::init_primal_opt(
        float64_t regularization,
        SGMatrix<float64_t> &A,
        SGVector<float64_t> a,
        SGMatrix<float64_t> B,
        SGVector<float64_t> &b,
        SGVector<float64_t> &lb,
        SGVector<float64_t> &ub,
        SGMatrix<float64_t> &C)
{
	C = SGMatrix<float64_t>::create_identity_matrix(get_dim(), regularization);
}

