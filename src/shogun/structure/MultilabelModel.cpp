/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written (W) 2014 Abinash Panda
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/structure/MultilabelModel.h>
#include <shogun/structure/MultilabelSOLabels.h>

using namespace shogun;

CMultilabelModel::CMultilabelModel()
	: CStructuredModel()
{
	init();
}

CMultilabelModel::CMultilabelModel(CFeatures * features, CStructuredLabels * labels)
	: CStructuredModel(features, labels)
{
	init();
}

CMultilabelModel::~CMultilabelModel()
{
}

CStructuredLabels * CMultilabelModel::structured_labels_factory(int32_t num_labels)
{
	return new CMultilabelSOLabels(num_labels, m_num_classes);
}

void CMultilabelModel::init()
{
	SG_ADD(&m_false_positive, "false_positive", "Misclassification cost for false positive",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_false_negative, "false_negative", "Misclassification cost for false negative",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_num_classes, "num_classes", "Number of (binary) class assignment per label",
	       MS_NOT_AVAILABLE);
	m_false_positive = 1;
	m_false_negative = 1;
	m_num_classes = 0;
}

int32_t CMultilabelModel::get_dim() const
{
	int32_t num_classes = ((CMultilabelSOLabels *)m_labels)->get_num_classes();
	int32_t feats_dim = ((CDotFeatures *)m_features)->get_dim_feature_space();

	return feats_dim * num_classes;
}

void CMultilabelModel::set_misclass_cost(float64_t false_positive, float64_t false_negative)
{
	m_false_positive = false_positive;
	m_false_negative = false_negative;
}

SGVector<float64_t> CMultilabelModel::get_joint_feature_vector(int32_t feat_idx,
                CStructuredData * y)
{
	SGVector<float64_t> psi(get_dim());
	psi.zero();

	SGVector<float64_t> x = ((CDotFeatures *)m_features)->
	                        get_computed_dot_feature_vector(feat_idx);
	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(y);
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

float64_t CMultilabelModel::delta_loss(CStructuredData * y1, CStructuredData * y2)
{
	CSparseMultilabel * y1_slabel = CSparseMultilabel::obtain_from_generic(y1);
	CSparseMultilabel * y2_slabel = CSparseMultilabel::obtain_from_generic(y2);

	ASSERT(y1_slabel != NULL);
	ASSERT(y2_slabel != NULL);

	CMultilabelSOLabels * multi_labels = (CMultilabelSOLabels *)m_labels;
	return delta_loss(
	               CMultilabelSOLabels::to_dense(y1_slabel,
	                               multi_labels->get_num_classes(), 1, 0),
	               CMultilabelSOLabels::to_dense(y2_slabel,
	                               multi_labels->get_num_classes(), 1, 0));
}

float64_t CMultilabelModel::delta_loss(SGVector<float64_t> y1, SGVector<float64_t> y2)
{
	REQUIRE(y1.vlen == y2.vlen, "Size of both the vectors should be same\n");

	float64_t loss = 0;

	for (index_t i = 0; i < y1.vlen; i++)
	{
		loss += delta_loss(y1[i], y2[i]);
	}

	return loss;
}

float64_t CMultilabelModel::delta_loss(float64_t y1, float64_t y2)
{
	return y1 > y2 ? m_false_negative : y1 < y2 ? m_false_positive : 0;
}

SGVector<int32_t> CMultilabelModel::to_sparse(SGVector<float64_t> dense_vec,
                float64_t d_true, float64_t d_false)
{
	int32_t size = 0;

	for (index_t i = 0; i < dense_vec.vlen; i++)
	{
		REQUIRE(dense_vec[i] == d_true || dense_vec[i] == d_false,
		        "The values of dense vector should be either (%d) or (%d).\n",
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

CResultSet * CMultilabelModel::argmax(SGVector<float64_t> w, int32_t feat_idx,
                                      bool const training)
{
	CDotFeatures * dot_feats = (CDotFeatures *)m_features;
	int32_t feats_dim = dot_feats->get_dim_feature_space();

	CMultilabelSOLabels * multi_labs = (CMultilabelSOLabels *)m_labels;

	if (training)
	{
		m_num_classes = multi_labs->get_num_classes();
	}
	else
	{
		REQUIRE(m_num_classes > 0, "The model needs to be trained before using "
		        "it for prediction\n");
	}

	int32_t dim = get_dim();
	ASSERT(dim == w.vlen);

	float64_t score = 0, total_score = 0;
	SGVector<float64_t> y_pred_dense(m_num_classes);
	y_pred_dense.zero();

	for (int32_t c = 0; c < m_num_classes; c++)
	{
		score = dot_feats->dense_dot(feat_idx, w.vector + c * feats_dim, feats_dim);

		if (score > 0)
		{
			y_pred_dense[c] = 1;
			total_score += score;
		}

	}

	SGVector<int32_t> y_pred_sparse = to_sparse(y_pred_dense, 1, 0);

	CResultSet * ret = new CResultSet();
	SG_REF(ret);
	ret->psi_computed = true;

	CSparseMultilabel * y_pred = new CSparseMultilabel(y_pred_sparse);
	SG_REF(y_pred);

	ret->psi_pred = get_joint_feature_vector(feat_idx, y_pred);
	ret->score = total_score;
	ret->argmax = y_pred;

	if (training)
	{
		ret->delta = CStructuredModel::delta_loss(feat_idx, y_pred);
		ret->psi_truth = CStructuredModel::get_joint_feature_vector(
		                         feat_idx, feat_idx);
		ret->score += (ret->delta - CMath::dot(w.vector,
		                ret->psi_truth.vector, dim));
	}

	return ret;
}

void CMultilabelModel::init_primal_opt(
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

