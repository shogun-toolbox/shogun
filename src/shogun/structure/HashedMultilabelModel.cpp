/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/lib/Hash.h>
#include <shogun/mathematics/Math.h>
#include <shogun/structure/HashedMultilabelModel.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <shogun/lib/DynamicArray.h>

using namespace shogun;

CHashedMultilabelModel::CHashedMultilabelModel()
	: CStructuredModel()
{
	init(0);
}

CHashedMultilabelModel::CHashedMultilabelModel(CFeatures * features,
                CStructuredLabels * labels, int32_t dim) : CStructuredModel(features, labels)
{
	init(dim);
}

CHashedMultilabelModel::~CHashedMultilabelModel()
{
}

CStructuredLabels * CHashedMultilabelModel::structured_labels_factory(
        int32_t num_examples)
{
	return new CMultilabelSOLabels(num_examples, m_num_classes);
}

void CHashedMultilabelModel::init(int32_t dim)
{
	SG_ADD(&m_false_positive, "false_positive", "Misclassification cost for false positive",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_false_negative, "false_negative", "Misclassification cost for false negative",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_num_classes, "num_classes", "Number of (binary) class assignment per label",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_dim, "dim", "New joint feature space dimension", MS_NOT_AVAILABLE);
	SG_ADD(&m_seeds, "seeds", "Vector of seeds used for hashing",
	       MS_NOT_AVAILABLE);

	m_false_positive = 1;
	m_false_negative = 1;
	m_num_classes = 0;
	m_dim = dim;

	if (m_labels != NULL)
	{
		m_seeds = SGVector<uint32_t>(
		                  ((CMultilabelSOLabels *)m_labels)->get_num_classes());
		SGVector<uint32_t>::range_fill_vector(m_seeds.vector, m_seeds.vlen);
	}
	else
	{
		m_seeds = SGVector<uint32_t>(0);
	}
}

int32_t CHashedMultilabelModel::get_dim() const
{
	return m_dim;
}

void CHashedMultilabelModel::set_misclass_cost(float64_t false_positive,
                float64_t false_negative)
{
	m_false_positive = false_positive;
	m_false_negative = false_negative;
}

void CHashedMultilabelModel::set_seeds(SGVector<uint32_t> seeds)
{
	REQUIRE(((CMultilabelSOLabels *)m_labels)->get_num_classes() == seeds.vlen,
	        "Seeds for all the classes not provided. \n");
	m_seeds = seeds;
}

SGVector<float64_t> CHashedMultilabelModel::get_hashed_feature_vector(
        int32_t feat_idx, uint32_t seed)
{
	SGVector<float64_t> x = ((CDotFeatures *)m_features)->
	                        get_computed_dot_feature_vector(feat_idx);
	SGVector<float64_t> h_vec(m_dim);
	h_vec.zero();

	for (index_t i = 0; i < x.vlen; i++)
	{
		uint32_t hash = CHash::MurmurHash3((uint8_t *)&i, sizeof(index_t), seed);
		h_vec[hash % m_dim] += x[i];
	}

	return h_vec;
}

SGVector<float64_t> CHashedMultilabelModel::get_joint_feature_vector(
        int32_t feat_idx, CStructuredData * y)
{
	SGVector<float64_t> psi(m_dim);
	psi.zero();

	SGVector<float64_t> x = ((CDotFeatures *)m_features)->
	                        get_computed_dot_feature_vector(feat_idx);
	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(y);
	ASSERT(slabel != NULL);
	SGVector<int32_t> slabel_data = slabel->get_data();

	for (index_t i = 0; i < slabel_data.vlen; i++)
	{
		SGVector<float64_t> h_vec = get_hashed_feature_vector(feat_idx,
		                            m_seeds[slabel_data[i]]);
		psi += h_vec;
	}

	return psi;
}

float64_t CHashedMultilabelModel::delta_loss(CStructuredData * y1,
                CStructuredData * y2)
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

float64_t CHashedMultilabelModel::delta_loss(SGVector<float64_t> y1,
                SGVector<float64_t> y2)
{
	REQUIRE(y1.vlen == y2.vlen, "Size of both the vectors should be same\n");

	float64_t loss = 0;

	for (index_t i = 0; i < y1.vlen; i++)
	{
		loss += delta_loss(y1[i], y2[i]);
	}

	return loss;
}

float64_t CHashedMultilabelModel::delta_loss(float64_t y1, float64_t y2)
{
	return y1 > y2 ? m_false_negative : y1 < y2 ? m_false_positive : 0;
}

void CHashedMultilabelModel::init_primal_opt(
        float64_t regularization,
        SGMatrix<float64_t> &A,
        SGVector<float64_t> a,
        SGMatrix<float64_t> B,
        SGVector<float64_t> &b,
        SGVector<float64_t> lb,
        SGVector<float64_t> ub,
        SGMatrix<float64_t> &C)
{
	C = SGMatrix<float64_t>::create_identity_matrix(get_dim(), regularization);
}

CResultSet * CHashedMultilabelModel::argmax(SGVector<float64_t> w,
                int32_t feat_idx, bool const training)
{
	CMultilabelSOLabels * multi_labs = (CMultilabelSOLabels *)m_labels;

	if (training)
	{
		m_num_classes = multi_labs->get_num_classes();
	}
	else
	{
		REQUIRE(m_num_classes > 0, "The model needs to be trained before using"
		        "it for prediction.\n");
	}

	int32_t dim = get_dim();
	ASSERT(dim == w.vlen);

	float64_t score = 0, total_score = 0;

	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(
	                                     multi_labs->get_label(feat_idx));
	SGVector<int32_t> slabel_data = slabel->get_data();
	SGVector<float64_t> y_truth = CMultilabelSOLabels::to_dense(
	                                      slabel, m_num_classes, 1, 0);
	SG_UNREF(slabel);

	CDynamicArray<int32_t> y_pred;

	for (int32_t c = 0; c < m_num_classes; c++)
	{
		SGVector<float64_t> phi = get_hashed_feature_vector(feat_idx,
		                          m_seeds[c]);
		score = SGVector<float64_t>::dot(w.vector, phi.vector, dim);

		if (training)
		{
			score += delta_loss(y_truth[c], 1);
		}

		if (score > 0)
		{
			y_pred.push_back(c);
		}

		total_score += score;
	}

	SGVector<int32_t> y_pred_sparse(y_pred.get_num_elements());
	memcpy(y_pred_sparse.vector, y_pred.get_array(),
	       sizeof(int32_t)*y_pred.get_num_elements());

	CResultSet * ret = new CResultSet();
	SG_REF(ret);
	CSparseMultilabel * y_pred_label = new CSparseMultilabel(y_pred_sparse);
	SG_REF(y_pred_label);

	ret->psi_pred = get_joint_feature_vector(feat_idx, y_pred_label);
	ret->score = total_score;
	ret->argmax = y_pred_label;

	if (training)
	{
		ret->delta = CStructuredModel::delta_loss(feat_idx, y_pred_label);
		ret->psi_truth = CStructuredModel::get_joint_feature_vector(
		                         feat_idx, feat_idx);
		ret->score -= SGVector<float64_t>::dot(w.vector, ret->psi_truth.vector,
		                                       dim);
	}

	return ret;
}

