/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/features/SparseFeatures.h>
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

SGVector<float64_t> CHashedMultilabelModel::get_joint_feature_vector(
        int32_t feat_idx, CStructuredData * y)
{
	SG_ERROR("compute_joint_feature(int32_t, CStructuredData*) is not "
	         "implemented for %s!\n", get_name());

	return SGVector<float64_t>();
}

SGSparseVector<float64_t> CHashedMultilabelModel::get_sparse_joint_feature_vector(
        int32_t feat_idx, CStructuredData * y)
{
	SGSparseVector<float64_t> vec = ((CSparseFeatures<float64_t> *)m_features)->
	                                get_sparse_feature_vector(feat_idx);

	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(y);
	ASSERT(slabel != NULL);
	SGVector<int32_t> slabel_data = slabel->get_data();

	SGSparseVector<float64_t> psi(vec.num_feat_entries * slabel_data.vlen);
	index_t k = 0;

	for (int32_t i = 0; i < slabel_data.vlen; i++)
	{
		int32_t label = slabel_data[i];
		uint32_t seed = (uint32_t)m_seeds[label];

		for (int32_t j = 0; j < vec.num_feat_entries; j++)
		{
			uint32_t hash = CHash::MurmurHash3(
			                        (uint8_t *)&vec.features[j].feat_index,
			                        sizeof(index_t), seed);
			psi.features[k].feat_index = (hash >> 1) % m_dim;
			psi.features[k++].entry =
			        (hash % 2 == 1 ? -1.0 : 1.0) * vec.features[j].entry;
		}

	}

	psi.sort_features(true);
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

SGSparseVector<float64_t> CHashedMultilabelModel::get_hashed_feature_vector(
        int32_t feat_idx, uint32_t seed)
{
	SGSparseVector<float64_t> vec = ((CSparseFeatures<float64_t> *)m_features)->
	                                get_sparse_feature_vector(feat_idx);

	SGSparseVector<float64_t> h_vec(vec.num_feat_entries);

	for (int32_t j = 0; j < vec.num_feat_entries; j++)
	{
		uint32_t hash = CHash::MurmurHash3(
		                        (uint8_t *)&vec.features[j].feat_index,
		                        sizeof(index_t), seed);
		h_vec.features[j].feat_index = (hash >> 1) % m_dim;
		h_vec.features[j].entry =
		        (hash % 2 == 1 ? -1.0 : 1.0) * vec.features[j].entry;
	}

	h_vec.sort_features(true);

	return h_vec;
}

SGVector<int32_t> CHashedMultilabelModel::to_sparse(SGVector<float64_t> dense_vec,
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

	SGVector<float64_t> y_pred_dense(m_num_classes);
	y_pred_dense.zero();

	for (int32_t c = 0; c < m_num_classes; c++)
	{
		SGSparseVector<float64_t> phi = get_hashed_feature_vector(feat_idx,
		                                m_seeds[c]);
		score = phi.dense_dot(1.0, w.vector, w.vlen, 0);

		if (training)
		{
			score += delta_loss(y_truth[c], 1);
		}

		if (score > 0)
		{
			y_pred_dense[c] = 1;
			total_score += score;
		}

	}

	SGVector<int32_t> y_pred_sparse = to_sparse(y_pred_dense, 1, 0);

	CResultSet * ret = new CResultSet();
	SG_REF(ret);
	ret->psi_computed_sparse = true;
	ret->psi_computed = false;

	CSparseMultilabel * y_pred_label = new CSparseMultilabel(y_pred_sparse);
	SG_REF(y_pred_label);

	ret->psi_pred_sparse = get_sparse_joint_feature_vector(feat_idx, y_pred_label);
	ret->score = total_score;
	ret->argmax = y_pred_label;

	if (training)
	{
		ret->delta = CStructuredModel::delta_loss(feat_idx, y_pred_label);
		ret->psi_truth_sparse = CStructuredModel::get_sparse_joint_feature_vector(
		                                feat_idx, feat_idx);
		ret->score -= ret->psi_truth_sparse.dense_dot(1, w.vector, w.vlen, 0);
	}

	return ret;
}

