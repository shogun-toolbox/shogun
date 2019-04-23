/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Thoralf Klein
 * Written(W) 2014 Thoralf Klein
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/structure/MultilabelCLRModel.h>
#include <shogun/structure/MultilabelSOLabels.h>

using namespace shogun;

MultilabelCLRModel::MultilabelCLRModel()
	: StructuredModel()
{
	init();
}

MultilabelCLRModel::MultilabelCLRModel(std::shared_ptr<Features > features,
                std::shared_ptr<StructuredLabels > labels) : StructuredModel(features, labels)
{
	init();
}

std::shared_ptr<StructuredLabels > MultilabelCLRModel::structured_labels_factory(
        int32_t num_labels)
{
	return std::make_shared<MultilabelSOLabels>(num_labels, m_num_classes);
}

MultilabelCLRModel::~MultilabelCLRModel()
{
}

void MultilabelCLRModel::init()
{
	SG_ADD(&m_num_classes, "num_classes", "Number of (binary) class assignment per label");
	m_num_classes = 0;
}

int32_t MultilabelCLRModel::get_dim() const
{
	int32_t num_classes = m_labels->as<MultilabelSOLabels>()->get_num_classes();
	int32_t feats_dim = m_features->as<DotFeatures>()->get_dim_feature_space();

	return feats_dim * (num_classes + 1);
}

SGVector<float64_t> MultilabelCLRModel::get_joint_feature_vector(
        int32_t feat_idx, std::shared_ptr<StructuredData > y)
{
	SGVector<float64_t> psi(get_dim());
	psi.zero();

	int32_t num_classes = m_labels->as<MultilabelSOLabels>()->get_num_classes();
	int32_t num_pos_labels = (y->as<SparseMultilabel>())->
	                         get_data().vlen;
	int32_t num_neg_labels = num_classes - num_pos_labels;

	// the calibrated label is considered to be the last label
	SGVector<float64_t> label_coeffs(num_classes + 1);
	label_coeffs.zero();
	// the label coefficients would be positive for the relevant/positive
	// labels (P), negative for irrelevant/negative labels (N) and would be the
	// difference of number of irrelevant labels and number of revelant labels
	// for the calibrated/virtual label as
	// labels_coeff = \sum_{i \in P}{l(i) - l(v)} + \sum_{j \in N}{l(v) - l(j)}
	// where $v$ is the calibrated/virtual label
	label_coeffs += MultilabelSOLabels::to_dense(y, num_classes + 1, 1, -1);
	label_coeffs[num_classes] = num_neg_labels - num_pos_labels;

	auto dot_feats = m_features->as<DotFeatures>();
	SGVector<float64_t> x = dot_feats->get_computed_dot_feature_vector(feat_idx);
	int32_t feats_dim = dot_feats->get_dim_feature_space();

	for (index_t i = 0; i < num_classes + 1; i++)
	{
		float64_t coeff = label_coeffs[i];

		for (index_t j = 0; j < feats_dim; j++)
		{
			psi[i * feats_dim + j] = coeff * x[j];
		}
	}

	return psi;
}

float64_t MultilabelCLRModel::delta_loss(std::shared_ptr<StructuredData > y1, std::shared_ptr<StructuredData > y2)
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

float64_t MultilabelCLRModel::delta_loss(SGVector<float64_t> y1, SGVector<float64_t> y2)
{
	REQUIRE(y1.vlen == y2.vlen, "Size of both the vectors should be same\n");

	float64_t loss = 0;

	for (index_t i = 0; i < y1.vlen; i++)
	{
		loss += delta_loss(y1[i], y2[i]);
	}

	return loss;
}

float64_t MultilabelCLRModel::delta_loss(float64_t y1, float64_t y2)
{
	return y1 != y2 ? 1 : 0;
}

SGVector<int32_t> MultilabelCLRModel::to_sparse(SGVector<float64_t> dense_vec,
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

std::shared_ptr<ResultSet > MultilabelCLRModel::argmax(SGVector<float64_t> w, int32_t feat_idx,
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
		REQUIRE(m_num_classes > 0, "The model needs to be trained before using "
		        "it for prediction\n");
	}

	int32_t dim = get_dim();
	ASSERT(dim == w.vlen);

	SGVector<float64_t> plus_minus_one(m_num_classes);

	if (training)
	{
		plus_minus_one.set_const(-1);

		auto y_true = multi_labs->get_label(feat_idx)->as<SparseMultilabel>();
		SGVector<int32_t> y_true_data = y_true->get_data();

		for (index_t i = 0; i < y_true_data.vlen; i++)
		{
			plus_minus_one[y_true_data[i]] = 1;
		}


	}
	else
	{
		plus_minus_one.zero();
	}

	float64_t score = 0, calibrated_score = 0;

	// last label (m_num_class + 1)th label is the calibrated/virtual label
	calibrated_score = dot_feats->dense_dot(feat_idx, w.vector + m_num_classes * feats_dim,
	                                        feats_dim);

	SGVector<float64_t> class_product(m_num_classes);

	for (index_t i = 0; i < m_num_classes; i++)
	{
		score = dot_feats->dense_dot(feat_idx, w.vector + i * feats_dim,
		                             feats_dim);
		class_product[i] = score - calibrated_score;
	}

	int32_t count = 0;
	SGVector<float64_t> y_pred_dense(m_num_classes);
	y_pred_dense.zero();

	for (index_t i = 0; i < m_num_classes; i++)
	{
		score = class_product[i] - plus_minus_one[i];

		if (score >= 0)
		{
			y_pred_dense[i] = 1;
			count++;
		}
	}

	SGVector<int32_t> y_pred_sparse = to_sparse(y_pred_dense, 1, 0);
	ASSERT(count == y_pred_sparse.vlen);

	auto ret = std::make_shared<ResultSet>();

	ret->psi_computed = true;

	auto y_pred = std::make_shared<SparseMultilabel>(y_pred_sparse);


	ret->psi_pred = get_joint_feature_vector(feat_idx, y_pred);
	ret->score = linalg::dot(w, ret->psi_pred);
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

void MultilabelCLRModel::init_primal_opt(
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

