/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Abinash Panda, Viktor Gal, Soeren Sonnenburg,
 *          Shell Hu, Thoralf Klein, Michal Uricar, Sanuj Sharma
 */

#include <shogun/features/DotFeatures.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/structure/MulticlassModel.h>
#include <shogun/structure/MulticlassSOLabels.h>

using namespace shogun;

MulticlassModel::MulticlassModel()
: StructuredModel()
{
	init();
}

	MulticlassModel::MulticlassModel(std::shared_ptr<Features> features, std::shared_ptr<StructuredLabels> labels)
: StructuredModel(features, labels)
{
	init();
}

MulticlassModel::~MulticlassModel()
{
}

std::shared_ptr<StructuredLabels> MulticlassModel::structured_labels_factory(int32_t num_labels)
{
	return std::make_shared<MulticlassSOLabels>(num_labels);
}

int32_t MulticlassModel::get_dim() const
{
	// TODO make the casts safe!
	int32_t num_classes = m_labels->as<MulticlassSOLabels>()->get_num_classes();
	int32_t feats_dim   = m_features->as<DotFeatures>()->get_dim_feature_space();

	return feats_dim*num_classes;
}

SGVector< float64_t > MulticlassModel::get_joint_feature_vector(int32_t feat_idx, std::shared_ptr<StructuredData> y)
{
	SGVector< float64_t > psi( get_dim() );
	psi.zero();

	SGVector< float64_t > x = m_features->as<DotFeatures>()->
		get_computed_dot_feature_vector(feat_idx);
	auto r = y->as<RealNumber>();
	ASSERT(r != NULL)
	float64_t label_value = r->value;

	for ( index_t i = 0, j = label_value*x.vlen ; i < x.vlen ; ++i, ++j )
		psi[j] = x[i];

	return psi;
}

std::shared_ptr<ResultSet> MulticlassModel::argmax(
		SGVector< float64_t > w,
		int32_t feat_idx,
		bool const training)
{
	auto df = m_features->as<DotFeatures>();
	int32_t feats_dim   = df->get_dim_feature_space();

	if ( training )
	{
		auto ml = m_labels->as<MulticlassSOLabels>();
		m_num_classes = ml->get_num_classes();
	}
	else
	{
		require(m_num_classes > 0, "The model needs to be trained before "
				"using it for prediction");
	}

	int32_t dim = get_dim();
	ASSERT(dim == w.vlen)

	// Find the class that gives the maximum score

	float64_t score = 0, ypred = 0;
	float64_t max_score = -Math::INFTY;

	for ( int32_t c = 0 ; c < m_num_classes ; ++c )
	{
		score = df->dot(feat_idx, w.slice(c*feats_dim, c*feats_dim + feats_dim));
		if ( training )
			score += delta_loss(feat_idx, c);

		if ( score > max_score )
		{
			max_score = score;
			ypred = c;
		}
	}

	// Build the ResultSet object to return
	auto ret = std::make_shared<ResultSet>();

	ret->psi_computed = true;
	auto y  = std::make_shared<RealNumber>(ypred);


	ret->psi_pred = get_joint_feature_vector(feat_idx, y);
	ret->score    = max_score;
	ret->argmax   = y;
	if ( training )
	{
		ret->delta     = StructuredModel::delta_loss(feat_idx, y);
		ret->psi_truth = StructuredModel::get_joint_feature_vector(
					feat_idx, feat_idx);
		ret->score    -= linalg::dot(w, ret->psi_truth);
	}

	return ret;
}

float64_t MulticlassModel::delta_loss(std::shared_ptr<StructuredData> y1, std::shared_ptr<StructuredData> y2)
{
	auto rn1 = y1->as<RealNumber>();
	auto rn2 = y2->as<RealNumber>();
	ASSERT(rn1 != NULL)
	ASSERT(rn2 != NULL)

	return delta_loss(rn1->value, rn2->value);
}

float64_t MulticlassModel::delta_loss(int32_t y1_idx, float64_t y2)
{
	require(y1_idx >= 0 || y1_idx < m_labels->get_num_labels(),
			"The label index must be inside [0, num_labels-1]");

	auto rn1 = m_labels->get_label(y1_idx)->as<RealNumber>();
	float64_t ret = delta_loss(rn1->value, y2);


	return ret;
}

float64_t MulticlassModel::delta_loss(float64_t y1, float64_t y2)
{
	return (y1 == y2) ? 0 : 1;
}

void MulticlassModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > & lb,
		SGVector< float64_t > & ub,
		SGMatrix< float64_t > & C)
{
	C = SGMatrix< float64_t >::create_identity_matrix(get_dim(), regularization);
}

void MulticlassModel::init()
{
	SG_ADD(&m_num_classes, "m_num_classes", "The number of classes");

	m_num_classes = 0;
}

