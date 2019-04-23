/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Sergey Lisitsyn, Thoralf Klein, Shell Hu,
 *          Soeren Sonnenburg, Viktor Gal, Abinash Panda, Michal Uricar
 */

#include <shogun/structure/StructuredModel.h>

using namespace shogun;

ResultSet::ResultSet()
: SGObject(), argmax(NULL),
	psi_computed_sparse(false),
	psi_computed(false),
	psi_truth(SGVector<float64_t>(0)),
	psi_pred(SGVector<float64_t>(0)),
	psi_truth_sparse(SGSparseVector<float64_t>(0)),
	psi_pred_sparse(SGSparseVector<float64_t>(0)),
	score(0),
	delta(0)
{
}

ResultSet::~ResultSet()
{

}

std::shared_ptr<StructuredLabels> StructuredModel::structured_labels_factory(int32_t num_labels)
{
	return std::make_shared<StructuredLabels>(num_labels);
}

const char* ResultSet::get_name() const
{
	return "ResultSet";
}

StructuredModel::StructuredModel() : SGObject()
{
	init();
}

StructuredModel::StructuredModel(
		std::shared_ptr<Features>         features,
		std::shared_ptr<StructuredLabels> labels)
: SGObject()
{
	init();

	set_labels(labels);
	set_features(features);
}

StructuredModel::~StructuredModel()
{


}

void StructuredModel::init_primal_opt(
		float64_t regularization,
		SGMatrix< float64_t > & A,
		SGVector< float64_t > a,
		SGMatrix< float64_t > B,
		SGVector< float64_t > & b,
		SGVector< float64_t > & lb,
		SGVector< float64_t > & ub,
		SGMatrix< float64_t > & C)
{
	SG_ERROR("init_primal_opt is not implemented for %s!\n", get_name())
}

void StructuredModel::set_labels(std::shared_ptr<StructuredLabels> labels)
{


	m_labels = labels;
}

std::shared_ptr<StructuredLabels> StructuredModel::get_labels()
{

	return m_labels;
}

void StructuredModel::set_features(std::shared_ptr<Features> features)
{


	m_features = features;
}

std::shared_ptr<Features> StructuredModel::get_features()
{

	return m_features;
}

SGVector< float64_t > StructuredModel::get_joint_feature_vector(
		int32_t feat_idx,
		int32_t lab_idx)
{
	auto label = m_labels->get_label(lab_idx);
	SGVector< float64_t > ret = get_joint_feature_vector(feat_idx, label);

	return ret;
}

SGVector< float64_t > StructuredModel::get_joint_feature_vector(
		int32_t feat_idx,
		std::shared_ptr<StructuredData> y)
{
	SG_ERROR("compute_joint_feature(int32_t, StructuredData*) is not "
			"implemented for %s!\n", get_name());

	return SGVector< float64_t >();
}

SGSparseVector< float64_t > StructuredModel::get_sparse_joint_feature_vector(
		int32_t feat_idx,
		int32_t lab_idx)
{
	auto label = m_labels->get_label(lab_idx);
	SGSparseVector< float64_t > ret = get_sparse_joint_feature_vector(feat_idx, label);

	return ret;
}

SGSparseVector< float64_t > StructuredModel::get_sparse_joint_feature_vector(
		int32_t feat_idx,
		std::shared_ptr<StructuredData> y)
{
	SG_ERROR("compute_sparse_joint_feature(int32_t, StructuredData*) is not "
			"implemented for %s!\n", get_name());

	return SGSparseVector< float64_t >();
}

float64_t StructuredModel::delta_loss(int32_t ytrue_idx, std::shared_ptr<StructuredData> ypred)
{
	REQUIRE(ytrue_idx >= 0 || ytrue_idx < m_labels->get_num_labels(),
			"The label index must be inside [0, num_labels-1]\n");

	auto ytrue = m_labels->get_label(ytrue_idx);
	float64_t ret = delta_loss(ytrue, ypred);

	return ret;
}

float64_t StructuredModel::delta_loss(std::shared_ptr<StructuredData> y1, std::shared_ptr<StructuredData> y2)
{
	SG_ERROR("delta_loss(StructuredData*, StructuredData*) is not "
			"implemented for %s!\n", get_name());

	return 0.0;
}

void StructuredModel::init()
{
	SG_ADD((std::shared_ptr<Labels>*) &m_labels, "m_labels", "Structured labels");
	SG_ADD((std::shared_ptr<Features>*) &m_features, "m_features", "Feature vectors");

	m_features = NULL;
	m_labels   = NULL;
}

void StructuredModel::init_training()
{
	// Nothing to do here
}

bool StructuredModel::check_training_setup() const
{
	// Nothing to do here
	return true;
}

int32_t StructuredModel::get_num_aux() const
{
	return 0;
}

int32_t StructuredModel::get_num_aux_con() const
{
	return 0;
}
