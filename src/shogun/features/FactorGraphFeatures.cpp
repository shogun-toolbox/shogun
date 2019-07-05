/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shell Hu
 */

#include <shogun/features/FactorGraphFeatures.h>

using namespace shogun;

FactorGraphFeatures::FactorGraphFeatures(): FactorGraphFeatures(0)
{
}

FactorGraphFeatures::FactorGraphFeatures(int32_t num_samples)
{
	init();
	m_samples.reserve(num_samples);
}

FactorGraphFeatures::~FactorGraphFeatures()
{

}

std::shared_ptr<Features> FactorGraphFeatures::duplicate() const
{
	return std::make_shared<FactorGraphFeatures>(*this);
}

EFeatureType FactorGraphFeatures::get_feature_type() const
{
	return F_ANY;
}

EFeatureClass FactorGraphFeatures::get_feature_class() const
{
	return C_FACTOR_GRAPH;
}


int32_t FactorGraphFeatures::get_num_vectors() const
{
	return m_samples.size();
}

bool FactorGraphFeatures::add_sample(std::shared_ptr<FactorGraph> example)
{
	m_samples.push_back(example);
	return true;
}

std::shared_ptr<FactorGraph> FactorGraphFeatures::get_sample(index_t idx)
{
	require(m_samples != NULL, "{}::get_sample(): m_samples is NULL!", get_name());
	require(idx >= 0 && idx < get_num_vectors(), "{}::get_sample(): out of index!", get_name());
	return m_samples[idx];
}

void FactorGraphFeatures::init()
{
	SG_ADD(&m_samples, "samples", "Array of examples");
}

std::shared_ptr<FactorGraphFeatures> FactorGraphFeatures::obtain_from_generic(std::shared_ptr<Features> base_feats)
{
	require(base_feats != NULL, "FactorGraphFeatures::obtain_from_generic(): base_feats is NULL!");

	if (base_feats->get_feature_class() == C_FACTOR_GRAPH)
		return std::dynamic_pointer_cast<FactorGraphFeatures>(base_feats);
	else
		error("base_labels must be of dynamic type FactorGraph!");

	return NULL;
}

