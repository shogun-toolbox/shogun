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
	m_samples = std::make_shared<DynamicObjectArray>(num_samples);

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
	if (m_samples == NULL)
		return 0;
	else
		return m_samples->get_array_size();
}

bool FactorGraphFeatures::add_sample(std::shared_ptr<FactorGraph> example)
{
	if (m_samples != NULL)
	{
		m_samples->push_back(example);
		return true;
	}
	else
		return false;
}

std::shared_ptr<FactorGraph> FactorGraphFeatures::get_sample(index_t idx)
{
	REQUIRE(m_samples != NULL, "%s::get_sample(): m_samples is NULL!\n", get_name());
	REQUIRE(idx >= 0 && idx < get_num_vectors(), "%s::get_sample(): out of index!\n", get_name());

	return m_samples->get_element<FactorGraph>(idx);
}

void FactorGraphFeatures::init()
{
	SG_ADD((std::shared_ptr<SGObject>*) &m_samples, "samples", "Array of examples");
}

std::shared_ptr<FactorGraphFeatures> FactorGraphFeatures::obtain_from_generic(std::shared_ptr<Features> base_feats)
{
	REQUIRE(base_feats != NULL, "FactorGraphFeatures::obtain_from_generic(): base_feats is NULL!\n");

	if (base_feats->get_feature_class() == C_FACTOR_GRAPH)
		return std::dynamic_pointer_cast<FactorGraphFeatures>(base_feats);
	else
		SG_SERROR("base_labels must be of dynamic type FactorGraph!\n")

	return NULL;
}

