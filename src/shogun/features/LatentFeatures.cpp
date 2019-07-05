/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Heiko Strathmann, Vladislav Horbatiuk,
 *          Soeren Sonnenburg
 */

#include <shogun/features/LatentFeatures.h>

using namespace shogun;

LatentFeatures::LatentFeatures():LatentFeatures(10)
{
}

LatentFeatures::LatentFeatures(int32_t num_samples)
{
	init();
	m_samples.reserve(num_samples);
}

LatentFeatures::~LatentFeatures()
{

}

std::shared_ptr<Features> LatentFeatures::duplicate() const
{
	return std::make_shared<LatentFeatures>(*this);
}

EFeatureType LatentFeatures::get_feature_type() const
{
	return F_ANY;
}

EFeatureClass LatentFeatures::get_feature_class() const
{
	return C_LATENT;
}


int32_t LatentFeatures::get_num_vectors() const
{
	return m_samples.size();
}

bool LatentFeatures::add_sample(std::shared_ptr<Data> example)
{
	m_samples.push_back(example);
	return true;
}

std::shared_ptr<Data> LatentFeatures::get_sample(index_t idx)
{
	if (idx < 0 || idx >= this->get_num_vectors())
		error("Out of index!");

	return m_samples[idx];
}

void LatentFeatures::init()
{
	SG_ADD(&m_samples, "samples", "Array of examples");
}

std::shared_ptr<LatentFeatures> LatentFeatures::obtain_from_generic(std::shared_ptr<Features> base_feats)
{
	ASSERT(base_feats != NULL)
	if (base_feats->get_feature_class() == C_LATENT)
		return std::static_pointer_cast<LatentFeatures>(base_feats);
	else
		error("base_labels must be of dynamic type CLatentLabels");

	return NULL;
}

