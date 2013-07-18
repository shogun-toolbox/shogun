/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/features/LatentFeatures.h>

using namespace shogun;

CLatentFeatures::CLatentFeatures()
{
	init();
	m_samples = new CDynamicObjectArray(10);
	SG_REF(m_samples);
}

CLatentFeatures::CLatentFeatures(int32_t num_samples)
{
	init();
	m_samples = new CDynamicObjectArray(num_samples);
	SG_REF(m_samples);
}

CLatentFeatures::~CLatentFeatures()
{
	SG_UNREF(m_samples);
}

CFeatures* CLatentFeatures::duplicate() const
{
	return new CLatentFeatures(*this);
}

EFeatureType CLatentFeatures::get_feature_type() const
{
	return F_ANY;
}

EFeatureClass CLatentFeatures::get_feature_class() const
{
	return C_LATENT;
}


int32_t CLatentFeatures::get_num_vectors() const
{
	if (m_samples == NULL)
		return 0;
	else
		return m_samples->get_array_size();
}

bool CLatentFeatures::add_sample(CData* example)
{
	ASSERT(m_samples != NULL)
	if (m_samples != NULL)
	{
		m_samples->push_back(example);
		return true;
	}
	else
		return false;
}

CData* CLatentFeatures::get_sample(index_t idx)
{
	ASSERT(m_samples != NULL)
	if (idx < 0 || idx >= this->get_num_vectors())
		SG_ERROR("Out of index!\n")

	return (CData*) m_samples->get_element(idx);

}

void CLatentFeatures::init()
{
	SG_ADD((CSGObject**) &m_samples, "samples", "Array of examples",
			MS_NOT_AVAILABLE);
}

CLatentFeatures* CLatentFeatures::obtain_from_generic(CFeatures* base_feats)
{
	ASSERT(base_feats != NULL)
	if (base_feats->get_feature_class() == C_LATENT)
		return (CLatentFeatures*) base_feats;
	else
		SG_SERROR("base_labels must be of dynamic type CLatentLabels\n")

	return NULL;
}

