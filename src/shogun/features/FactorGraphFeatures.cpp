/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Shell Hu
 * Copyright (C) 2013 Shell Hu
 */

#include <shogun/features/FactorGraphFeatures.h>

using namespace shogun;

CFactorGraphFeatures::CFactorGraphFeatures()
{
	init();
	m_samples = new CDynamicObjectArray();
	SG_REF(m_samples);
}

CFactorGraphFeatures::CFactorGraphFeatures(int32_t num_samples)
{
	init();
	m_samples = new CDynamicObjectArray(num_samples);
	SG_REF(m_samples);
}

CFactorGraphFeatures::~CFactorGraphFeatures()
{
	SG_UNREF(m_samples);
}

CFeatures* CFactorGraphFeatures::duplicate() const
{
	return new CFactorGraphFeatures(*this);
}

EFeatureType CFactorGraphFeatures::get_feature_type() const
{
	return F_ANY;
}

EFeatureClass CFactorGraphFeatures::get_feature_class() const
{
	return C_FACTOR_GRAPH;
}


int32_t CFactorGraphFeatures::get_num_vectors() const
{
	if (m_samples == NULL)
		return 0;
	else
		return m_samples->get_array_size();
}

bool CFactorGraphFeatures::add_sample(CFactorGraph* example)
{
	if (m_samples != NULL)
	{
		m_samples->push_back(example);
		return true;
	}
	else
		return false;
}

CFactorGraph* CFactorGraphFeatures::get_sample(index_t idx)
{
	REQUIRE(m_samples != NULL, "%s::get_sample(): m_samples is NULL!\n", get_name());
	REQUIRE(idx >= 0 && idx < get_num_vectors(), "%s::get_sample(): out of index!\n", get_name());

	return dynamic_cast<CFactorGraph*>(m_samples->get_element(idx));
}

void CFactorGraphFeatures::init()
{
	SG_ADD((CSGObject**) &m_samples, "samples", "Array of examples",
			MS_NOT_AVAILABLE);
}

CFactorGraphFeatures* CFactorGraphFeatures::obtain_from_generic(CFeatures* base_feats)
{
	REQUIRE(base_feats != NULL, "CFactorGraphFeatures::obtain_from_generic(): base_feats is NULL!\n");

	if (base_feats->get_feature_class() == C_FACTOR_GRAPH)
		return dynamic_cast<CFactorGraphFeatures*>(base_feats);
	else
		SG_SERROR("base_labels must be of dynamic type CFactorGraph!\n")

	return NULL;
}

