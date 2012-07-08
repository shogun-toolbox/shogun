/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/labels/LatentLabels.h>

using namespace shogun;

CLatentData::CLatentData()
{

}

CLatentData::~CLatentData()
{

}

CLatentLabels::CLatentLabels()
	: CBinaryLabels()
{
	init();
}

CLatentLabels::CLatentLabels(int32_t num_labels)
	: CBinaryLabels(num_labels)
{
	init();
	m_latent_labels = new CDynamicObjectArray(num_labels);
	SG_REF(m_latent_labels);
}

CLatentLabels::~CLatentLabels()
{
	SG_UNREF(m_latent_labels);
}

void CLatentLabels::init()
{
	SG_ADD((CSGObject**) &m_latent_labels, "m_labels", "The labels", MS_NOT_AVAILABLE);
	m_latent_labels = NULL;
}

CDynamicObjectArray* CLatentLabels::get_labels() const
{
	SG_REF(m_latent_labels);
	return m_latent_labels;
}

CLatentData* CLatentLabels::get_latent_label(int32_t idx)
{
	ASSERT(m_latent_labels != NULL);
	if (idx < 0 || idx >= get_num_labels())
		SG_ERROR("Out of index!\n");

	return (CLatentData*) m_latent_labels->get_element(idx);
}

void CLatentLabels::add_latent_label(CLatentData* label)
{
	ASSERT(m_latent_labels != NULL);
	m_latent_labels->push_back(label);
}

bool CLatentLabels::set_latent_label(int32_t idx, CLatentData* label)
{
	if (idx < get_num_labels())
	{
		return m_latent_labels->set_element(label, idx);
	}
	else
	{
		return false;
	}
}

void CLatentLabels::ensure_valid(const char* context)
{
	if (m_latent_labels == NULL)
		SG_ERROR("Non-valid LatentLabels in %s", context);
}

CLatentLabels* CLatentLabels::obtain_from_generic(CLabels* base_labels)
{
	ASSERT(base_labels != NULL);
	if (base_labels->get_label_type() == LT_LATENT)
		return (CLatentLabels*) base_labels;
	else
		SG_SERROR("base_labels must be of dynamic type CLatentLabels\n");

	return NULL;
}

