/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <labels/StructuredLabels.h>

using namespace shogun;

CStructuredLabels::CStructuredLabels()
: CLabels()
{
	init();
}

CStructuredLabels::CStructuredLabels(int32_t num_labels)
: CLabels()
{
	init();
	m_labels = new CDynamicObjectArray(num_labels);
	SG_REF(m_labels);
}

CStructuredLabels::~CStructuredLabels()
{
	SG_UNREF(m_labels);
}

void CStructuredLabels::ensure_valid(const char* context)
{
	if ( m_labels == NULL )
		SG_ERROR("Non-valid StructuredLabels in %s", context)
}

CDynamicObjectArray* CStructuredLabels::get_labels() const
{
	SG_REF(m_labels);
	return m_labels;
}

CStructuredData* CStructuredLabels::get_label(int32_t idx)
{
	ensure_valid("CStructuredLabels::get_label(int32_t)");
	if ( idx < 0 || idx >= get_num_labels() )
		SG_ERROR("Index must be inside [0, num_labels-1]\n")

	return (CStructuredData*) m_labels->get_element(idx);
}

void CStructuredLabels::add_label(CStructuredData* label)
{
	ensure_valid_sdt(label);
	m_labels->push_back(label);
}

bool CStructuredLabels::set_label(int32_t idx, CStructuredData* label)
{
	ensure_valid_sdt(label);
	int32_t real_idx = m_subset_stack->subset_idx_conversion(idx);

	if ( real_idx < get_num_labels() )
	{
		return m_labels->set_element(label, real_idx);
	}
	else
	{
		return false;
	}
}

int32_t CStructuredLabels::get_num_labels() const
{
	if ( m_labels == NULL )
		return 0;
	else
		return m_labels->get_num_elements();
}

void CStructuredLabels::init()
{
	SG_ADD((CSGObject**) &m_labels, "m_labels", "The labels", MS_NOT_AVAILABLE);

	m_labels = NULL;
	m_sdt = SDT_UNKNOWN;
}

void CStructuredLabels::ensure_valid_sdt(CStructuredData* label)
{
	if ( m_sdt == SDT_UNKNOWN )
	{
		m_sdt = label->get_structured_data_type();
	}
	else
	{
		REQUIRE(label->get_structured_data_type() == m_sdt, "All the labels must "
				"belong to the same CStructuredData child class\n");
	}
}
