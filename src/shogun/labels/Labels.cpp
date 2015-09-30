/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/labels/Labels.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CLabels::CLabels()
	: CSGObject()
{
	init();
}

CLabels::~CLabels()
{
	SG_UNREF(m_subset_stack);
}

void CLabels::init()
{
	SG_ADD((CSGObject **)&m_subset_stack, "subset_stack",
	       "Current subset stack", MS_NOT_AVAILABLE);

	m_subset_stack = new CSubsetStack();
	SG_REF(m_subset_stack);
}

void CLabels::add_subset(SGVector<index_t> subset)
{
	m_subset_stack->add_subset(subset);
}

void CLabels::add_subset_in_place(SGVector<index_t> subset)
{
	m_subset_stack->add_subset_in_place(subset);
}

void CLabels::remove_subset()
{
	m_subset_stack->remove_subset();
}

void CLabels::remove_all_subsets()
{
	m_subset_stack->remove_all_subsets();
}

float64_t CLabels::get_value(int32_t idx)
{
	ASSERT(m_current_values.vector && idx < get_num_labels())
	int32_t real_num = m_subset_stack->subset_idx_conversion(idx);
	return m_current_values.vector[real_num];
}

void CLabels::set_value(float64_t value, int32_t idx)
{

	REQUIRE(m_current_values.vector, "%s::set_value(%f, %d): No values vector"
	        " set!\n", get_name(), value, idx);
	REQUIRE(get_num_labels(), "%s::set_value(%f, %d): Number of values is "
	        "zero!\n", get_name(), value, idx);

	int32_t real_num = m_subset_stack->subset_idx_conversion(idx);
	m_current_values.vector[real_num] = value;
}

void CLabels::set_values(SGVector<float64_t> values)
{
	if (m_current_values.vlen != 0 && m_current_values.vlen != get_num_labels())
	{
		SG_ERROR("length of value values should match number of labels or"
		         " have zero length (len(labels)=%d, len(values)=%d)\n",
		         get_num_labels(), values.vlen);
	}

	m_current_values = values;
}

SGVector<float64_t> CLabels::get_values()
{
	return m_current_values;
}
