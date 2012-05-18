/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/labels/Labels.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>

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
	SG_ADD((CSGObject**)&m_subset_stack, "subset_stack",
			"Current subset stack", MS_NOT_AVAILABLE);

	m_subset_stack=new CSubsetStack();
	SG_REF(m_subset_stack);
}

void CLabels::add_subset(SGVector<index_t> subset)
{
	m_subset_stack->add_subset(subset);
}

void CLabels::remove_subset()
{
	m_subset_stack->remove_subset();
}

void CLabels::remove_all_subsets()
{
	m_subset_stack->remove_all_subsets();
}

float64_t CLabels::get_confidence(int32_t idx)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(m_confidences.vector && idx<get_num_labels());
	return m_confidences.vector[real_num];
}

void CLabels::set_confidence(float64_t confidence, int32_t idx)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(m_confidences.vector && idx<get_num_labels());
	m_confidences.vector[real_num]=confidence;
}

void CLabels::set_confidences(SGVector<float64_t> confidences)
{
	if (confidences.vlen!=0 && confidences.vlen!=get_num_labels())
	{
		SG_ERROR("length of confidence values should match number of labels or"
				" have zero length (len(labels)=%d, len(confidences)=%d)\n",
				get_num_labels(), confidences.vlen);
	}

	m_confidences=confidences;
}

SGVector<float64_t> CLabels::get_confidences()
{
	return m_confidences;
}
