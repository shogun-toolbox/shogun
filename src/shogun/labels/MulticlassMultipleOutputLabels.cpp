/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/labels/MulticlassMultipleOutputLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

CMulticlassMultipleOutputLabels::CMulticlassMultipleOutputLabels()
: CLabels()
{
	init();
}

CMulticlassMultipleOutputLabels::CMulticlassMultipleOutputLabels(int32_t num_labels)
: CLabels()
{
	init();
	m_labels = SG_MALLOC(SGVector<index_t>, num_labels);
	m_n_labels = num_labels;
}

CMulticlassMultipleOutputLabels::~CMulticlassMultipleOutputLabels()
{
	SG_FREE(m_labels);
}

void CMulticlassMultipleOutputLabels::ensure_valid(const char* context)
{
	if ( m_labels == NULL )
		SG_ERROR("Non-valid MulticlassMultipleOutputLabels in %s", context)
}

SGMatrix<index_t> CMulticlassMultipleOutputLabels::get_labels() const
{
	if (m_n_labels==0)
		return SGMatrix<index_t>();
	int n_outputs = m_labels[0].vlen;
	SGMatrix<index_t> labels(m_n_labels,n_outputs);
	for (int32_t i=0; i<m_n_labels; i++)
	{
		for (int32_t j=0; j<n_outputs; j++)
			labels(i,j) = m_labels[i][j];
	}
	return labels;
}

SGVector<index_t> CMulticlassMultipleOutputLabels::get_label(int32_t idx)
{
	ensure_valid("CMulticlassMultipleOutputLabels::get_label(int32_t)");
	if ( idx < 0 || idx >= get_num_labels() )
		SG_ERROR("Index must be inside [0, num_labels-1]\n")

	return m_labels[m_subset_stack->subset_idx_conversion(idx)];
}

bool CMulticlassMultipleOutputLabels::set_label(int32_t idx, SGVector<index_t> label)
{
	int32_t real_idx = m_subset_stack->subset_idx_conversion(idx);

	if (real_idx < get_num_labels())
	{
		m_labels[real_idx] = label;
		return true;
	}
	else
		return false;
}

int32_t CMulticlassMultipleOutputLabels::get_num_labels() const
{
	return m_n_labels;
}

void CMulticlassMultipleOutputLabels::init()
{
	m_labels = NULL;
	m_n_labels = 0;
}
