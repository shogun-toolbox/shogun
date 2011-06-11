/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "evaluation/SplittingStrategy.h"
#include "features/Labels.h"

using namespace shogun;

CSplittingStrategy::CSplittingStrategy() :
	m_labels(NULL)
{
}

CSplittingStrategy::CSplittingStrategy(CLabels* labels, int32_t num_subsets) :
	m_labels(labels)
{
	/* construct all arrays */
	for (index_t i=0; i<num_subsets; ++i)
		m_subset_indices.append_element(new DynArray<index_t> ());

	SG_REF(m_labels);
}

CSplittingStrategy::~CSplittingStrategy()
{
	/* delete all created arrays */
	for (index_t i=0; i<m_subset_indices.get_num_elements(); ++i)
		delete m_subset_indices[i];

	SG_UNREF(m_labels);
}

void CSplittingStrategy::generate_subset_indices(index_t subset_idx,
		SGVector<index_t>& result)
{
	/* construct SGVector copy from index vector */
	DynArray<index_t>* to_copy=m_subset_indices.get_element_safe(subset_idx);

	index_t num_elements=to_copy->get_num_elements();

	/* fill result vector */
	result.vector=new index_t[num_elements];
	result.length=num_elements;

	/* copy data */
	memcpy(result.vector, to_copy->get_array(), sizeof(index_t)*num_elements);
}

void CSplittingStrategy::generate_subset_inverse(index_t subset_idx,
		SGVector<index_t>& result)
{
	DynArray<index_t>* to_invert=m_subset_indices.get_element_safe(subset_idx);

	/* fill result vector */
	result.length=m_labels->get_num_labels()-to_invert->get_num_elements();
	result.vector=new index_t[result.length];

	index_t index=0;
	for (index_t i=0; i<m_labels->get_num_labels(); ++i)
	{
		/* add i to inverse indices if it is not in the to be inverted set */
		if (to_invert->find_element(i)==-1)
			result.vector[index++]=i;
	}
}
