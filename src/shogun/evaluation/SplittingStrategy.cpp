/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/features/Labels.h>

using namespace shogun;

CSplittingStrategy::CSplittingStrategy()
{
	init();
}

CSplittingStrategy::CSplittingStrategy(CLabels* labels, int32_t num_subsets)
{
	init();

	/* "assert" that num_subsets is smaller than num labels */
	if (labels->get_num_labels()<num_subsets)
	{
		SG_ERROR("Only %d labels for %d subsets!\n", labels->get_num_labels(),
				num_subsets);
	}

	m_labels=labels;
	SG_REF(m_labels);

	/* construct all arrays */
	for (index_t i=0; i<num_subsets; ++i)
		m_subset_indices->append_element(new CDynamicArray<index_t> ());
}

void CSplittingStrategy::init()
{
	m_labels=NULL;
	m_subset_indices=new CDynamicObjectArray<CDynamicArray<index_t> >();
	SG_REF(m_subset_indices);

	m_parameters->add((CSGObject**)m_labels, "labels", "Labels for subsets");
	m_parameters->add((CSGObject**)m_subset_indices, "subset_indices",
			"Set of sets of subset indices");
}

CSplittingStrategy::~CSplittingStrategy()
{
	SG_UNREF(m_labels);
	SG_UNREF(m_subset_indices);
}

SGVector<index_t> CSplittingStrategy::generate_subset_indices(index_t subset_idx)
{
	/* construct SGVector copy from index vector */
	CDynamicArray<index_t>* to_copy=m_subset_indices->get_element_safe(
			subset_idx);

	index_t num_elements=to_copy->get_num_elements();
	SGVector<index_t> result(num_elements, true);

	/* copy data */
	memcpy(result.vector, to_copy->get_array(), sizeof(index_t)*num_elements);

	SG_UNREF(to_copy);

	return result;
}

SGVector<index_t> CSplittingStrategy::generate_subset_inverse(index_t subset_idx)
{
	CDynamicArray<index_t>* to_invert=m_subset_indices->get_element_safe(
			subset_idx);

	SGVector<index_t> result(
			m_labels->get_num_labels()-to_invert->get_num_elements(), true);

	index_t index=0;
	for (index_t i=0; i<m_labels->get_num_labels(); ++i)
	{
		/* add i to inverse indices if it is not in the to be inverted set */
		if (to_invert->find_element(i)==-1)
			result.vector[index++]=i;
	}

	SG_UNREF(to_invert);

	return result;
}
