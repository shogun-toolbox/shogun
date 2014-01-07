/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <evaluation/SplittingStrategy.h>
#include <labels/Labels.h>

using namespace shogun;

CSplittingStrategy::CSplittingStrategy()
{
	init();
}

CSplittingStrategy::CSplittingStrategy(CLabels* labels, int32_t num_subsets)
{
	init();

	m_num_subsets=num_subsets;

	/* "assert" that num_subsets is smaller than num labels */
	if (labels->get_num_labels()<num_subsets)
	{
		SG_ERROR("Only %d labels for %d subsets!\n", labels->get_num_labels(),
				num_subsets);
	}

	m_labels=labels;
	SG_REF(m_labels);

	reset_subsets();
}

void CSplittingStrategy::reset_subsets()
{
	if (m_subset_indices)
		SG_UNREF(m_subset_indices);

	m_subset_indices=new CDynamicObjectArray();
	SG_REF(m_subset_indices);

	/* construct all arrays */
	for (index_t i=0; i<m_num_subsets; ++i)
		m_subset_indices->append_element(new CDynamicArray<index_t> ());

	m_is_filled=false;
}

void CSplittingStrategy::init()
{
	m_labels=NULL;
	m_subset_indices=NULL;
	SG_REF(m_subset_indices);
	m_is_filled=false;
	m_num_subsets=0;

	m_parameters->add((CSGObject**)&m_labels, "labels", "Labels for subsets");
	m_parameters->add((CSGObject**)&m_subset_indices, "subset_indices",
			"Set of sets of subset indices");
	m_parameters->add(&m_is_filled, "is_filled", "Whether ther are index sets");
	m_parameters->add(&m_num_subsets, "num_subsets", "Number of index sets");
}

CSplittingStrategy::~CSplittingStrategy()
{
	SG_UNREF(m_labels);
	SG_UNREF(m_subset_indices);
}

SGVector<index_t> CSplittingStrategy::generate_subset_indices(index_t subset_idx)
{
	if (!m_is_filled)
	{
		SG_ERROR("Call %s::build_subsets() before accessing them! If this error"
				" stays, its an implementation error of %s::build_subsets()\n",
				get_name(), get_name());
	}

	/* construct SGVector copy from index vector */
	CDynamicArray<index_t>* to_copy=(CDynamicArray<index_t>*)
			m_subset_indices->get_element_safe(subset_idx);

	index_t num_elements=to_copy->get_num_elements();
	SGVector<index_t> result(num_elements, true);

	/* copy data */
	memcpy(result.vector, to_copy->get_array(), sizeof(index_t)*num_elements);

	SG_UNREF(to_copy);

	return result;
}

SGVector<index_t> CSplittingStrategy::generate_subset_inverse(index_t subset_idx)
{
	if (!m_is_filled)
	{
		SG_ERROR("Call %s::build_subsets() before accessing them! If this error"
				" stays, its an implementation error of %s::build_subsets()\n",
				get_name(), get_name());
	}

	CDynamicArray<index_t>* to_invert=(CDynamicArray<index_t>*)
			m_subset_indices->get_element_safe(subset_idx);

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

index_t CSplittingStrategy::get_num_subsets() const
{
	return m_subset_indices->get_num_elements();
}
