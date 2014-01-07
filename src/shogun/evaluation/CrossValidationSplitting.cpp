/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2012 Berlin Institute of Technology and Max-Planck-Society
 */

#include <evaluation/CrossValidationSplitting.h>
#include <labels/Labels.h>

using namespace shogun;

CCrossValidationSplitting::CCrossValidationSplitting() :
	CSplittingStrategy()
{
	m_rng = sg_rand;
}

CCrossValidationSplitting::CCrossValidationSplitting(
		CLabels* labels, index_t num_subsets) :
	CSplittingStrategy(labels, num_subsets)
{
	m_rng = sg_rand;
}

void CCrossValidationSplitting::build_subsets()
{
	/* ensure that subsets are empty and set flag to filled */
	reset_subsets();
	m_is_filled=true;

	/* permute indices */
	SGVector<index_t> indices(m_labels->get_num_labels());
	indices.range_fill();
	indices.permute(m_rng);

	index_t num_subsets=m_subset_indices->get_num_elements();

	/* distribute indices to subsets */
	index_t current_subset=0;
	for (index_t i=0; i<indices.vlen; ++i)
	{
		/* fill current subset */
		CDynamicArray<index_t>* current=(CDynamicArray<index_t>*)
				m_subset_indices->get_element(current_subset);

		/* add element of current index */
		current->append_element(indices.vector[i]);

		/* unref */
		SG_UNREF(current);

		/* iterate over subsets */
		current_subset=(current_subset+1) % num_subsets;
	}

	/* finally shuffle to avoid that subsets with low indices have more
	 * elements, which happens if the number of class labels is not equal to
	 * the number of subsets (external random state important for threads) */
	m_subset_indices->shuffle(m_rng);
}
