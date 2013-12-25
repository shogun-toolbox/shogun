/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Saurabh Mahindre
 */

#include <shogun/evaluation/LOOCrossValidationSplitting.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

CLOOCrossValidationSplitting::CLOOCrossValidationSplitting() :
	CSplittingStrategy()
{
	m_rng = sg_rand;
}

CLOOCrossValidationSplitting::CLOOCrossValidationSplitting(
		CLabels* labels) :
	CSplittingStrategy(labels, labels->get_num_labels())
{
	m_rng = sg_rand;
}

void CLOOCrossValidationSplitting::build_subsets()
{
	/* ensure that subsets are empty and set flag to filled */
	reset_subsets();
	m_is_filled=true;

	/* permute indices */
	SGVector<index_t> indices(m_labels->get_num_labels());
	indices.range_fill();
	indices.permute(m_rng);

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
		current_subset=(current_subset+1);
	}
}
