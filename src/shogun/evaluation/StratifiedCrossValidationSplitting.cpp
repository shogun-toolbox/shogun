/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "evaluation/StratifiedCrossValidationSplitting.h"
#include "features/Labels.h"
#include "lib/Set.h"

using namespace shogun;

CStratifiedCrossValidationSplitting::CStratifiedCrossValidationSplitting() :
	CSplittingStrategy(0, 0)
{
}

CStratifiedCrossValidationSplitting::CStratifiedCrossValidationSplitting(
		CLabels* labels, index_t num_subsets) :
	CSplittingStrategy(labels, num_subsets)
{
	build_subsets();
}

void CStratifiedCrossValidationSplitting::build_subsets()
{
	/* extract all labels */
	CSet<float64_t> unique_labels;
	for (index_t i=0; i<m_labels->get_num_labels(); ++i)
		unique_labels.add(m_labels->get_label(i));

	/* for every label, build set for indices */
	CDynamicObjectArray<CDynamicArray<index_t> > label_indices;
	for (index_t i=0; i<unique_labels.get_num_elements(); ++i)
		label_indices.append_element(new CDynamicArray<index_t> ());

	/* fill set with indices, for each label type ... */
	for (index_t i=0; i<unique_labels.get_num_elements(); ++i)
	{
		/* ... iterate over all labels and add indices with same label to set */
		for (index_t j=0; j<m_labels->get_num_labels(); ++j)
		{
			if (m_labels->get_label(j)==unique_labels[i])
			{
				CDynamicArray<index_t>* current=label_indices.get_element(i);
				current->append_element(j);
				SG_UNREF(current);
			}
		}
	}

	/* shuffle created label sets */
	for (index_t i=0; i<label_indices.get_num_elements(); ++i)
	{
		CDynamicArray<index_t>* current=label_indices.get_element(i);
		current->shuffle();
		SG_UNREF(current);
	}

	/* distribute labels to subsets for all label types */
	index_t target_set=0;
	for (index_t i=0; i<unique_labels.get_num_elements(); ++i)
	{
		/* current index set for current label */
		CDynamicArray<index_t>* current=label_indices.get_element(i);

		for (index_t j=0; j<current->get_num_elements(); ++j)
		{
			CDynamicArray<index_t>* next=m_subset_indices.get_element(
					target_set++);
			next->append_element(current->get_element(j));
			target_set%=m_subset_indices.get_num_elements();
			SG_UNREF(next);
		}

		SG_UNREF(current);
	}

	/* finally shuffle to avoid that subsets with low indices have more
	 * elements, which happens if the number of class labels is not equal to
	 * the number of subsets */
	m_subset_indices.shuffle();
}
