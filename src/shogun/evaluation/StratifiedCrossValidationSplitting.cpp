/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Thoralf Klein, Viktor Gal
 */

#include <shogun/base/range.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>

using namespace shogun;

CStratifiedCrossValidationSplitting::CStratifiedCrossValidationSplitting() :
	CSplittingStrategy()
{
	m_rng = sg_rand;
}

CStratifiedCrossValidationSplitting::CStratifiedCrossValidationSplitting(
		CLabels* labels, index_t num_subsets) :
	CSplittingStrategy(labels, num_subsets)
{

	m_rng = sg_rand;
}

void CStratifiedCrossValidationSplitting::check_labels() const
{
	auto dense_labels = m_labels->as<CDenseLabels>();
	auto classes = dense_labels->get_labels().unique();

	SGVector<index_t> labels_per_class(classes.size());

	for (auto i : range(classes.size()))
	{
		labels_per_class[i] = 0;
		for (auto j : range(dense_labels->get_num_labels()))
		{
			if (classes[i] == dense_labels->get_label(j))
				labels_per_class[i]++;
		}
	}

	for (index_t i = 0; i < classes.size(); ++i)
	{
		if (labels_per_class[i] < m_num_subsets)
		{
			SG_WARNING(
			    "There are only %d labels of class %.18g, but %d "
			    "subsets. Labels of that class will not appear in every "
			    "subset!\n",
			    labels_per_class[i], classes[i], m_num_subsets);
		}
	}
}

void CStratifiedCrossValidationSplitting::build_subsets()
{
	check_labels();

	/* ensure that subsets are empty and set flag to filled */
	reset_subsets();
	m_is_filled=true;

	auto dense_labels = m_labels->as<CDenseLabels>();
	auto classes = dense_labels->get_labels().unique();

	/* for every label, build set for indices */
	CDynamicObjectArray label_indices;
	for (auto i : range(classes.size()))
		label_indices.append_element(new CDynamicArray<index_t> ());

	/* fill set with indices, for each label type ... */
	for (auto i : range(classes.size()))
	{
		/* ... iterate over all labels and add indices with same label to set */
		for (auto j : range(m_labels->get_num_labels()))
		{
			if (dense_labels->get_label(j) == classes[i])
			{
				CDynamicArray<index_t>* current=(CDynamicArray<index_t>*)
						label_indices.get_element(i);
				current->append_element(j);
				SG_UNREF(current);
			}
		}
	}

	/* shuffle created label sets */
	for (index_t i=0; i<label_indices.get_num_elements(); ++i)
	{
		CDynamicArray<index_t>* current=(CDynamicArray<index_t>*)
				label_indices.get_element(i);

		// external random state important for threads
		current->shuffle(m_rng);

		SG_UNREF(current);
	}

	/* distribute labels to subsets for all label types */
	index_t target_set=0;
	for (auto i : range(classes.size()))
	{
		/* current index set for current label */
		CDynamicArray<index_t>* current=(CDynamicArray<index_t>*)
				label_indices.get_element(i);

		for (index_t j=0; j<current->get_num_elements(); ++j)
		{
			CDynamicArray<index_t>* next=(CDynamicArray<index_t>*)
					m_subset_indices->get_element(target_set++);
			next->append_element(current->get_element(j));
			target_set%=m_subset_indices->get_num_elements();
			SG_UNREF(next);
		}

		SG_UNREF(current);
	}

	/* finally shuffle to avoid that subsets with low indices have more
	 * elements, which happens if the number of class labels is not equal to
	 * the number of subsets (external random state important for threads) */
	m_subset_indices->shuffle(m_rng);
}
