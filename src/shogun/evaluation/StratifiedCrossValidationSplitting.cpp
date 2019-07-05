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
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;

StratifiedCrossValidationSplitting::StratifiedCrossValidationSplitting() :
	RandomMixin<SplittingStrategy>()
{
}

StratifiedCrossValidationSplitting::StratifiedCrossValidationSplitting(
		std::shared_ptr<Labels> labels, index_t num_subsets) :
	RandomMixin<SplittingStrategy>(labels, num_subsets)
{
}

void StratifiedCrossValidationSplitting::check_labels() const
{
	auto dense_labels = std::static_pointer_cast<DenseLabels>(m_labels);
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
			io::warn(
			    "There are only {} labels of class {:.18g}, but {} "
			    "subsets. Labels of that class will not appear in every "
			    "subset!",
			    labels_per_class[i], classes[i], m_num_subsets);
		}
	}
}

void StratifiedCrossValidationSplitting::build_subsets()
{
	check_labels();

	/* ensure that subsets are empty and set flag to filled */
	reset_subsets();
	m_is_filled=true;

	auto dense_labels = m_labels->as<DenseLabels>();
	auto classes = dense_labels->get_labels().unique();

	/* for every label, build set for indices */
	std::vector<std::vector<index_t>> label_indices;
	label_indices.resize(classes.size());

	/* fill set with indices, for each label type ... */
	for (auto i : range(classes.size()))
	{
		/* ... iterate over all labels and add indices with same label to set */
		for (auto j : range(m_labels->get_num_labels()))
		{
			if (dense_labels->get_label(j) == classes[i])
			{
				label_indices[i].push_back(j);

			}
		}
	}

	/* shuffle created label sets */
	for (index_t i=0; i<label_indices.size(); ++i)
	{
		auto current=
				label_indices[i];

		// external random state important for threads
		random::shuffle(current, m_prng);


	}

	/* distribute labels to subsets for all label types */
	index_t target_set=0;
	for (auto i : range(classes.size()))
	{
		/* current index set for current label */
		auto current=
				label_indices[i];

		for (index_t j=0; j<current.size(); ++j)
		{
			auto next=
					m_subset_indices[target_set++];
			next.push_back(current[j]);
			target_set%=m_subset_indices.size();

		}


	}

	/* finally shuffle to avoid that subsets with low indices have more
	 * elements, which happens if the number of class labels is not equal to
	 * the number of subsets (external random state important for threads) */
	random::shuffle(m_subset_indices, m_prng);
}
