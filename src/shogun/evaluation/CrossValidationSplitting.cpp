/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Thoralf Klein, Soeren Sonnenburg,
 *          Fernando Iglesias, Viktor Gal
 */

#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;

CrossValidationSplitting::CrossValidationSplitting() :
	RandomMixin<SplittingStrategy>()
{
}

CrossValidationSplitting::CrossValidationSplitting(
		std::shared_ptr<Labels> labels, index_t num_subsets) :
	RandomMixin<SplittingStrategy>(labels, num_subsets)
{
}

void CrossValidationSplitting::build_subsets()
{
	require(m_labels, "No labels provided.");
	/* ensure that subsets are empty and set flag to filled */
	reset_subsets();
	m_is_filled=true;

	/* permute indices */
	SGVector<index_t> indices(m_labels->get_num_labels());
	indices.range_fill();
	random::shuffle(indices, m_prng);

	index_t num_subsets=m_subset_indices.size();

	/* distribute indices to subsets */
	index_t current_subset=0;
	for (index_t i=0; i<indices.vlen; ++i)
	{
		/* fill current subset */
		auto& current = m_subset_indices[current_subset];

		/* add element of current index */
		current.push_back(indices.vector[i]);

		/* iterate over subsets */
		current_subset=(current_subset+1) % num_subsets;
	}

	/* finally shuffle to avoid that subsets with low indices have more
	 * elements, which happens if the number of class labels is not equal to
	 * the number of subsets (external random state important for threads) */
	random::shuffle(m_subset_indices, m_prng);
}
