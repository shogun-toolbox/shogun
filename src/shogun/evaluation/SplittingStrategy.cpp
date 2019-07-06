/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Weijie Lin
 */

#include <shogun/evaluation/SplittingStrategy.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

SplittingStrategy::SplittingStrategy()
{
	init();
}

SplittingStrategy::SplittingStrategy(std::shared_ptr<Labels> labels, int32_t num_subsets)
{
	init();

	m_num_subsets=num_subsets;

	/* "assert" that num_subsets is smaller than num labels */
	if (labels->get_num_labels()<num_subsets)
	{
		error("Only {} labels for {} subsets!", labels->get_num_labels(),
				num_subsets);
	}

	m_labels=labels;


	reset_subsets();
}

void SplittingStrategy::reset_subsets()
{
	/* construct all arrays */
	m_subset_indices.clear();
	m_subset_indices.resize(m_num_subsets);

	m_is_filled=false;
}

void SplittingStrategy::init()
{
	m_labels=NULL;
	m_subset_indices.clear();

	m_is_filled=false;
	m_num_subsets=0;

	SG_ADD(&m_labels, "labels", "Labels for subsets");
	SG_ADD(
	    &m_subset_indices, "subset_indices", "Set of sets of subset indices");
	SG_ADD(
	    &m_is_filled, "is_filled", "Whether ther are index sets");
	SG_ADD(
	    &m_num_subsets, "num_subsets", "Number of index sets");
}

SplittingStrategy::~SplittingStrategy()
{


}

SGVector<index_t> SplittingStrategy::generate_subset_indices(index_t subset_idx) const
{
	if (!m_is_filled)
	{
		error("Call {}::build_subsets() before accessing them! If this error"
				" stays, its an implementation error of {}::build_subsets()",
				get_name(), get_name());
	}

	/* construct SGVector copy from index vector */
	auto& to_copy = m_subset_indices.at(subset_idx);

	index_t num_elements=to_copy.size();
	SGVector<index_t> result(num_elements, true);

	/* copy data */
	sg_memcpy(result.vector, to_copy.data(), sizeof(index_t)*num_elements);



	return result;
}

SGVector<index_t> SplittingStrategy::generate_subset_inverse(index_t subset_idx) const
{
	if (!m_is_filled)
	{
		error("Call {}::build_subsets() before accessing them! If this error"
				" stays, its an implementation error of {}::build_subsets()",
				get_name(), get_name());
	}

	auto& to_invert = m_subset_indices.at(subset_idx);

	SGVector<index_t> result(
			m_labels->get_num_labels()-to_invert.size(), true);

	index_t index=0;
	for (index_t i=0; i<m_labels->get_num_labels(); ++i)
	{
		/* add i to inverse indices if it is not in the to be inverted set */
		if (std::find(to_invert.begin(), to_invert.end(), i)==to_invert.end())
			result.vector[index++]=i;
	}



	return result;
}

index_t SplittingStrategy::get_num_subsets() const
{
	return m_num_subsets;
}
