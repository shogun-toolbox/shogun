#include <shogun/evaluation/TimeSeriesSplitting.h>
#include <shogun/labels/Labels.h>

using namespace shogun;

CTimeSeriesSplitting::CTimeSeriesSplitting() : CSplittingStrategy()
{
	m_rng = sg_rand;
}

CTimeSeriesSplitting::CTimeSeriesSplitting(CLabels* labels, index_t num_subsets)
    : CSplittingStrategy(labels, num_subsets)
{
	m_rng = sg_rand;
}

void CTimeSeriesSplitting::build_subsets()
{
	reset_subsets();
	m_is_filled = true;

	SGVector<index_t> indices(m_labels->get_num_labels());
	indices.range_fill();
	index_t num_subsets = m_subset_indices->get_num_elements();

	for (index_t i = 0; i < num_subsets; i++)
	{
		CDynamicArray<index_t>* current =
		    (CDynamicArray<index_t>*)m_subset_indices->get_element(i);

		/* filling current with indices on right end  */
		for (index_t k = i == num_subsets - 1
		                     ? indices.vlen - m_h
		                     : (i + 1) * (indices.vlen / num_subsets);
		     k < indices.vlen; k++)
		{
			current->append_element(indices.vector[k]);
		}

		/* unref */
		SG_UNREF(current);
	}

	m_subset_indices->shuffle(m_rng);
}

/* To ensure to get h value in future in test set*/
void CTimeSeriesSplitting::set_h(index_t h)
{
	index_t num_subsets = m_subset_indices->get_num_elements();
	index_t num_labels = m_labels->get_num_labels();

	/* h value should not be greater than difference between number of labels
	 * and start index of second last split point */
	if (h >= num_labels - (num_subsets - 1) * (num_labels / num_subsets))
	{
		SG_WARNING("h can not be %d. Setting h value to default 1.\n", h);
		m_h = 1;
		return;
	}
	m_h = h;
}

index_t CTimeSeriesSplitting::get_h()
{
	return m_h;
}