#include <shogun/multiclass/MulticlassStrategy.h>

using namespace shogun;


CMulticlassStrategy::CMulticlassStrategy()
	:m_train_labels(NULL), m_orig_labels(NULL), m_train_iter(0)
{
}



CMulticlassOneVsRestStrategy::CMulticlassOneVsRestStrategy()
	:CMulticlassStrategy(), m_num_machines(0)
{
}

CSubset *CMulticlassOneVsRestStrategy::train_prepare_next()
{
	CMulticlassStrategy::train_prepare_next();

	for (int32_t i=0; i < m_orig_labels->get_num_labels(); ++i)
	{
		if (m_orig_labels->get_int_label(i)==m_train_iter)
			m_train_labels->set_label(i, +1.0);
		else
			m_train_labels->set_label(i, -1.0);
	}

	return NULL;
}



CMulticlassOneVsOneStrategy::CMulticlassOneVsOneStrategy()
	:CMulticlassStrategy(), m_num_machines(0), m_num_classes(0)
{
}

CSubset *CMulticlassOneVsOneStrategy::train_prepare_next()
{
	CMulticlassStrategy::train_prepare_next();

	SGVector<index_t> subset(m_orig_labels->get_num_labels());
	int32_t tot=0;
	for (int32_t k=0; k < m_orig_labels->get_num_labels(); ++k)
	{
		if (m_orig_labels->get_int_label(k)==m_train_pair_idx_1)
		{
			m_train_labels->set_label(k, +1.0);
			subset[tot]=k;
			tot++;
		}
		else if (m_orig_labels->get_int_label(k)==m_train_pair_idx_2)
		{
			m_train_labels->set_label(k, -1.0);
			subset[tot]=k;
			tot++;
		}
	}

	m_train_pair_idx_2++;
	if (m_train_pair_idx_2 >= m_num_classes)
	{
		m_train_pair_idx_1++;
		m_train_pair_idx_2=m_train_pair_idx_1+1;
	}

	return new CSubset(SGVector<index_t>(subset.vector, tot));
}

