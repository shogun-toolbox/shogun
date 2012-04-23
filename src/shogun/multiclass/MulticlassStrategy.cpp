#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;


CMulticlassStrategy::CMulticlassStrategy()
	:m_train_labels(NULL), m_orig_labels(NULL), m_train_iter(0)
{
}



CMulticlassOneVsRestStrategy::CMulticlassOneVsRestStrategy()
	:CMulticlassStrategy(), m_num_machines(0), m_rejection_strategy(NULL)
{
}

CMulticlassOneVsRestStrategy::CMulticlassOneVsRestStrategy(CRejectionStrategy *rejection_strategy)
	:CMulticlassStrategy(), m_num_machines(0), m_rejection_strategy(rejection_strategy)
{
}

SGVector<int32_t> CMulticlassOneVsRestStrategy::train_prepare_next()
{
	CMulticlassStrategy::train_prepare_next();

	for (int32_t i=0; i < m_orig_labels->get_num_labels(); ++i)
	{
		if (m_orig_labels->get_int_label(i)==m_train_iter)
			m_train_labels->set_label(i, +1.0);
		else
			m_train_labels->set_label(i, -1.0);
	}

	return SGVector<int32_t>(0);
}

int32_t CMulticlassOneVsRestStrategy::decide_label(const SGVector<float64_t> &outputs, int32_t num_classes)
{
	if (m_rejection_strategy && m_rejection_strategy->reject(outputs))
		return CLabels::REJECTION_LABEL;

	return CMath::arg_max(outputs.vector, 1, outputs.vlen);
}


CMulticlassOneVsOneStrategy::CMulticlassOneVsOneStrategy()
	:CMulticlassStrategy(), m_num_machines(0), m_num_classes(0)
{
}

SGVector<int32_t> CMulticlassOneVsOneStrategy::train_prepare_next()
{
	CMulticlassStrategy::train_prepare_next();

	SGVector<int32_t> subset(m_orig_labels->get_num_labels());
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

	return SGVector<int32_t>(subset.vector, tot);
}

int32_t CMulticlassOneVsOneStrategy::decide_label(const SGVector<float64_t> &outputs, int32_t num_classes)
{
	int32_t s=0;
	SGVector<int32_t> votes(num_classes);
	votes.zero();

	for (int32_t i=0; i<num_classes; i++)
	{
		for (int32_t j=i+1; j<num_classes; j++)
		{
			if (outputs[s++]>0)
				votes[i]++;
			else
				votes[j]++;
		}
	}

	int32_t result=CMath::arg_max(votes.vector, 1, votes.vlen);
	votes.destroy_vector();

	return result;
}
