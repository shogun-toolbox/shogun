/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/MulticlassOneVsOneStrategy.h>

using namespace shogun;

CMulticlassOneVsOneStrategy::CMulticlassOneVsOneStrategy()
	:CMulticlassStrategy(), m_num_machines(0), m_num_classes(0)
{
}

void CMulticlassOneVsOneStrategy::train_start(CLabels *orig_labels, CLabels *train_labels)
{
	CMulticlassStrategy::train_start(orig_labels, train_labels);
	m_num_classes = m_orig_labels->get_num_classes();
	m_num_machines=m_num_classes*(m_num_classes-1)/2;

	m_train_pair_idx_1 = 0;
	m_train_pair_idx_2 = 1;
}

bool CMulticlassOneVsOneStrategy::train_has_more()
{
	return m_train_iter < m_num_machines;
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
