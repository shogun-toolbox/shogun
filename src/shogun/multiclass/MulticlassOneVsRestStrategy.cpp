/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>

using namespace shogun;

CMulticlassOneVsRestStrategy::CMulticlassOneVsRestStrategy()
	:CMulticlassStrategy(), m_rejection_strategy(NULL)
{
}

CMulticlassOneVsRestStrategy::CMulticlassOneVsRestStrategy(CRejectionStrategy *rejection_strategy)
	:CMulticlassStrategy(), m_rejection_strategy(rejection_strategy)
{
	SG_REF(m_rejection_strategy);
}

SGVector<int32_t> CMulticlassOneVsRestStrategy::train_prepare_next()
{
	for (int32_t i=0; i < m_orig_labels->get_num_labels(); ++i)
	{
		if (m_orig_labels->get_int_label(i)==m_train_iter)
			m_train_labels->set_label(i, +1.0);
		else
			m_train_labels->set_label(i, -1.0);
	}

	// increase m_train_iter *after* setting labels
	CMulticlassStrategy::train_prepare_next();

	return SGVector<int32_t>();
}

int32_t CMulticlassOneVsRestStrategy::decide_label(SGVector<float64_t> outputs)
{
	if (m_rejection_strategy && m_rejection_strategy->reject(outputs))
		return CLabels::REJECTION_LABEL;

	return CMath::arg_max(outputs.vector, 1, outputs.vlen);
}
