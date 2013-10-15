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
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CMulticlassOneVsRestStrategy::CMulticlassOneVsRestStrategy()
	: CMulticlassStrategy()
{
}

CMulticlassOneVsRestStrategy::CMulticlassOneVsRestStrategy(EProbHeuristicType prob_heuris)
	: CMulticlassStrategy(prob_heuris)
{
}

SGVector<int32_t> CMulticlassOneVsRestStrategy::train_prepare_next()
{
	for (int32_t i=0; i < m_orig_labels->get_num_labels(); ++i)
	{
		if (((CMulticlassLabels*) m_orig_labels)->get_int_label(i)==m_train_iter)
			((CBinaryLabels*) m_train_labels)->set_label(i, +1.0);
		else
			((CBinaryLabels*) m_train_labels)->set_label(i, -1.0);
	}

	// increase m_train_iter *after* setting labels
	CMulticlassStrategy::train_prepare_next();

	return SGVector<int32_t>();
}

int32_t CMulticlassOneVsRestStrategy::decide_label(SGVector<float64_t> outputs)
{
	if (m_rejection_strategy && m_rejection_strategy->reject(outputs))
		return CDenseLabels::REJECTION_LABEL;

	return SGVector<float64_t>::arg_max(outputs.vector, 1, outputs.vlen);
}

SGVector<index_t> CMulticlassOneVsRestStrategy::decide_label_multiple_output(SGVector<float64_t> outputs, int32_t n_outputs)
{
	float64_t* outputs_ = SG_MALLOC(float64_t, outputs.vlen);
	int32_t* indices_ = SG_MALLOC(int32_t, outputs.vlen);
	for (int32_t i=0; i<outputs.vlen; i++)
	{
		outputs_[i] = outputs[i];
		indices_[i] = i;
	}
	CMath::qsort_backward_index(outputs_,indices_,outputs.vlen);
	SGVector<index_t> result(n_outputs);
	for (int32_t i=0; i<n_outputs; i++)
		result[i] = indices_[i];
	SG_FREE(outputs_);
	SG_FREE(indices_);
	return result;
}

void CMulticlassOneVsRestStrategy::rescale_outputs(SGVector<float64_t> outputs)
{
	switch(get_prob_heuris_type())
	{
		case OVA_NORM:
			rescale_heuris_norm(outputs);
			break;
		case OVA_SOFTMAX:
			SG_ERROR("%s::rescale_outputs(): Need to specify sigmoid parameters!\n", get_name());
			break;
		case PROB_HEURIS_NONE:
			break;
		default:
			SG_ERROR("%s::rescale_outputs(): Unknown OVA probability heuristic type!\n", get_name());
			break;
	}
}

void CMulticlassOneVsRestStrategy::rescale_outputs(SGVector<float64_t> outputs,
		const SGVector<float64_t> As, const SGVector<float64_t> Bs)
{
	if (get_prob_heuris_type()==OVA_SOFTMAX)
		rescale_heuris_softmax(outputs,As,Bs);
	else
		rescale_outputs(outputs);
}

void CMulticlassOneVsRestStrategy::rescale_heuris_norm(SGVector<float64_t> outputs)
{
	if (m_num_classes != outputs.vlen)
	{
		SG_ERROR("%s::rescale_heuris_norm(): size(outputs) = %d != m_num_classes = %d\n",
				get_name(), outputs.vlen, m_num_classes);
	}

	float64_t norm = SGVector<float64_t>::sum(outputs);
	norm += 1E-10;
	for (int32_t i=0; i<outputs.vlen; i++)
		outputs[i] /= norm;
}

void CMulticlassOneVsRestStrategy::rescale_heuris_softmax(SGVector<float64_t> outputs,
		const SGVector<float64_t> As, const SGVector<float64_t> Bs)
{
	if (m_num_classes != outputs.vlen)
	{
		SG_ERROR("%s::rescale_heuris_softmax(): size(outputs) = %d != m_num_classes = %d\n",
				get_name(), outputs.vlen, m_num_classes);
	}

	for (int32_t i=0; i<outputs.vlen; i++)
		outputs[i] = CMath::exp(-As[i]*outputs[i]-Bs[i]);

	float64_t norm = SGVector<float64_t>::sum(outputs);
	norm += 1E-10;
	for (int32_t i=0; i<outputs.vlen; i++)
		outputs[i] /= norm;
}
