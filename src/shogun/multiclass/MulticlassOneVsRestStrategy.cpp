/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Shell Hu, Soeren Sonnenburg, Sergey Lisitsyn,
 *          Bjoern Esser, Sanuj Sharma
 */

#include <shogun/multiclass/MulticlassOneVsRestStrategy.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

MulticlassOneVsRestStrategy::MulticlassOneVsRestStrategy()
	: MulticlassStrategy()
{
}

MulticlassOneVsRestStrategy::MulticlassOneVsRestStrategy(EProbHeuristicType prob_heuris)
	: MulticlassStrategy(prob_heuris)
{
}

SGVector<int32_t> MulticlassOneVsRestStrategy::train_prepare_next()
{
	auto mc_orig = multiclass_labels(m_orig_labels);
	auto binary_train = binary_labels(m_train_labels);
	for (int32_t i=0; i < m_orig_labels->get_num_labels(); ++i)
	{
		if (mc_orig->get_int_label(i)==m_train_iter)
			binary_train->set_label(i, +1.0);
		else
			binary_train->set_label(i, -1.0);
	}

	// increase m_train_iter *after* setting labels
	MulticlassStrategy::train_prepare_next();

	return SGVector<int32_t>();
}

int32_t MulticlassOneVsRestStrategy::decide_label(SGVector<float64_t> outputs)
{
	if (m_rejection_strategy && m_rejection_strategy->reject(outputs))
		return DenseLabels::REJECTION_LABEL;

	return Math::arg_max(outputs.vector, 1, outputs.vlen);
}

SGVector<index_t> MulticlassOneVsRestStrategy::decide_label_multiple_output(SGVector<float64_t> outputs, int32_t n_outputs)
{
	float64_t* outputs_ = SG_MALLOC(float64_t, outputs.vlen);
	int32_t* indices_ = SG_MALLOC(int32_t, outputs.vlen);
	for (int32_t i=0; i<outputs.vlen; i++)
	{
		outputs_[i] = outputs[i];
		indices_[i] = i;
	}
	Math::qsort_backward_index(outputs_,indices_,outputs.vlen);
	SGVector<index_t> result(n_outputs);
	for (int32_t i=0; i<n_outputs; i++)
		result[i] = indices_[i];
	SG_FREE(outputs_);
	SG_FREE(indices_);
	return result;
}

void MulticlassOneVsRestStrategy::rescale_outputs(SGVector<float64_t> outputs)
{
	switch(get_prob_heuris_type())
	{
		case OVA_NORM:
			rescale_heuris_norm(outputs);
			break;
		case OVA_SOFTMAX:
			error("{}::rescale_outputs(): Need to specify sigmoid parameters!", get_name());
			break;
		case PROB_HEURIS_NONE:
			break;
		default:
			error("{}::rescale_outputs(): Unknown OVA probability heuristic type!", get_name());
			break;
	}
}

void MulticlassOneVsRestStrategy::rescale_outputs(SGVector<float64_t> outputs,
		const SGVector<float64_t> As, const SGVector<float64_t> Bs)
{
	if (get_prob_heuris_type()==OVA_SOFTMAX)
		rescale_heuris_softmax(outputs,As,Bs);
	else
		rescale_outputs(outputs);
}

void MulticlassOneVsRestStrategy::rescale_heuris_norm(SGVector<float64_t> outputs)
{
	if (m_num_classes != outputs.vlen)
	{
		error("{}::rescale_heuris_norm(): size(outputs) = {} != m_num_classes = {}",
				get_name(), outputs.vlen, m_num_classes);
	}

	float64_t norm = SGVector<float64_t>::sum(outputs);
	norm += 1E-10;
	for (int32_t i=0; i<outputs.vlen; i++)
		outputs[i] /= norm;
}

void MulticlassOneVsRestStrategy::rescale_heuris_softmax(SGVector<float64_t> outputs,
		const SGVector<float64_t>& As, const SGVector<float64_t>& Bs)
{
	if (m_num_classes != outputs.vlen)
	{
		error("{}::rescale_heuris_softmax(): size(outputs) = {} != m_num_classes = {}",
				get_name(), outputs.vlen, m_num_classes);
	}

	for (int32_t i=0; i<outputs.vlen; i++)
		outputs[i] = std::exp(-As[i] * outputs[i] - Bs[i]);

	float64_t norm = SGVector<float64_t>::sum(outputs);
	norm += 1E-10;
	for (int32_t i=0; i<outputs.vlen; i++)
		outputs[i] /= norm;
}
