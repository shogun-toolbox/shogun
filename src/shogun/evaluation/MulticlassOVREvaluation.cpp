/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Bjoern Esser, Sergey Lisitsyn, Soeren Sonnenburg
 */

#include <shogun/evaluation/MulticlassOVREvaluation.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/evaluation/PRCEvaluation.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

MulticlassOVREvaluation::MulticlassOVREvaluation() :
	MulticlassOVREvaluation(nullptr)
{
}

MulticlassOVREvaluation::MulticlassOVREvaluation(std::shared_ptr<BinaryClassEvaluation> binary_evaluation) :
	Evaluation(), m_binary_evaluation(nullptr), m_graph_results(nullptr), m_num_graph_results(0)
{
    SG_ADD((std::shared_ptr<Evaluation>*)&m_binary_evaluation, "binary_evaluation", "binary evaluator")
}

MulticlassOVREvaluation::~MulticlassOVREvaluation()
{
	if (m_graph_results)
	{
		SG_FREE(m_graph_results);
	}
}

float64_t MulticlassOVREvaluation::evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	ASSERT(m_binary_evaluation)
	ASSERT(predicted)
	ASSERT(ground_truth)
	int32_t n_labels = predicted->get_num_labels();
	ASSERT(n_labels)
	auto predicted_mc = multiclass_labels(predicted);
	auto ground_truth_mc = multiclass_labels(ground_truth);
	int32_t n_classes = predicted_mc->get_multiclass_confidences(0).size();
	ASSERT(n_classes>0)
	m_last_results = SGVector<float64_t>(n_classes);

	SGMatrix<float64_t> all(n_labels,n_classes);
	for (int32_t i=0; i<n_labels; i++)
	{
		SGVector<float64_t> confs = predicted_mc->get_multiclass_confidences(i);
		for (int32_t j=0; j<n_classes; j++)
		{
			all(i,j) = confs[j];
		}
	}
	if (std::dynamic_pointer_cast<ROCEvaluation>(m_binary_evaluation) || std::dynamic_pointer_cast<PRCEvaluation>(m_binary_evaluation))
	{
		for (int32_t i=0; i<m_num_graph_results; i++)
			m_graph_results[i].~SGMatrix<float64_t>();
		SG_FREE(m_graph_results);
		m_graph_results = SG_MALLOC(SGMatrix<float64_t>, n_classes);
		m_num_graph_results = n_classes;
	}
	for (int32_t c=0; c<n_classes; c++)
	{
		auto pred = std::make_shared<BinaryLabels>(SGVector<float64_t>(all.get_column_vector(c),n_labels,false));
		SGVector<float64_t> gt_vec(n_labels);
		for (int32_t i=0; i<n_labels; i++)
		{
			if (ground_truth_mc->get_label(i)==c)
				gt_vec[i] = +1.0;
			else
				gt_vec[i] = -1.0;
		}
		auto gt = std::make_shared<BinaryLabels>(gt_vec);
		m_last_results[c] = m_binary_evaluation->evaluate(pred, gt);

		if (std::dynamic_pointer_cast<ROCEvaluation>(m_binary_evaluation))
		{
			new (&m_graph_results[c]) SGMatrix<float64_t>();
			m_graph_results[c] = (std::static_pointer_cast<ROCEvaluation>(m_binary_evaluation))->get_ROC();
		}
		if (std::dynamic_pointer_cast<PRCEvaluation>(m_binary_evaluation))
		{
			new (&m_graph_results[c]) SGMatrix<float64_t>();
			m_graph_results[c] = (std::static_pointer_cast<PRCEvaluation>(m_binary_evaluation))->get_PRC();
		}
	}
	return Statistics::mean(m_last_results);
}
