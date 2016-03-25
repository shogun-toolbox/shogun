/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/evaluation/MulticlassOVREvaluation.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/evaluation/PRCEvaluation.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/linalg/linalg.h>

using namespace shogun;

CMulticlassOVREvaluation::CMulticlassOVREvaluation() :
	CEvaluation(), m_binary_evaluation(NULL), m_graph_results(NULL), m_num_graph_results(0)
{
}

CMulticlassOVREvaluation::CMulticlassOVREvaluation(CBinaryClassEvaluation* binary_evaluation) :
	CEvaluation(), m_binary_evaluation(NULL), m_graph_results(NULL), m_num_graph_results(0)
{
	set_binary_evaluation(binary_evaluation);
}

CMulticlassOVREvaluation::~CMulticlassOVREvaluation()
{
	if (m_graph_results)
	{
		SG_FREE(m_graph_results);
	}

	if (m_binary_evaluation)
	{
		SG_UNREF(m_binary_evaluation);
	}
}

float64_t CMulticlassOVREvaluation::evaluate(CLabels* predicted, CLabels* ground_truth)
{
	ASSERT(m_binary_evaluation)
	ASSERT(predicted)
	ASSERT(ground_truth)
	int32_t n_labels = predicted->get_num_labels();
	ASSERT(n_labels)
	CMulticlassLabels* predicted_mc = (CMulticlassLabels*)predicted;
	CMulticlassLabels* ground_truth_mc = (CMulticlassLabels*)ground_truth;
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
	if (dynamic_cast<CROCEvaluation*>(m_binary_evaluation) || dynamic_cast<CPRCEvaluation*>(m_binary_evaluation))
	{
		for (int32_t i=0; i<m_num_graph_results; i++)
			m_graph_results[i].~SGMatrix<float64_t>();
		SG_FREE(m_graph_results);
		m_graph_results = SG_MALLOC(SGMatrix<float64_t>, n_classes);
		m_num_graph_results = n_classes;
	}
	for (int32_t c=0; c<n_classes; c++)
	{
		CLabels* pred = new CBinaryLabels(SGVector<float64_t>(all.get_column_vector(c),n_labels,false));
		SGVector<float64_t> gt_vec(n_labels);
		for (int32_t i=0; i<n_labels; i++)
		{
			if (ground_truth_mc->get_label(i)==c)
				gt_vec[i] = +1.0;
			else
				gt_vec[i] = -1.0;
		}
		CLabels* gt = new CBinaryLabels(gt_vec);
		m_last_results[c] = m_binary_evaluation->evaluate(pred, gt);

		if (dynamic_cast<CROCEvaluation*>(m_binary_evaluation))
		{
			new (&m_graph_results[c]) SGMatrix<float64_t>();
			m_graph_results[c] = ((CROCEvaluation*)m_binary_evaluation)->get_ROC();
		}
		if (dynamic_cast<CPRCEvaluation*>(m_binary_evaluation))
		{
			new (&m_graph_results[c]) SGMatrix<float64_t>();
			m_graph_results[c] = ((CPRCEvaluation*)m_binary_evaluation)->get_PRC();
		}
	}
	return linalg::mean(m_last_results);
}
