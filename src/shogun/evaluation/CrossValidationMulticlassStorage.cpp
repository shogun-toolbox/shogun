/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Sergey Lisitsyn, Heiko Strathmann
 */

#include <shogun/evaluation/CrossValidationMulticlassStorage.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/evaluation/PRCEvaluation.h>

using namespace shogun;

void CCrossValidationMulticlassStorage::post_init()
{
	SG_DEBUG("Allocating %d ROC graphs\n", m_num_folds*m_num_runs*m_num_classes);
	m_fold_ROC_graphs = SG_MALLOC(SGMatrix<float64_t>, m_num_folds*m_num_runs*m_num_classes);
	for (int32_t i=0; i<m_num_folds*m_num_runs*m_num_classes; i++)
		new (&m_fold_ROC_graphs[i]) SGMatrix<float64_t>();
	
	SG_DEBUG("Allocating %d PRC graphs\n", m_num_folds*m_num_runs*m_num_classes);
	m_fold_PRC_graphs = SG_MALLOC(SGMatrix<float64_t>, m_num_folds*m_num_runs*m_num_classes);
	for (int32_t i=0; i<m_num_folds*m_num_runs*m_num_classes; i++)
		new (&m_fold_PRC_graphs[i]) SGMatrix<float64_t>();

	m_evaluations_results = SGVector<float64_t>(m_num_folds*m_num_runs*m_num_classes*m_binary_evaluations->get_num_elements());
}

void CCrossValidationMulticlassStorage::init_expose_labels(CLabels* labels)
{
	ASSERT((CMulticlassLabels*)labels);
	m_num_classes = ((CMulticlassLabels*)labels)->get_num_classes();
}

void CCrossValidationMulticlassStorage::post_update_results()
{
	CROCEvaluation eval_ROC;
	CPRCEvaluation eval_PRC;
	int32_t n_evals = m_binary_evaluations->get_num_elements();
	for (int32_t c=0; c<m_num_classes; c++)
	{
		SG_DEBUG("Computing ROC for run %d fold %d class %d", m_current_run_index, m_current_fold_index, c);
		CBinaryLabels* pred_labels_binary = m_pred_labels->get_binary_for_class(c);
		CBinaryLabels* true_labels_binary = m_true_labels->get_binary_for_class(c);
		eval_ROC.evaluate(pred_labels_binary, true_labels_binary);
		m_fold_ROC_graphs[m_current_run_index*m_num_folds*m_num_classes+m_current_fold_index*m_num_classes+c] = 
			eval_ROC.get_ROC();
		eval_PRC.evaluate(pred_labels_binary, true_labels_binary);
		m_fold_PRC_graphs[m_current_run_index*m_num_folds*m_num_classes+m_current_fold_index*m_num_classes+c] = 
			eval_PRC.get_PRC();
		
		for (int32_t i=0; i<n_evals; i++)
		{
			CBinaryClassEvaluation* evaluator = (CBinaryClassEvaluation*)m_binary_evaluations->get_element_safe(i);
			m_evaluations_results[m_current_run_index*m_num_folds*m_num_classes*n_evals+m_current_fold_index*m_num_classes*n_evals+c*n_evals+i] =
				evaluator->evaluate(pred_labels_binary, true_labels_binary);
			SG_UNREF(evaluator);
		}

		SG_UNREF(pred_labels_binary);
		SG_UNREF(true_labels_binary);
	}
}

void CCrossValidationMulticlassStorage::update_test_result(CLabels* results, const char* prefix)
{
	m_pred_labels = (CMulticlassLabels*)results;
}

void CCrossValidationMulticlassStorage::update_test_true_result(CLabels* results, const char* prefix)
{
	m_true_labels = (CMulticlassLabels*)results;
}

