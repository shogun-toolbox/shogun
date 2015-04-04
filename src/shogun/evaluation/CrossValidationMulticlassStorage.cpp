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
#include <shogun/evaluation/MulticlassAccuracy.h>

using namespace shogun;

CCrossValidationMulticlassStorage::CCrossValidationMulticlassStorage(bool compute_ROC, bool compute_PRC, bool compute_conf_matrices) :
	CCrossValidationOutput()
{
	m_initialized = false;
	m_compute_ROC = compute_ROC;
	m_compute_PRC = compute_PRC;
	m_compute_conf_matrices = compute_conf_matrices;
	m_pred_labels = NULL;
	m_true_labels = NULL;
	m_num_classes = 0;
	m_binary_evaluations = new CDynamicObjectArray();

	m_fold_ROC_graphs=NULL;
	m_conf_matrices=NULL;
}


CCrossValidationMulticlassStorage::~CCrossValidationMulticlassStorage()
{
	if (m_compute_ROC && m_fold_ROC_graphs)
	{
		SG_FREE(m_fold_ROC_graphs);
	}

	if (m_compute_PRC && m_fold_PRC_graphs)
	{
		SG_FREE(m_fold_PRC_graphs);
	}

	if (m_compute_conf_matrices && m_conf_matrices)
	{
		SG_FREE(m_conf_matrices);
	}

	if (m_binary_evaluations)
	{
		SG_UNREF(m_binary_evaluations);
	}
};


void CCrossValidationMulticlassStorage::post_init()
{
	if (m_initialized)
		SG_ERROR("CrossValidationMulticlassStorage was already initialized once\n")

	if (m_compute_ROC)
	{
		SG_DEBUG("Allocating %d ROC graphs\n", m_num_folds*m_num_runs*m_num_classes)
		m_fold_ROC_graphs = SG_MALLOC(SGMatrix<float64_t>, m_num_folds*m_num_runs*m_num_classes);
		for (int32_t i=0; i<m_num_folds*m_num_runs*m_num_classes; i++)
			new (&m_fold_ROC_graphs[i]) SGMatrix<float64_t>();
	}

	if (m_compute_PRC)
	{
		SG_DEBUG("Allocating %d PRC graphs\n", m_num_folds*m_num_runs*m_num_classes)
		m_fold_PRC_graphs = SG_MALLOC(SGMatrix<float64_t>, m_num_folds*m_num_runs*m_num_classes);
		for (int32_t i=0; i<m_num_folds*m_num_runs*m_num_classes; i++)
			new (&m_fold_PRC_graphs[i]) SGMatrix<float64_t>();
	}

	if (m_binary_evaluations->get_num_elements())
		m_evaluations_results = SGVector<float64_t>(m_num_folds*m_num_runs*m_num_classes*m_binary_evaluations->get_num_elements());

	m_accuracies = SGVector<float64_t>(m_num_folds*m_num_runs);

	if (m_compute_conf_matrices)
	{
		m_conf_matrices = SG_MALLOC(SGMatrix<int32_t>, m_num_folds*m_num_runs);
		for (int32_t i=0; i<m_num_folds*m_num_runs; i++)
			new (&m_conf_matrices[i]) SGMatrix<int32_t>();
	}

	m_initialized = true;
}

void CCrossValidationMulticlassStorage::init_expose_labels(CLabels* labels)
{
	ASSERT((CMulticlassLabels*)labels)
	m_num_classes = ((CMulticlassLabels*)labels)->get_num_classes();
}

void CCrossValidationMulticlassStorage::post_update_results()
{
	CROCEvaluation eval_ROC;
	CPRCEvaluation eval_PRC;
	int32_t n_evals = m_binary_evaluations->get_num_elements();
	for (int32_t c=0; c<m_num_classes; c++)
	{
		SG_DEBUG("Computing ROC for run %d fold %d class %d", m_current_run_index, m_current_fold_index, c)
		CBinaryLabels* pred_labels_binary = m_pred_labels->get_binary_for_class(c);
		CBinaryLabels* true_labels_binary = m_true_labels->get_binary_for_class(c);
		if (m_compute_ROC)
		{
			eval_ROC.evaluate(pred_labels_binary, true_labels_binary);
			m_fold_ROC_graphs[m_current_run_index*m_num_folds*m_num_classes+m_current_fold_index*m_num_classes+c] =
				eval_ROC.get_ROC();
		}
		if (m_compute_PRC)
		{
			eval_PRC.evaluate(pred_labels_binary, true_labels_binary);
			m_fold_PRC_graphs[m_current_run_index*m_num_folds*m_num_classes+m_current_fold_index*m_num_classes+c] =
				eval_PRC.get_PRC();
		}

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
	CMulticlassAccuracy accuracy;

	m_accuracies[m_current_run_index*m_num_folds+m_current_fold_index] = accuracy.evaluate(m_pred_labels, m_true_labels);

	if (m_compute_conf_matrices)
	{
		m_conf_matrices[m_current_run_index*m_num_folds+m_current_fold_index] = CMulticlassAccuracy::get_confusion_matrix(m_pred_labels, m_true_labels);
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

