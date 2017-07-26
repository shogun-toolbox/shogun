/*
* BSD 3-Clause License
*
* Copyright (c) 2017, Shogun-Toolbox e.V. <shogun-team@shogun-toolbox.org>
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* * Redistributions of source code must retain the above copyright notice, this
*   list of conditions and the following disclaimer.
*
* * Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
* * Neither the name of the copyright holder nor the names of its
*   contributors may be used to endorse or promote products derived from
*   this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* Written (W) 2017 Giovanni De Toni
*
* This code was adapted from the previous CrossValidationMulticlassStorage
* class, which was written by Heiko Strathmann and Sergey Lisitsyn.
*
*/

#include <shogun/evaluation/BinaryClassEvaluation.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/evaluation/PRCEvaluation.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/parameter_observers/ParameterObserverCVMulticlass.h>

using namespace shogun;

ParameterObserverCVMulticlass::ParameterObserverCVMulticlass(
    bool compute_ROC, bool compute_PRC, bool compute_conf_matrices,
    bool verbose)
    : ParameterObserverCV(verbose), m_initialized(false),
      m_compute_ROC(compute_ROC), m_compute_PRC(compute_PRC),
      m_compute_conf_matrices(compute_conf_matrices), m_fold_ROC_graphs(NULL),
      m_fold_PRC_graphs(NULL), m_conf_matrices(NULL), m_num_classes(0),
      m_pred_labels(NULL), m_true_labels(NULL),
      m_binary_evaluations(new CDynamicObjectArray())
{
}

ParameterObserverCVMulticlass::~ParameterObserverCVMulticlass()
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
}

void ParameterObserverCVMulticlass::append_binary_evaluation(
    CBinaryClassEvaluation* evaluation)
{
	m_binary_evaluations->push_back(evaluation);
}

CBinaryClassEvaluation*
ParameterObserverCVMulticlass::get_binary_evaluation(int32_t idx)
{
	return (CBinaryClassEvaluation*)m_binary_evaluations->get_element_safe(idx);
}

void ParameterObserverCVMulticlass::initialize(std::string name)
{
	if (m_initialized)
	{
		return;
	}
	else if (m_observations.size() == 0)
	{
		SG_SERROR(
		    "Cannot run %s because the observer didn't catch any observation.",
		    name.c_str());
		return;
	}

	m_num_folds = m_observations[0]->get_num_folds();
	m_num_runs = m_observations[0]->get_num_runs();

	m_num_classes = ((CMulticlassLabels*)m_observations[0]->get_expose_labels())
	                    ->get_num_classes();

	if (m_compute_ROC)
	{
		SG_SDEBUG(
		    "Allocating %d ROC graphs\n",
		    m_num_folds * m_num_runs * m_num_classes)
		m_fold_ROC_graphs = SG_MALLOC(
		    SGMatrix<float64_t>, m_num_folds * m_num_runs * m_num_classes);
		for (int32_t i = 0; i < m_num_folds * m_num_runs * m_num_classes; i++)
			new (&m_fold_ROC_graphs[i]) SGMatrix<float64_t>();
	}

	if (m_compute_PRC)
	{
		SG_SDEBUG(
		    "Allocating %d PRC graphs\n",
		    m_num_folds * m_num_runs * m_num_classes)
		m_fold_PRC_graphs = SG_MALLOC(
		    SGMatrix<float64_t>, m_num_folds * m_num_runs * m_num_classes);
		for (int32_t i = 0; i < m_num_folds * m_num_runs * m_num_classes; i++)
			new (&m_fold_PRC_graphs[i]) SGMatrix<float64_t>();
	}

	if (m_binary_evaluations->get_num_elements())
		m_evaluations_results = SGVector<float64_t>(
		    m_num_folds * m_num_runs * m_num_classes *
		    m_binary_evaluations->get_num_elements());

	m_accuracies = SGVector<float64_t>(m_num_folds * m_num_runs);

	if (m_compute_conf_matrices)
	{
		m_conf_matrices =
		    SG_MALLOC(SGMatrix<int32_t>, m_num_folds * m_num_runs);
		for (int32_t i = 0; i < m_num_folds * m_num_runs; i++)
			new (&m_conf_matrices[i]) SGMatrix<int32_t>();
	}

	m_initialized = true;

	/* Compute things */
	for (auto o : m_observations)
	{
		for (auto f : o->get_folds_results())
		{
			m_pred_labels = (CMulticlassLabels*)f->get_test_result();
			m_true_labels = (CMulticlassLabels*)f->get_test_true_result();
			compute(o, f);
		}
	}
}

void ParameterObserverCVMulticlass::compute(
    CrossValidationStorage* storage, CrossValidationFoldStorage* fold)
{
	CROCEvaluation eval_ROC;
	CPRCEvaluation eval_PRC;
	int32_t n_evals = m_binary_evaluations->get_num_elements();
	for (int32_t c = 0; c < m_num_classes; c++)
	{
		SG_SDEBUG(
		    "Computing ROC for run %d fold %d class %d",
		    fold->get_current_run_index(), fold->get_current_fold_index(), c)
		CBinaryLabels* pred_labels_binary =
		    m_pred_labels->get_binary_for_class(c);
		CBinaryLabels* true_labels_binary =
		    m_true_labels->get_binary_for_class(c);
		if (m_compute_ROC)
		{
			eval_ROC.evaluate(pred_labels_binary, true_labels_binary);
			m_fold_ROC_graphs[fold->get_current_run_index() * m_num_folds *
			                      m_num_classes +
			                  fold->get_current_fold_index() * m_num_classes +
			                  c] = eval_ROC.get_ROC();
		}
		if (m_compute_PRC)
		{
			eval_PRC.evaluate(pred_labels_binary, true_labels_binary);
			m_fold_PRC_graphs[fold->get_current_run_index() * m_num_folds *
			                      m_num_classes +
			                  fold->get_current_fold_index() * m_num_classes +
			                  c] = eval_PRC.get_PRC();
		}

		for (int32_t i = 0; i < n_evals; i++)
		{
			CBinaryClassEvaluation* evaluator =
			    (CBinaryClassEvaluation*)m_binary_evaluations->get_element_safe(
			        i);
			m_evaluations_results[fold->get_current_run_index() * m_num_folds *
			                          m_num_classes * n_evals +
			                      fold->get_current_fold_index() *
			                          m_num_classes * n_evals +
			                      c * n_evals + i] =
			    evaluator->evaluate(pred_labels_binary, true_labels_binary);
			SG_UNREF(evaluator);
		}

		SG_UNREF(pred_labels_binary);
		SG_UNREF(true_labels_binary);
	}
	CMulticlassAccuracy accuracy;

	m_accuracies[fold->get_current_run_index() * m_num_folds +
	             fold->get_current_fold_index()] =
	    accuracy.evaluate(m_pred_labels, m_true_labels);

	if (m_compute_conf_matrices)
	{
		m_conf_matrices[fold->get_current_run_index() * m_num_folds +
		                fold->get_current_fold_index()] =
		    CMulticlassAccuracy::get_confusion_matrix(
		        m_pred_labels, m_true_labels);
	}
}

SGMatrix<float64_t> ParameterObserverCVMulticlass::get_fold_ROC(
    int32_t run, int32_t fold, int32_t c)
{
	if (!m_initialized)
		initialize("get_fold_ROC");

	ASSERT(0 <= run)
	ASSERT(run < m_num_runs)
	ASSERT(0 <= fold)
	ASSERT(fold < m_num_folds)
	ASSERT(0 <= c)
	ASSERT(c < m_num_classes)
	REQUIRE(m_compute_ROC, "ROC computation was not enabled\n")
	return m_fold_ROC_graphs[run * m_num_folds * m_num_classes +
	                         fold * m_num_classes + c];
}

SGMatrix<float64_t> ParameterObserverCVMulticlass::get_fold_PRC(
    int32_t run, int32_t fold, int32_t c)
{
	if (!m_initialized)
		initialize("get_fold_PRC");

	ASSERT(0 <= run)
	ASSERT(run < m_num_runs)
	ASSERT(0 <= fold)
	ASSERT(fold < m_num_folds)
	ASSERT(0 <= c)
	ASSERT(c < m_num_classes)
	REQUIRE(m_compute_PRC, "PRC computation was not enabled\n")
	return m_fold_PRC_graphs[run * m_num_folds * m_num_classes +
	                         fold * m_num_classes + c];
}

float64_t ParameterObserverCVMulticlass::get_fold_evaluation_result(
    int32_t run, int32_t fold, int32_t c, int32_t e)
{
	if (!m_initialized)
		initialize("get_fold_evaluation_result");

	ASSERT(0 <= run)
	ASSERT(run < m_num_runs)
	ASSERT(0 <= fold)
	ASSERT(fold < m_num_folds)
	ASSERT(0 <= c)
	ASSERT(c < m_num_classes)
	ASSERT(0 <= e)
	int32_t n_evals = m_binary_evaluations->get_num_elements();
	ASSERT(e < n_evals)
	return m_evaluations_results[run * m_num_folds * m_num_classes * n_evals +
	                             fold * m_num_classes * n_evals + c * n_evals +
	                             e];
}

float64_t
ParameterObserverCVMulticlass::get_fold_accuracy(int32_t run, int32_t fold)
{
	if (!m_initialized)
		initialize("get_fold_accuracy");

	ASSERT(0 <= run)
	ASSERT(run < m_num_runs)
	ASSERT(0 <= fold)
	ASSERT(fold < m_num_folds)
	return m_accuracies[run * m_num_folds + fold];
}

SGMatrix<int32_t>
ParameterObserverCVMulticlass::get_fold_conf_matrix(int32_t run, int32_t fold)
{
	if (!m_initialized)
		initialize("get_fold_conf_matrix");

	ASSERT(0 <= run)
	ASSERT(run < m_num_runs)
	ASSERT(0 <= fold)
	ASSERT(fold < m_num_folds)
	REQUIRE(
	    m_compute_conf_matrices,
	    "Confusion matrices computation was not enabled\n")
	return m_conf_matrices[run * m_num_folds + fold];
}