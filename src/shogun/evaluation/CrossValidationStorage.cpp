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
*/

#include "CrossValidationStorage.h"
#include <shogun/labels/Labels.h>
#include <shogun/machine/Machine.h>

using namespace shogun;

CrossValidationFoldStorage::CrossValidationFoldStorage() : CSGObject()
{
	m_current_run_index = 0;
	m_current_fold_index = 0;
	m_trained_machine = NULL;
	m_test_result = NULL;
	m_test_true_result = NULL;

	SG_ADD(
	    &m_current_run_index, "m_current_run_index",
	    "The current run index of this fold", MS_AVAILABLE)
	SG_ADD(
	    &m_current_fold_index, "m_current_fold_index", "The current fold index",
	    MS_AVAILABLE)
	SG_ADD(
	    (CSGObject**)&m_trained_machine, "m_trained_machine",
	    "The machine trained by this fold", MS_AVAILABLE)
	SG_ADD(
	    (CSGObject**)&m_test_result, "m_test_result",
	    "The test result of this fold", MS_AVAILABLE)
	SG_ADD(
	    (CSGObject**)&m_test_true_result, "m_test_true_result",
	    "The true test result for this fold", MS_AVAILABLE)
}

CrossValidationFoldStorage::~CrossValidationFoldStorage()
{
	SG_UNREF(m_test_result);
	SG_UNREF(m_test_true_result);
	SG_UNREF(m_trained_machine);
}

void CrossValidationFoldStorage::set_run_index(index_t run_index)
{
	m_current_run_index = run_index;
}

void CrossValidationFoldStorage::set_fold_index(index_t fold_index)
{
	m_current_fold_index = fold_index;
}

void CrossValidationFoldStorage::set_train_indices(SGVector<index_t> indices)
{
	m_train_indices = indices;
}

void CrossValidationFoldStorage::set_test_indices(SGVector<index_t> indices)
{
	m_test_indices = indices;
}

void CrossValidationFoldStorage::set_trained_machine(CMachine* machine)
{
	SG_REF(machine)
	SG_UNREF(m_trained_machine)
	m_trained_machine = machine;
}

void CrossValidationFoldStorage::set_test_result(CLabels* results)
{
	SG_REF(results)
	SG_UNREF(m_test_result)
	m_test_result = results;
}

void CrossValidationFoldStorage::set_test_true_result(CLabels* results)
{
	SG_REF(results)
	SG_UNREF(m_test_true_result)
	m_test_true_result = results;
}

void CrossValidationFoldStorage::post_update_results()
{
}

void CrossValidationFoldStorage::set_evaluation_result(float64_t result)
{
	m_evaluation_result = result;
}

index_t CrossValidationFoldStorage::get_current_run_index() const
{
	return m_current_run_index;
}

index_t CrossValidationFoldStorage::get_current_fold_index() const
{
	return m_current_fold_index;
}

const SGVector<index_t>& CrossValidationFoldStorage::get_train_indices() const
{
	return m_train_indices;
}

const SGVector<index_t>& CrossValidationFoldStorage::get_test_indices() const
{
	return m_test_indices;
}

CMachine* CrossValidationFoldStorage::get_trained_machine() const
{
	return m_trained_machine;
}

CLabels* CrossValidationFoldStorage::get_test_result() const
{
	return m_test_result;
}

CLabels* CrossValidationFoldStorage::get_test_true_result() const
{
	return m_test_true_result;
}

float64_t CrossValidationFoldStorage::get_evaluation_result() const
{
	return m_evaluation_result;
}

CrossValidationFoldStorage* CrossValidationStorage::get_fold(int fold) const
{
	REQUIRE(
	    fold < get_num_folds(), "The fold number must be less than %i",
	    get_num_folds())

	CrossValidationFoldStorage* fld = m_folds_results[fold];
	SG_REF(fld);
	return fld;
}

bool CrossValidationFoldStorage::
operator==(const CrossValidationFoldStorage& rhs) const
{
	return m_current_run_index == rhs.m_current_run_index &&
	       m_current_fold_index == rhs.m_current_fold_index &&
	       // m_train_indices.equals(rhs.m_train_indices) &&
	       // m_test_indices.equals(rhs.m_test_indices) &&
	       m_trained_machine->equals(rhs.m_trained_machine) &&
	       m_test_result->equals(rhs.m_test_result) &&
	       m_test_true_result->equals(rhs.m_test_true_result) &&
	       m_evaluation_result == rhs.m_evaluation_result;
}

/** CrossValidationStorage **/

CrossValidationStorage::CrossValidationStorage() : CSGObject()
{
	m_num_runs = 0;
	m_num_folds = 0;
	m_expose_labels = NULL;

	SG_ADD(
	    &m_num_runs, "m_num_runs", "The total number of cross-validation runs",
	    MS_AVAILABLE)
	SG_ADD(
	    &m_num_folds, "m_num_folds",
	    "The total number of cross-validation folds", MS_AVAILABLE)
	SG_ADD(
	    (CSGObject**)&m_expose_labels, "m_expose_labels",
	    "The labels used for this cross-validation", MS_AVAILABLE)
}

CrossValidationStorage::~CrossValidationStorage()
{
	SG_UNREF(m_expose_labels);
	for (auto i : m_folds_results)
		SG_UNREF(i)
}

void CrossValidationStorage::set_num_runs(index_t num_runs)
{
	m_num_runs = num_runs;
}

void CrossValidationStorage::set_num_folds(index_t num_folds)
{
	m_num_folds = num_folds;
}

void CrossValidationStorage::set_expose_labels(CLabels* labels)
{
	SG_REF(labels)
	SG_UNREF(m_expose_labels)
	m_expose_labels = labels;
}

void CrossValidationStorage::post_init()
{
}

index_t CrossValidationStorage::get_num_runs() const
{
	return m_num_runs;
}

index_t CrossValidationStorage::get_num_folds() const
{
	return m_num_folds;
}

CLabels* CrossValidationStorage::get_expose_labels() const
{
	return m_expose_labels;
}

void CrossValidationStorage::append_fold_result(
    CrossValidationFoldStorage* result)
{
	SG_REF(result);
	m_folds_results.push_back(result);
}

bool CrossValidationStorage::operator==(const CrossValidationStorage& rhs) const
{
	auto member_vars = m_num_runs == rhs.m_num_runs &&
	                   m_num_folds == rhs.m_num_folds &&
	                   m_expose_labels->equals(rhs.m_expose_labels);

	if (!member_vars)
		return member_vars;

	if (rhs.m_folds_results.size() != m_folds_results.size())
		return false;
	for (index_t i = 0; i < m_folds_results.size(); i++)
	{
		if (!(m_folds_results[i] == rhs.m_folds_results[i]))
			return false;
	}
	return member_vars;
}