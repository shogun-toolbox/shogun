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

#include <shogun/base/range.h>

using namespace shogun;

CrossValidationFoldStorage::CrossValidationFoldStorage() : EvaluationResult()
{
	SG_ADD(
	    &m_current_run_index, kCurrentRunIndex, "The current run index of this fold",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_current_fold_index, kCurrentFoldIndex, "The current fold index",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_trained_machine, kTrainedMachine,
	    "The machine trained by this fold", ParameterProperties::HYPER);
	SG_ADD(
	    &m_test_result, kTestResult,
	    "The test result of this fold", ParameterProperties::HYPER);
	SG_ADD(
	    &m_test_true_result, kTestTrueResult,
	    "The true test result for this fold", ParameterProperties::HYPER);
	SG_ADD(
	    &m_train_indices, kTrainIndices, "Indices used for training",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_test_indices, kTestIndices, "Indices used for testing",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_evaluation_result, kEvalulationResult, "Result of the evaluation",
	    ParameterProperties::HYPER);
}

CrossValidationFoldStorage::~CrossValidationFoldStorage() = default;

void CrossValidationFoldStorage::post_update_results()
{
}

void CrossValidationFoldStorage::print_result()
{
}

std::shared_ptr<SGObject> CrossValidationFoldStorage::create_empty() const
{
	return std::make_shared<CrossValidationFoldStorage>();
}

/** CrossValidationStorage **/

CrossValidationStorage::CrossValidationStorage() : EvaluationResult()
{
	m_num_runs = 0;
	m_num_folds = 0;

	SG_ADD(
	    &m_num_runs, kNumRuns, "The total number of cross-validation runs",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_num_folds, kNumFold, "The total number of cross-validation folds",
	    ParameterProperties::HYPER);
	SG_ADD(
	    &m_original_labels, kOriginalLabels,
	    "The labels used for this cross-validation",
	    ParameterProperties::HYPER);
	this->watch_param(
	    "folds", &m_folds_results, AnyParameterProperties("Fold results"));
}

CrossValidationStorage::~CrossValidationStorage() = default;

void CrossValidationStorage::post_init()
{
}

void CrossValidationStorage::append_fold_result(
    std::shared_ptr<CrossValidationFoldStorage> result)
{
	auto cloned = result->clone()->as<CrossValidationFoldStorage>();
	m_folds_results.push_back(cloned);
}

void CrossValidationStorage::print_result()
{
}

std::shared_ptr<SGObject> CrossValidationStorage::create_empty() const
{
	return std::make_shared<CrossValidationStorage>();
}
