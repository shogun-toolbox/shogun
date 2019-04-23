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
#include <gtest/gtest.h>

#include <shogun/base/init.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/lib/observers/ParameterObserverCV.h>
#include <shogun/regression/KernelRidgeRegression.h>

#include "environments/RegressionTestEnvironment.h"
#include <memory>

using namespace shogun;

/**
 * This test was inspired by the meta example
 * examples/undocumented/libshogun/evaluation_cross_validation_regression.cpp
 * written by Heiko Strathmann.
 */

extern RegressionTestEnvironment* regression_test_env;

/* data matrix dimensions */
index_t num_vectors = 100;
index_t num_features = 1;

/* training label data */
SGVector<float64_t> lab(num_vectors);
std::shared_ptr<DenseFeatures<float64_t>> features = NULL;
std::shared_ptr<RegressionLabels> labels = NULL;

std::shared_ptr<ParameterObserverCV> generate(bool locked = true)
{
	/* training features */
	features = regression_test_env->get_features_train();


	/* training labels */
	labels = regression_test_env->get_labels_train();

	/* kernel */
	auto kernel = std::make_shared<LinearKernel>();
	kernel->init(features, features);

	/* kernel ridge regression*/
	float64_t tau = 0.0001;
	auto krr =
	    std::make_shared<KernelRidgeRegression>(tau, kernel, labels);

	/* evaluation criterion */
	auto eval_crit = std::make_shared<MeanSquaredError>();

	/* splitting strategy */
	index_t n_folds = 5;
	auto splitting =
	    std::make_shared<CrossValidationSplitting>(labels, n_folds);

	/* cross validation instance, 100 runs, 95% confidence interval */
	auto cross =
	    std::make_shared<CrossValidation>(krr, features, labels, splitting, eval_crit);
	cross->set_num_runs(10);
	cross->set_autolock(locked);

	/* Create the parameter observer */
	auto par = std::make_shared<ParameterObserverCV>();
	cross->subscribe(par);

	/* actual evaluation */
	auto result = cross->evaluate()->as<CrossValidationResult>();

	return par;
}

TEST(ParameterObserverCV, get_observations_locked)
{
	std::shared_ptr<ParameterObserverCV> par{generate(true)};

	for (index_t i = 0; i < par->get<index_t>("num_observations"); i++)
	{
		auto name = par->get_observation(i)->get<std::string>("name");
		auto run = par->get_observation(i)->get(name);
		ASSERT(run)
		EXPECT_EQ(run->get<index_t>("num_runs"), 10);
		EXPECT_EQ(run->get<index_t>("num_folds"), 5);
		EXPECT_TRUE(run->get("labels")->equals(labels));
		for (int j = 0; j < 5; j++)
		{
			auto fold = run->get("folds", j);
			EXPECT_EQ(fold->get<index_t>("run_index"), i);
			EXPECT_EQ(fold->get<index_t>("fold_index"), j);
			EXPECT_TRUE(
			    fold->get<SGVector<index_t>>("train_indices").size() != 0);
			EXPECT_TRUE(
			    fold->get<SGVector<index_t>>("test_indices").size() != 0);
			EXPECT_TRUE(fold->get<Machine>("trained_machine") != NULL);
			EXPECT_TRUE(
			    fold->get<Labels>("predicted_labels")->get_num_labels() != 0);
			EXPECT_TRUE(
			    fold->get<Labels>("ground_truth_labels")->get_num_labels() !=
			    0);
			EXPECT_TRUE(fold->get<float64_t>("evaluation_result") != 0);
		}

	}
}

TEST(ParameterObserverCV, get_observations_unlocked)
{
	std::shared_ptr<ParameterObserverCV> par{generate(false)};

	for (index_t i = 0; i < par->get<index_t>("num_observations"); i++)
	{
		auto name = par->get_observation(i)->get<std::string>("name");
		auto run = par->get_observation(i)->get(name);
		ASSERT(run)
		EXPECT_EQ(run->get<index_t>("num_runs"), 10);
		EXPECT_EQ(run->get<index_t>("num_folds"), 5);
		EXPECT_TRUE(run->get("labels")->equals(labels));
		for (int j = 0; j < run->get<index_t>("num_folds"); j++)
		{
			auto fold = run->get("folds", j);
			EXPECT_EQ(fold->get<index_t>("run_index"), i);
			EXPECT_EQ(fold->get<index_t>("fold_index"), j);
			EXPECT_TRUE(
			    fold->get<SGVector<index_t>>("train_indices").size() != 0);
			EXPECT_TRUE(
			    fold->get<SGVector<index_t>>("test_indices").size() != 0);
			EXPECT_TRUE(fold->get<Machine>("trained_machine") != NULL);
			EXPECT_TRUE(
			    fold->get<Labels>("predicted_labels")->get_num_labels() != 0);
			EXPECT_TRUE(
			    fold->get<Labels>("ground_truth_labels")->get_num_labels() !=
			    0);
			EXPECT_TRUE(fold->get<float64_t>("evaluation_result") != 0);
		}

	}
}
