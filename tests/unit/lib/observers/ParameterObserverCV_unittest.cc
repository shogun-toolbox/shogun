/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni
 *
 */
#include <gtest/gtest.h>

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

	/* Create the parameter observer */
	auto par = std::make_shared<ParameterObserverCV>();
	cross->subscribe(par);

	/* actual evaluation */
	auto result = cross->evaluate()->as<CrossValidationResult>();

	return par;
}

TEST(ParameterObserverCV, DISABLED_get_observations_locked)
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

TEST(ParameterObserverCV, DISABLED_get_observations_unlocked)
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
