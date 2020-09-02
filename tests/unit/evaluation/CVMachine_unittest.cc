/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Yuhui Liu
 */

#include <shogun/evaluation/CVMachine.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/evaluation/MeanAbsoluteError.h>
#include <shogun/evaluation/MeanSquaredError.h>
#include <gtest/gtest.h>
#include "environments/RegressionTestEnvironment.h"
extern RegressionTestEnvironment* regression_test_env;


using namespace shogun;

TEST(CV_Machine, fit)
{
	auto train_feats = regression_test_env->get_features_train();
	auto test_feats = regression_test_env->get_features_test();

	auto labels_test = regression_test_env->get_labels_test();
	auto labels_train = regression_test_env->get_labels_train();
    auto strategy = std::make_shared<CrossValidationSplitting>(labels_train, 2);
    strategy->put(random::kSeed, 2);
    auto machine = std::make_shared<LinearRidgeRegression>();
    auto evaluation_criterion = std::make_shared<MeanSquaredError>();
    auto cv = std::make_shared<CrossValidation>(machine, strategy, evaluation_criterion);
    std::vector<std::pair<std::string_view, std::vector<double>>> params{{"tau", {0.1, 0.2, 0.5, 0.8, 2}}};
    auto cv_wrapper = std::make_shared<CVMachine<LinearRidgeRegression>>(params, cv);
    cv_wrapper->fit(train_feats, labels_train);

    auto pred = machine->apply(test_feats);
}
