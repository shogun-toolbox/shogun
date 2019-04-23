/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

#include <gtest/gtest.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/regression/svr/LibLinearRegression.h>

#include "environments/LinearTestEnvironment.h"

using namespace shogun;

// get the test environment
extern LinearTestEnvironment* linear_test_env;

TEST(LibLinearRegression, lr_with_bias)
{
	bool use_bias = true;
	double epsilon = 1E-6;

	std::shared_ptr<LinearRegressionDataGenerator> mockData =
			linear_test_env->get_one_dimensional_regression_data(use_bias);

	auto train_feats = mockData->get_features_train();
	auto test_feats = mockData->get_features_test();

	auto labels_test = mockData->get_labels_test();
	auto labels_train = mockData->get_labels_train();

	auto lr =
		std::make_shared<LibLinearRegression>(1., train_feats, labels_train);
	lr->set_use_bias(use_bias);
	lr->set_epsilon(epsilon);
	lr->set_tube_epsilon(epsilon);
	lr->train();

	auto predicted_labels =
		lr->apply(test_feats)->as<RegressionLabels>();

	EXPECT_NEAR(lr->get_w()[0], mockData->get_coefficient(0), 1E-5);
	EXPECT_NEAR(lr->get_bias(), mockData->get_bias(), 1E-5);

	for (index_t i = 0; i < mockData->get_test_size(); ++i)
		EXPECT_NEAR(predicted_labels->get_label(i), labels_test->get_label(i), 1E-5);

	/* clean up */


}

TEST(LibLinearRegression, lr_without_bias)
{
	// not using bias
	bool use_bias = false;
	double epsilon = 1E-6;

	std::shared_ptr<LinearRegressionDataGenerator> mockData =
			linear_test_env->get_one_dimensional_regression_data(use_bias);

	auto train_feats = mockData->get_features_train();
	auto test_feats = mockData->get_features_test();

	auto labels_test = mockData->get_labels_test();
	auto labels_train = mockData->get_labels_train();

	auto lr =
			std::make_shared<LibLinearRegression>(1., train_feats, labels_train);
	lr->set_use_bias(use_bias);
	lr->set_epsilon(epsilon);
	lr->set_tube_epsilon(epsilon);
	lr->train();

	auto predicted_labels =
			lr->apply(test_feats)->as<RegressionLabels>();

	EXPECT_NEAR(lr->get_w()[0], mockData->get_coefficient(0), 1E-5);
	EXPECT_NEAR(lr->get_bias(), mockData->get_bias(), 1E-5);

	for (index_t i = 0; i < mockData->get_test_size(); ++i)
		EXPECT_NEAR(predicted_labels->get_label(i), labels_test->get_label(i), 1E-5);

	/* clean up */


}