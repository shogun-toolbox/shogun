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
	// using bias
	bool use_bias = true;
	// set epsilon to an order of magnitude lower than test tolerance
	double epsilon = 1E-6;

	// create some easy regression data: y = 3x + 2 (defined in LinearTestEnvironment.h)
	std::shared_ptr<LinearRegressionDataGenerator> mockData =
			linear_test_env->getOneDimensionalRegressionData(use_bias);

	// fetch data from test environment
	CDenseFeatures<float64_t>* train_feats = mockData->get_features_train();
	CDenseFeatures<float64_t>* test_feats = mockData->get_features_test();

	CRegressionLabels* labels_test = mockData->get_labels_test();
	CRegressionLabels* labels_train = mockData->get_labels_train();

	// train model
	CLibLinearRegression* lr =
	    new CLibLinearRegression(1., train_feats, labels_train);
	lr->set_use_bias(use_bias);
	lr->set_epsilon(epsilon);
	lr->train();

	// run predictions
	CRegressionLabels* predicted_labels =
	    lr->apply(test_feats)->as<CRegressionLabels>();

	// test coefficients and bias
	EXPECT_NEAR(lr->get_w()[0], mockData->get_coefficient(0), 1E-5);
	EXPECT_NEAR(lr->get_bias(), mockData->get_bias(), 1E-5);

	// test predictions
	for (index_t i = 0; i < mockData->get_test_size(); ++i)
		EXPECT_NEAR(predicted_labels->get_label(i), labels_test->get_label(i), 1E-5);

	/* clean up */
	SG_UNREF(predicted_labels);
	SG_UNREF(lr);
}

TEST(LibLinearRegression, lr_without_bias)
{
	// not using bias
	bool use_bias = false;
	// set epsilon to an order of magnitude lower than test tolerance
	double epsilon = 1E-6;

	// create some easy regression data: y = 3x + 2 (defined in LinearTestEnvironment.h)
	std::shared_ptr<LinearRegressionDataGenerator> mockData =
			linear_test_env->getOneDimensionalRegressionData(use_bias);

	// fetch data from test environment
	CDenseFeatures<float64_t>* train_feats = mockData->get_features_train();
	CDenseFeatures<float64_t>* test_feats = mockData->get_features_test();

	CRegressionLabels* labels_test = mockData->get_labels_test();
	CRegressionLabels* labels_train = mockData->get_labels_train();

	// train model
	CLibLinearRegression* lr =
			new CLibLinearRegression(1., train_feats, labels_train);
	lr->set_use_bias(use_bias);
	lr->set_epsilon(epsilon);
	lr->train();

	// run predictions
	CRegressionLabels* predicted_labels =
			lr->apply(test_feats)->as<CRegressionLabels>();

	// test coefficients and bias
	EXPECT_NEAR(lr->get_w()[0], mockData->get_coefficient(0), 1E-5);
	EXPECT_NEAR(lr->get_bias(), mockData->get_bias(), 1E-5);

	// test predictions
	for (index_t i = 0; i < mockData->get_test_size(); ++i)
	EXPECT_NEAR(predicted_labels->get_label(i), labels_test->get_label(i), 1E-5);

	/* clean up */
	SG_UNREF(predicted_labels);
	SG_UNREF(lr);
}