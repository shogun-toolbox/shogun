/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */
#include <gtest/gtest.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/regression/svr/LibLinearRegression.h>

using namespace shogun;

TEST(LibLinearRegression, lr_with_bias)
{
    // using bias
    bool use_bias = true;
    // set epsilon to an order of magnitude lower than test tolerance
    double epsilon = 1E-6;

    /* create some easy regression data: y = 3x + 2 */
    index_t n=201;
    float64_t m = 3;
	float64_t b = 2;
	double initial_value = -100;

    SGMatrix<float64_t> feat_train(1, n);
    SGMatrix<float64_t> feat_test(1, n);
    SGVector<float64_t> lab_train(n);

	for (index_t i = 0; i<n; ++i)
		feat_train[i] = initial_value + i * 0.5;

    for (index_t i = 0; i<n; ++i)
    	lab_train[i] = m * feat_train[i] + b;

    feat_test[0]=-3.2;
    feat_test[1]=-2.1;
    feat_test[2]=-1.4;
    feat_test[3]=3.05;
    feat_test[4]=5.7;

    /* shogun representation */
    CRegressionLabels* labels_train=new CRegressionLabels(lab_train);
    CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
            feat_train);
    CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(
            feat_test);


    CLibLinearRegression* lr=new CLibLinearRegression(1., features_train, labels_train);
    lr->set_use_bias(use_bias);
    lr->set_epsilon(epsilon);
    lr->train();

    CRegressionLabels* predicted_labels =
            lr->apply(features_test)->as<CRegressionLabels>();

    EXPECT_NEAR(lr->get_w()[0], m, 1E-5);
    EXPECT_NEAR(lr->get_bias(), b, 1E-5);

    EXPECT_NEAR(predicted_labels->get_labels()[0], -7.6, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[1], -4.3, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[2], -2.2, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[3], 11.15, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[4], 19.1, 1E-5);

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

	/* create some easy regression data: y = 3x*/
	index_t n=201;
	float64_t m = 3;
	float64_t b = 0;
	double initial_value = -100;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> feat_test(1, n);
	SGVector<float64_t> lab_train(n);

	for (index_t i = 0; i<n; ++i)
		feat_train[i] = initial_value + i * 0.5;

	for (index_t i = 0; i<n; ++i)
		lab_train[i] = m * feat_train[i] + b;

	feat_test[0]=-3.2;
	feat_test[1]=-2.1;
	feat_test[2]=-1.4;
	feat_test[3]= 3.05;
	feat_test[4]= 5.7;

	/* shogun representation */
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(
			feat_test);


	CLibLinearRegression* lr=new CLibLinearRegression(1., features_train, labels_train);
	lr->set_use_bias(use_bias);
	lr->set_epsilon(epsilon);
	lr->train();

	CRegressionLabels* predicted_labels =
			lr->apply(features_test)->as<CRegressionLabels>();

	EXPECT_NEAR(lr->get_w()[0], 3, 1E-5);

	EXPECT_NEAR(predicted_labels->get_labels()[0], -9.6, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[1], -6.3, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[2], -4.2, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[3], 9.15, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[4], 17.1, 1E-5);

	/* clean up */
	SG_UNREF(predicted_labels);
	SG_UNREF(lr);
}