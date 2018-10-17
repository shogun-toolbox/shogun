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
    index_t n=5;

    SGMatrix<float64_t> feat_train(1, n);
    SGMatrix<float64_t> feat_test(1, n);
    SGVector<float64_t> lab_train(n);

    feat_train[0]=-2;
    feat_train[1]=-1;
    feat_train[2]=0;
    feat_train[3]=0.5;
    feat_train[4]=1.5;

    lab_train[0]=-4;
    lab_train[1]=-1;
    lab_train[2]=2;
    lab_train[3]=3.5;
    lab_train[4]=6.5;

    feat_test[0]=-3;
    feat_test[1]=-2.5;
    feat_test[2]=-1.5;
    feat_test[3]=3;
    feat_test[4]=5;

    /* shogun representation */
    CRegressionLabels* labels_train=new CRegressionLabels(lab_train);
    CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
            feat_train);
    CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(
            feat_test);


    CLibLinearRegression* lr=new CLibLinearRegression(1.5, features_train, labels_train);
    lr->set_use_bias(use_bias);
    lr->set_epsilon(epsilon);
    lr->train();

    CRegressionLabels* predicted_labels =
            lr->apply(features_test)->as<CRegressionLabels>();

    EXPECT_NEAR(lr->get_w()[0], 3.0, 1E-5);
    EXPECT_NEAR(lr->get_bias(), 2.0, 1E-5);

    EXPECT_NEAR(predicted_labels->get_labels()[0], -7.0, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[1], -5.5, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[2], -2.5, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[3], 11.0, 1E-5);
    EXPECT_NEAR(predicted_labels->get_labels()[4], 17.0, 1E-5);

    /* clean up */
    SG_UNREF(predicted_labels);
    SG_UNREF(lr);
}
