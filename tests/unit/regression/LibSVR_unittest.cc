/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/regression/svr/LibSVR.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LibSVR,epsilon_svr_apply)
{
	const int32_t kernel_cache=0;
	const float64_t rbf_width=1;
	const float64_t svm_C=1;
	const float64_t svm_eps=0.1;

	/* create some easy regression data: 1d noisy sine wave */
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> feat_test(1, n);
	SGVector<float64_t> lab_train(n);
	SGVector<float64_t> lab_test(n);

	/* a one dimensional quadratic function */
	feat_train[0]=-2;
	feat_train[1]=-1;
	feat_train[2]=0;
	feat_train[3]=1;
	feat_train[4]=2;

	lab_train[0]=4;
	lab_train[1]=1;
	lab_train[2]=0;
	lab_train[3]=1;
	lab_train[4]=4;

	feat_test[0]=-2.2;
	feat_test[1]=-1.1;
	feat_test[2]=0.2;
	feat_test[3]=1.3;
	feat_test[4]=1.9;

	lab_test[0]=4.84;
	lab_test[1]=1.21;
	lab_test[2]=0.04;
	lab_test[3]=1.69;
	lab_test[4]=3.61;

	/* shogun representation */
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);
	CRegressionLabels* labels_test=new CRegressionLabels(lab_test);
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(
			feat_test);

	CGaussianKernel* kernel=new CGaussianKernel(kernel_cache, rbf_width);
	kernel->init(features_train, features_train);

	LIBSVR_SOLVER_TYPE st=LIBSVR_EPSILON_SVR;
	CLibSVR* svm=new CLibSVR(svm_C, svm_eps, kernel, labels_train, st);
	svm->train();

	/* predict */
	CRegressionLabels* predicted_labels=CLabelsFactory::to_regression(
			svm->apply(features_test));

	/* LibSVM regression comparison (with easy.py script) */
	EXPECT_NEAR(predicted_labels->get_labels()[0], 2.44343, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[1], 1.25466, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[2], 0.313201, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[3], 1.57767, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[4], 2.34949, 1E-5);

	EXPECT_NEAR(CMath::abs(svm->get_bias()), 1.60903, 1E-5);
	EXPECT_EQ(svm->get_num_support_vectors(), 5);

	 /* clean up */
	SG_UNREF(labels_test)
	SG_UNREF(predicted_labels);
	SG_UNREF(svm);
}

TEST(LibSVR,nu_svr_apply)
{
	const int32_t kernel_cache=0;
	const float64_t rbf_width=1;
	const float64_t svm_C=1;
	const float64_t svm_nu=0.1;

	/* create some easy regression data: 1d noisy sine wave */
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> feat_test(1, n);
	SGVector<float64_t> lab_train(n);
	SGVector<float64_t> lab_test(n);

	/* a one dimensional quadratic function */
	feat_train[0]=-2;
	feat_train[1]=-1;
	feat_train[2]=0;
	feat_train[3]=1;
	feat_train[4]=2;

	lab_train[0]=4;
	lab_train[1]=1;
	lab_train[2]=0;
	lab_train[3]=1;
	lab_train[4]=4;

	feat_test[0]=-2.2;
	feat_test[1]=-1.1;
	feat_test[2]=0.2;
	feat_test[3]=1.3;
	feat_test[4]=1.9;

	lab_test[0]=4.84;
	lab_test[1]=1.21;
	lab_test[2]=0.04;
	lab_test[3]=1.69;
	lab_test[4]=3.61;

	/* shogun representation */
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);
	CRegressionLabels* labels_test=new CRegressionLabels(lab_test);
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(
			feat_test);

	CGaussianKernel* kernel=new CGaussianKernel(kernel_cache, rbf_width);
	kernel->init(features_train, features_train);

	LIBSVR_SOLVER_TYPE st=LIBSVR_NU_SVR;
	CLibSVR* svm=new CLibSVR(svm_C, svm_nu, kernel, labels_train, st);
	svm->train();

	/* predict */
	CRegressionLabels* predicted_labels=CLabelsFactory::to_regression(
			svm->apply(features_test));

	/* LibSVM regression comparison (with easy.py script) */
	EXPECT_NEAR(predicted_labels->get_labels()[0], 2.18062, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[1], 2.04357, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[2], 1.82819, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[3], 2.09295, 1E-5);
	EXPECT_NEAR(predicted_labels->get_labels()[4], 2.17949, 1E-5);

	EXPECT_NEAR(CMath::abs(svm->get_bias()), 2.0625, 1E-5);
	EXPECT_EQ(svm->get_num_support_vectors(), 3);

	 /* clean up */
	SG_UNREF(labels_test)
	SG_UNREF(predicted_labels);
	SG_UNREF(svm);
}
