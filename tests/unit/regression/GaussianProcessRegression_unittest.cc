/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(GaussianProcessRegression,apply_regression)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();

	// apply regression
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);
	SGVector<float64_t> prediction_vector=predictions->get_labels();

	/* do some checks against gpml toolbox*/
	// m =
	// 0.221198406887592
	// 0.537437461176145
	// 0.431605035301329
	EXPECT_LE(CMath::abs(prediction_vector[0]-0.221198406887592), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[1]-0.537437461176145), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[2]-0.431605035301329), 10E-15);

	// cleanup
	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression,apply_regression_larger_test)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;
	index_t n_test=5;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n_test);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;
	X_test[3]=2.7;
	X_test[4]=3.1;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();

	// apply regression
	CRegressionLabels* predictions=gpr->apply_regression(feat_test);
	SGVector<float64_t> prediction_vector=predictions->get_labels();

	/* do some checks against gpml toolbox*/
	// m =
	// 0.221198406887592
	// 0.537437461176145
	// 0.431605035301329
	// 0.373048041692408
	// 0.253688340068952
	EXPECT_LE(CMath::abs(prediction_vector[0]-0.221198406887592), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[1]-0.537437461176145), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[2]-0.431605035301329), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[3]-0.373048041692408), 10E-15);
	EXPECT_LE(CMath::abs(prediction_vector[4]-0.253688340068952), 10E-15);

	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression, apply_regression_on_training_features)
{
	// create some easy regression data: 1d noisy sine wave
	index_t ntr=5;

	SGMatrix<float64_t> feat_train(1, ntr);
	SGVector<float64_t> lab_train(ntr);

	feat_train[0]=1.25107;
	feat_train[1]=2.16097;
	feat_train[2]=0.00034;
	feat_train[3]=0.90699;
	feat_train[4]=0.44026;

	lab_train[0]=0.39635;
	lab_train[1]=0.00358;
	lab_train[2]=-1.18139;
	lab_train[3]=1.35533;
	lab_train[4]=-0.08232;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 0.02 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 0.02);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.25
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(0.25);

	// specify GP regression with exact inference
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, features_train,
			mean, labels_train, liklihood);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();

	// apply regression
	CRegressionLabels* predictions=gpr->apply_regression();
	SGVector<float64_t> prediction_vector=predictions->get_labels();

	// comparison of predictions with result from GPML package:
	// ymu =
	// 0.3732367
	// 0.0033694
	// -1.1118968
	// 1.2756631
	// -0.0774804
	EXPECT_NEAR(prediction_vector[0], 0.3732367, 1E-7);
	EXPECT_NEAR(prediction_vector[1], 0.0033694, 1E-7);
	EXPECT_NEAR(prediction_vector[2], -1.1118968, 1E-7);
	EXPECT_NEAR(prediction_vector[3], 1.2756631, 1E-7);
	EXPECT_NEAR(prediction_vector[4], -0.0774804, 1E-7);

	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression, get_mean_vector)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();

	// get mean value for each input feature
	SGVector<float64_t> mean_vector=gpr->get_mean_vector(feat_test);

	/* do some checks against gpml toolbox*/
	// m =
	// 0.221198406887592
	// 0.537437461176145
	// 0.431605035301329
	EXPECT_LE(CMath::abs(mean_vector[0]-0.221198406887592), 10E-15);
	EXPECT_LE(CMath::abs(mean_vector[1]-0.537437461176145), 10E-15);
	EXPECT_LE(CMath::abs(mean_vector[2]-0.431605035301329), 10E-15);

	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression, get_variance_vector_1)
{
	/* create some easy regression data: 1d noisy sine wave */
	index_t n=3;

	SGMatrix<float64_t> X(1, n);
	SGMatrix<float64_t> X_test(1, n);
	SGVector<float64_t> Y(n);

	X[0]=0;
	X[1]=1.1;
	X[2]=2.2;

	X_test[0]=0.3;
	X_test[1]=1.3;
	X_test[2]=2.5;

	for (index_t i=0; i<n; ++i)
	{
		Y[i]=CMath::sin(X(0, i));
	}

	/* shogun representation */
	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* specity GPR with exact inference */
	float64_t sigma=1;
	float64_t shogun_sigma=sigma*sigma*2;
	CGaussianKernel* kernel=new CGaussianKernel(10, shogun_sigma);
	CZeroMean* mean=new CZeroMean();
	CGaussianLikelihood* lik=new CGaussianLikelihood();
	lik->set_sigma(1);
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, feat_train,
			mean, label_train, lik);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();

	// get variance value for each input feature
	SGVector<float64_t> variance_vector=gpr->get_variance_vector(feat_test);

	/* do some checks against gpml toolbox*/
	// s2 =
	// 1.426104216614624
	// 1.416896787316447
	// 1.535464779087576
	EXPECT_LE(CMath::abs(variance_vector[0]-1.426104216614624), 10E-15);
	EXPECT_LE(CMath::abs(variance_vector[1]-1.416896787316447), 10E-15);
	EXPECT_LE(CMath::abs(variance_vector[2]-1.535464779087576), 10E-15);

	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression, get_variance_vector_2)
{
	// create some easy regression data: 1d noisy sine wave
	index_t ntr=5, ntst=15;

	SGMatrix<float64_t> feat_train(1, ntr);
	SGMatrix<float64_t> feat_test(1, ntst);
	SGVector<float64_t> lab_train(ntr);

	feat_train[0]=1.25107;
	feat_train[1]=2.16097;
	feat_train[2]=0.00034;
	feat_train[3]=0.90699;
	feat_train[4]=0.44026;

	lab_train[0]=0.39635;
	lab_train[1]=0.00358;
	lab_train[2]=-1.18139;
	lab_train[3]=1.35533;
	lab_train[4]=-0.08232;

	for (index_t i=0; i<ntst; i++)
		feat_test[i]=(float64_t)i/3.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 1 (by default)
	CGaussianLikelihood* liklihood=new CGaussianLikelihood();

	// specify GP regression with exact inference
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, features_train,
			mean, labels_train, liklihood);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();

	// get variance value for each input feature
	SGVector<float64_t> variance_vector=gpr->get_variance_vector(features_test);

	// comparison of variance with result from GPML package:
	// ys2 =
	// 1.3522
	// 1.2720
	// 1.2456
	// 1.2581
	// 1.2939
	// 1.3396
	// 1.3960
	// 1.4821
	// 1.6083
	// 1.7511
	// 1.8709
	// 1.9461
	// 1.9820
	// 1.9952
	// 1.9990
	EXPECT_NEAR(variance_vector[0], 1.3522, 1E-4);
	EXPECT_NEAR(variance_vector[1], 1.2720, 1E-4);
	EXPECT_NEAR(variance_vector[2], 1.2456, 1E-4);
	EXPECT_NEAR(variance_vector[3], 1.2581, 1E-4);
	EXPECT_NEAR(variance_vector[4], 1.2939, 1E-4);
	EXPECT_NEAR(variance_vector[5], 1.3396, 1E-4);
	EXPECT_NEAR(variance_vector[6], 1.3960, 1E-4);
	EXPECT_NEAR(variance_vector[7], 1.4821, 1E-4);
	EXPECT_NEAR(variance_vector[8], 1.6083, 1E-4);
	EXPECT_NEAR(variance_vector[9], 1.7511, 1E-4);
	EXPECT_NEAR(variance_vector[10], 1.8709, 1E-4);
	EXPECT_NEAR(variance_vector[11], 1.9461, 1E-4);
	EXPECT_NEAR(variance_vector[12], 1.9820, 1E-4);
	EXPECT_NEAR(variance_vector[13], 1.9952, 1E-4);
	EXPECT_NEAR(variance_vector[14], 1.9990, 1E-4);

	SG_UNREF(gpr);
}

TEST(GaussianProcessRegression,apply_regression_scaled_kernel)
{
	// create some easy regression data: 1d noisy sine wave
	index_t ntr=5;

	SGMatrix<float64_t> feat_train(1, ntr);
	SGVector<float64_t> lab_train(ntr);

	feat_train[0]=1.25107;
	feat_train[1]=2.16097;
	feat_train[2]=0.00034;
	feat_train[3]=0.90699;
	feat_train[4]=0.44026;

	lab_train[0]=0.39635;
	lab_train[1]=0.00358;
	lab_train[2]=-1.18139;
	lab_train[3]=1.35533;
	lab_train[4]=-0.08232;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with width = 2 * ell^2 = 0.02 and zero mean
	// function
	CGaussianKernel* kernel=new CGaussianKernel(10, 0.02);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.25
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.25);

	// specify GP regression with exact inference
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, features_train,
			mean, labels_train, lik);
	inf->set_scale(0.8);

	// create GPR and train
	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);
	gpr->train();

	// apply regression to train features
	CRegressionLabels* predictions=gpr->apply_regression();

	// comparison of predictions with result from GPML package
	SGVector<float64_t> mu=predictions->get_labels();

	EXPECT_NEAR(mu[0], 0.36138244503573730, 1E-15);
	EXPECT_NEAR(mu[1], 0.00326149466192171, 1E-15);
	EXPECT_NEAR(mu[2], -1.07628454651757055, 1E-15);
	EXPECT_NEAR(mu[3], 1.23483449459758354, 1E-15);
	EXPECT_NEAR(mu[4], -0.07500012155336166, 1E-15);

	SG_UNREF(predictions);
	SG_UNREF(gpr);
}

#endif
