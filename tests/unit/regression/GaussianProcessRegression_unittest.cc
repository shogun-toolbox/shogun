/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * Written (W) 2013 Roman Votyakov
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */
#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/GaussianARDSparseKernel.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/machine/gp/SparseVGInferenceMethod.h>

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

TEST(GaussianProcessRegression,sparse_vg_regression)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t abs_tolerance,rel_tolerance=1e-6;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=1.51752;
	feat_train(0,4)=1.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=0.5;
	feat_train(1,1)=0.4576;
	feat_train(1,2)=5.17637;
	feat_train(1,3)=2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=3.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=0.46;
	lab_train[1]=0.7;
	lab_train[2]=-1.16;
	lab_train[3]=1.5;
	lab_train[4]=3.5;
	lab_train[5]=-5.0;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	float64_t ell=log(2.0);
	CKernel* kernel=new CGaussianKernel(10,2.0*exp(ell*2.0));


	float64_t mean_weight=0.0;
	CConstMean* mean=new CConstMean(mean_weight);

	float64_t sigma=0.5;
	CGaussianLikelihood* lik=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CSparseVGInferenceMethod* inf=new CSparseVGInferenceMethod(kernel, features_train,
		mean, labels_train, lik, inducing_features_train);

	float64_t ind_noise=1e-6;
	inf->set_inducing_noise(ind_noise);

	float64_t scale=1.5;
	inf->set_scale(scale);

	inf->enable_optimizing_inducing_features(false);

	int32_t k=4;
	SGMatrix<float64_t> feat_test(dim, k);
	feat_test(0,0)=0.81263;
	feat_test(0,1)=-0.9976;
	feat_test(0,2)=0.17037;
	feat_test(0,3)=0.5172;

	feat_test(1,0)=0.5;
	feat_test(1,1)=0.456;
	feat_test(1,2)=3.1767;
	feat_test(1,3)=1.5652;

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();
	SG_REF(features_test);
	SGVector<float64_t> mean_vector=gpr->get_mean_vector(features_test);
	SGVector<float64_t> var_vector=gpr->get_variance_vector(features_test);
	SG_UNREF(features_test);
	// comparison of mean and variance with result from varsgpv package:
	// http://www.aueb.gr/users/mtitsias/code/varsgp.tar.gz
	//mustar =
	//-0.246280335053918
	//0.781735233521474
	//2.841201927903070
	//1.072110958530009
	//varstar =
	//1.970193540002217
	//2.341819379690606
	//0.670229188503544
	//1.293072726669006
	
	abs_tolerance = CMath::get_abs_tolerance(-0.246280335053918, rel_tolerance);
	EXPECT_NEAR(mean_vector[0],  -0.246280335053918,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.781735233521474, rel_tolerance);
	EXPECT_NEAR(mean_vector[1],  0.781735233521474,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(2.841201927903070, rel_tolerance);
	EXPECT_NEAR(mean_vector[2],  2.841201927903070,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.072110958530009, rel_tolerance);
	EXPECT_NEAR(mean_vector[3],  1.072110958530009,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(1.970193540002217, rel_tolerance);
	EXPECT_NEAR(var_vector[0],  1.970193540002217,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(2.341819379690606, rel_tolerance);
	EXPECT_NEAR(var_vector[1],  2.341819379690606,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.670229188503544, rel_tolerance);
	EXPECT_NEAR(var_vector[2],  0.670229188503544,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(1.293072726669006, rel_tolerance);
	EXPECT_NEAR(var_vector[3],  1.293072726669006,  abs_tolerance);
	
	// clean up
	SG_UNREF(gpr);
}

#ifdef HAVE_LINALG_LIB
TEST(GaussianProcessRegression,fitc_regression)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=-0.81263;
	feat_train(0,1)=-0.99976;
	feat_train(0,2)=1.17037;
	feat_train(0,3)=-1.51752;
	feat_train(0,4)=8.57765;
	feat_train(0,5)=3.89440;

	feat_train(1,0)=-0.5;
	feat_train(1,1)=5.4576;
	feat_train(1,2)=7.17637;
	feat_train(1,3)=-2.56752;
	feat_train(1,4)=4.57765;
	feat_train(1,5)=2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lab_train[0]=0.46015;
	lab_train[1]=0.69979;
	lab_train[2]=2.15589;
	lab_train[3]=1.51672;
	lab_train[4]=3.59764;
	lab_train[5]=2.39475;

	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	CGaussianARDFITCKernel* kernel=new CGaussianARDFITCKernel(10);
	int32_t t_dim=2;
	SGMatrix<float64_t> weights(dim,t_dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(1,0)=weight2;
	weights(0,1)=weight3;
	weights(1,1)=weight4;
	kernel->set_matrix_weights(weights);

	float64_t mean_weight=2.0;
	CConstMean* mean=new CConstMean(mean_weight);

	// Gaussian likelihood with sigma = 0.5
	float64_t sigma=0.5;
	CGaussianLikelihood* lik=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

	float64_t scale=4.0;
	inf->set_scale(scale);

	int32_t k=4;
	SGMatrix<float64_t> feat_test(dim, k);
	feat_test(0,0)=-0.81263;
	feat_test(0,1)=5.4576;
	feat_test(0,2)=-0.239;
	feat_test(0,3)=2.45;

	feat_test(1,0)=-0.5;
	feat_test(1,1)=0.69979;
	feat_test(1,2)=2.3546;
	feat_test(1,3)=-0.46;

	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(feat_test);

	CGaussianProcessRegression* gpr=new CGaussianProcessRegression(inf);

	// train model
	gpr->train();

	SG_REF(features_test);
	SGVector<float64_t> mean_vector=gpr->get_mean_vector(features_test);
	SGVector<float64_t> var_vector=gpr->get_variance_vector(features_test);
	SG_UNREF(features_test);

	// comparison of mean and variance with result from GPML 3.5 package:
	//ymu =
	//0.817143553262107
	//1.001048686764744
	//2.182234371254691
	//0.814785544659520
	//ys2 =
	//3.937450814687706
	//1.878118517080519
	//0.697568637099934
	//4.354657330167651

	EXPECT_NEAR(mean_vector[0], 0.817143553262107, 1E-10);
	EXPECT_NEAR(mean_vector[1], 1.001048686764744, 1E-10);
	EXPECT_NEAR(mean_vector[2], 2.182234371254691, 1E-10);
	EXPECT_NEAR(mean_vector[3], 0.814785544659520, 1E-10);

	EXPECT_NEAR(var_vector[0], 3.937450814687706, 1E-10);
	EXPECT_NEAR(var_vector[1], 1.878118517080519, 1E-10);
	EXPECT_NEAR(var_vector[2], 0.697568637099934, 1E-10);
	EXPECT_NEAR(var_vector[3], 4.354657330167651, 1E-10);

	// clean up
	SG_UNREF(gpr);
}
#endif /* HAVE_LINALG_LIB */

#endif /* HAVE_EIGEN3 */
