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
#include <shogun/regression/gp/FITCInferenceMethod.h>
#include <shogun/regression/gp/ZeroMean.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(FITCInferenceMethod,get_cholesky)
{
	// create some easy regression data with latent features:
	// y approximately equals to x^sin(x)
	index_t n=6;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> lat_feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.81263;
	feat_train[1]=0.99976;
	feat_train[2]=1.17037;
	feat_train[3]=1.51752;
	feat_train[4]=1.57765;
	feat_train[5]=3.89440;

	lat_feat_train[0]=0.00000;
	lat_feat_train[1]=0.80000;
	lat_feat_train[2]=1.60000;
	lat_feat_train[3]=2.40000;
	lat_feat_train[4]=3.20000;
	lat_feat_train[5]=4.00000;

	lab_train[0]=0.86015;
	lab_train[1]=0.99979;
	lab_train[2]=1.15589;
	lab_train[3]=1.51662;
	lab_train[4]=1.57764;
	lab_train[5]=0.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(0.1);

	// specify GP regression with Laplacian inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, latent_features_train);

	// comparison of posterior cholesky with result from GPML package:
	// L =
	// -0.326180   0.148601   0.405579  -0.683624   0.319057  -0.073608
    //  0.148601  -2.222957   1.480121   0.170280  -0.102392  -0.016981
    //  0.405579   1.480121  -2.887356   1.091245  -0.481484   0.129348
    // -0.683624   0.170280   1.091245  -1.628117   0.779654  -0.016188
    //  0.319057  -0.102392  -0.481484   0.779654  -0.410200  -0.152221
    // -0.073608  -0.016981   0.129348  -0.016188  -0.152221  -0.722832
	SGMatrix<float64_t> L=inf->get_cholesky();

	EXPECT_NEAR(L(0,0), -0.326180, 1E-5);
	EXPECT_NEAR(L(0,1), 0.148601, 1E-5);
	EXPECT_NEAR(L(0,2), 0.405579, 1E-5);
	EXPECT_NEAR(L(0,3), -0.683624, 1E-5);
	EXPECT_NEAR(L(0,4), 0.319057, 1E-5);
	EXPECT_NEAR(L(0,5), -0.073608, 1E-5);

	EXPECT_NEAR(L(1,0), 0.148601, 1E-5);
	EXPECT_NEAR(L(1,1), -2.222957, 1E-5);
	EXPECT_NEAR(L(1,2), 1.480121, 1E-5);
	EXPECT_NEAR(L(1,3), 0.170280, 1E-5);
	EXPECT_NEAR(L(1,4), -0.102392, 1E-5);
	EXPECT_NEAR(L(1,5), -0.016981, 1E-5);

	EXPECT_NEAR(L(2,0), 0.405579, 1E-5);
	EXPECT_NEAR(L(2,1), 1.480121, 1E-5);
	EXPECT_NEAR(L(2,2), -2.887356, 1E-5);
	EXPECT_NEAR(L(2,3), 1.091245, 1E-5);
	EXPECT_NEAR(L(2,4), -0.481484, 1E-5);
	EXPECT_NEAR(L(2,5), 0.129348, 1E-5);

	EXPECT_NEAR(L(3,0), -0.683624, 1E-5);
	EXPECT_NEAR(L(3,1), 0.170280, 1E-5);
	EXPECT_NEAR(L(3,2), 1.091245, 1E-5);
	EXPECT_NEAR(L(3,3), -1.628117, 1E-5);
	EXPECT_NEAR(L(3,4), 0.779654, 1E-5);
	EXPECT_NEAR(L(3,5), -0.016188, 1E-5);

	EXPECT_NEAR(L(4,0), 0.319057, 1E-5);
	EXPECT_NEAR(L(4,1), -0.102392, 1E-5);
	EXPECT_NEAR(L(4,2), -0.481484, 1E-5);
	EXPECT_NEAR(L(4,3), 0.779654, 1E-5);
	EXPECT_NEAR(L(4,4), -0.410200, 1E-5);
	EXPECT_NEAR(L(4,5), -0.152221, 1E-5);

	EXPECT_NEAR(L(5,0), -0.073608, 1E-5);
	EXPECT_NEAR(L(5,1), -0.016981, 1E-5);
	EXPECT_NEAR(L(5,2), 0.129348, 1E-5);
	EXPECT_NEAR(L(5,3), -0.016188, 1E-5);
	EXPECT_NEAR(L(5,4), -0.152221, 1E-5);
	EXPECT_NEAR(L(5,5), -0.722832, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_alpha)
{
	// create some easy regression data with latent features:
	// y approximately equals to x^sin(x)
	index_t n=6;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> lat_feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.81263;
	feat_train[1]=0.99976;
	feat_train[2]=1.17037;
	feat_train[3]=1.51752;
	feat_train[4]=1.57765;
	feat_train[5]=3.89440;

	lat_feat_train[0]=0.00000;
	lat_feat_train[1]=0.80000;
	lat_feat_train[2]=1.60000;
	lat_feat_train[3]=2.40000;
	lat_feat_train[4]=3.20000;
	lat_feat_train[5]=4.00000;

	lab_train[0]=0.86015;
	lab_train[1]=0.99979;
	lab_train[2]=1.15589;
	lab_train[3]=1.51662;
	lab_train[4]=1.57764;
	lab_train[5]=0.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(0.1);

	// specify GP regression with Laplacian inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, latent_features_train);

	// comparison of posterior alpha with result from GPML package:
	// alpha =
	//  0.40342
	// -0.77257
	//  1.51352
	//  0.78448
	// -0.32513
	//  0.29023
	SGVector<float64_t> alpha=inf->get_alpha();

	EXPECT_NEAR(alpha[0], 0.40342, 1E-5);
	EXPECT_NEAR(alpha[1], -0.77257, 1E-5);
	EXPECT_NEAR(alpha[2], 1.51352, 1E-5);
	EXPECT_NEAR(alpha[3], 0.78448, 1E-5);
	EXPECT_NEAR(alpha[4], -0.32513, 1E-5);
	EXPECT_NEAR(alpha[5], 0.29023, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_negative_marginal_likelihood)
{
	// create some easy regression data with latent features:
	// y approximately equals to x^sin(x)
	index_t n=6;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> lat_feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.81263;
	feat_train[1]=0.99976;
	feat_train[2]=1.17037;
	feat_train[3]=1.51752;
	feat_train[4]=1.57765;
	feat_train[5]=3.89440;

	lat_feat_train[0]=0.00000;
	lat_feat_train[1]=0.80000;
	lat_feat_train[2]=1.60000;
	lat_feat_train[3]=2.40000;
	lat_feat_train[4]=3.20000;
	lat_feat_train[5]=4.00000;

	lab_train[0]=0.86015;
	lab_train[1]=0.99979;
	lab_train[2]=1.15589;
	lab_train[3]=1.51662;
	lab_train[4]=1.57764;
	lab_train[5]=0.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(0.1);

	// specify GP regression with Laplacian inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, latent_features_train);

	// comparison of posterior negative marginal likelihood with
	// result from GPML package:
	// nlZ =
	// 0.84354
	float64_t nml=inf->get_negative_marginal_likelihood();

	EXPECT_NEAR(nml, 0.84354, 1E-5);

	// clean up
	SG_UNREF(inf);
}

#endif
