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
#include <shogun/regression/gp/LaplacianInferenceMethod.h>
#include <shogun/regression/gp/ZeroMean.h>
#include <shogun/regression/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LaplacianInferenceMethod,get_cholesky)
{
	// create some easy regression data:
	// y approximately equals to 1/5*sin(10*x) + sqrt(x)
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 1 (by default)
	CGaussianLikelihood* liklihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel, features_train,
			mean, labels_train, liklihood);

	// comparison of posterior cholesky with result from GPML package:
	// L =
	// 1.41421   0.49295   0.43341   0.32346   0.01929
	// 0.00000   1.32552   0.58588   0.57578   0.13309
	// 0.00000   0.00000   1.21198   0.40341   0.12515
	// 0.00000   0.00000   0.00000   1.18369   0.18988
	// 0.00000   0.00000   0.00000   0.00000   1.38932
	SGMatrix<float64_t> L=inf->get_cholesky();

	EXPECT_NEAR(L(0,0), 1.41421, 1E-5);
	EXPECT_NEAR(L(0,1), 0.49295, 1E-5);
	EXPECT_NEAR(L(0,2), 0.43341, 1E-5);
	EXPECT_NEAR(L(0,3), 0.32346, 1E-5);
	EXPECT_NEAR(L(0,4), 0.01929, 1E-5);

	EXPECT_NEAR(L(1,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(1,1), 1.32552, 1E-5);
	EXPECT_NEAR(L(1,2), 0.58588, 1E-5);
	EXPECT_NEAR(L(1,3), 0.57578, 1E-5);
	EXPECT_NEAR(L(1,4), 0.13309, 1E-5);

	EXPECT_NEAR(L(2,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,2), 1.21198, 1E-5);
	EXPECT_NEAR(L(2,3), 0.40341, 1E-5);
	EXPECT_NEAR(L(2,4), 0.12515, 1E-5);

	EXPECT_NEAR(L(3,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,3), 1.18369, 1E-5);
	EXPECT_NEAR(L(3,4), 0.18988, 1E-5);

	EXPECT_NEAR(L(4,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,3), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,4), 1.38932, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha)
{
	// create some easy regression data:
	// y approximately equals to 1/5*sin(10*x) + sqrt(x)
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 1 (by default)
	CGaussianLikelihood* liklihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel, features_train,
			mean, labels_train, liklihood);

	// comparison of posterior alpha with result from GPML package:
	// alpha =
	// 0.112590
	// 0.030952
	// 0.265522
	// 0.372392
	// 0.660354
	SGVector<float64_t> alpha=inf->get_alpha();

	EXPECT_NEAR(alpha[0], 0.112590, 1E-6);
	EXPECT_NEAR(alpha[1], 0.030952, 1E-6);
	EXPECT_NEAR(alpha[2], 0.265522, 1E-6);
	EXPECT_NEAR(alpha[3], 0.372392, 1E-6);
	EXPECT_NEAR(alpha[4], 0.660354, 1E-6);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood)
{
	// create some easy regression data:
	// y approximately equals to 1/5*sin(10*x) + sqrt(x)
	index_t n=5;

	SGMatrix<float64_t> feat_train(1, n);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.27815;
	feat_train[1]=1.12759;
	feat_train[2]=1.26760;
	feat_train[3]=1.52883;
	feat_train[4]=2.96195;

	lab_train[0]=0.59787;
	lab_train[1]=0.86969;
	lab_train[2]=1.14778;
	lab_train[3]=1.31794;
	lab_train[4]=1.52609;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 1 (by default)
	CGaussianLikelihood* liklihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel, features_train,
			mean, labels_train, liklihood);

	// comparison of posterior negative marginal likelihood with
	// result from GPML package:
	// nlZ =
	// 6.8615
	float64_t nml=inf->get_negative_marginal_likelihood();

	EXPECT_NEAR(nml, 6.8615, 1E-4);

	// clean up
	SG_UNREF(inf);
}

#endif
