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
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/LaplacianInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/StudentsTLikelihood.h>
#include <shogun/machine/gp/LogitLikelihood.h>
#include <shogun/machine/gp/ProbitLikelihood.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LaplacianInferenceMethod,get_cholesky_gaussian_likelihood)
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
	CGaussianLikelihood* likelihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

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

TEST(LaplacianInferenceMethod,get_cholesky_t_likelihood)
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

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(1, 3);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior cholesky with result from GPML package:
	// L =
	// 1.52370   0.60734   0.52104   0.37860   0.01995
	// 0.00000   1.40156   0.69336   0.66974   0.13981
	// 0.00000   0.00000   1.23173   0.43255   0.12387
	// 0.00000   0.00000   0.00000   1.19342   0.18933
	// 0.00000   0.00000   0.00000   0.00000   1.36684
	SGMatrix<float64_t> L=inf->get_cholesky();

	EXPECT_NEAR(L(0,0), 1.52370, 1E-5);
	EXPECT_NEAR(L(0,1), 0.60734, 1E-5);
	EXPECT_NEAR(L(0,2), 0.52104, 1E-5);
	EXPECT_NEAR(L(0,3), 0.37860, 1E-5);
	EXPECT_NEAR(L(0,4), 0.01995, 1E-5);

	EXPECT_NEAR(L(1,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(1,1), 1.40156, 1E-5);
	EXPECT_NEAR(L(1,2), 0.69336, 1E-5);
	EXPECT_NEAR(L(1,3), 0.66974, 1E-5);
	EXPECT_NEAR(L(1,4), 0.13981, 1E-5);

	EXPECT_NEAR(L(2,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,2), 1.23173, 1E-5);
	EXPECT_NEAR(L(2,3), 0.43255, 1E-5);
	EXPECT_NEAR(L(2,4), 0.12387, 1E-5);

	EXPECT_NEAR(L(3,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,3), 1.19342, 1E-5);
	EXPECT_NEAR(L(3,4), 0.18933, 1E-5);

	EXPECT_NEAR(L(4,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,3), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,4), 1.36684, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_cholesky_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior cholesky with result from GPML package:
	// L =
	// 1.11695   0.03594   0.04463   0.09123   0.07623
	// 0.00000   1.10397   0.03866   0.15833   0.14793
	// 0.00000   0.00000   1.11470   0.09049   0.01420
	// 0.00000   0.00000   0.00000   1.09297   0.11357
	// 0.00000   0.00000   0.00000   0.00000   1.08875
	SGMatrix<float64_t> L=inf->get_cholesky();

	EXPECT_NEAR(L(0,0), 1.11695, 1E-5);
	EXPECT_NEAR(L(0,1), 0.03594, 1E-5);
	EXPECT_NEAR(L(0,2), 0.04463, 1E-5);
	EXPECT_NEAR(L(0,3), 0.09123, 1E-5);
	EXPECT_NEAR(L(0,4), 0.07623, 1E-5);

	EXPECT_NEAR(L(1,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(1,1), 1.10397, 1E-5);
	EXPECT_NEAR(L(1,2), 0.03866, 1E-5);
	EXPECT_NEAR(L(1,3), 0.15833, 1E-5);
	EXPECT_NEAR(L(1,4), 0.14793, 1E-5);

	EXPECT_NEAR(L(2,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,2), 1.11470, 1E-5);
	EXPECT_NEAR(L(2,3), 0.09049, 1E-5);
	EXPECT_NEAR(L(2,4), 0.01420, 1E-5);

	EXPECT_NEAR(L(3,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,3), 1.09297, 1E-5);
	EXPECT_NEAR(L(3,4), 0.11357, 1E-5);

	EXPECT_NEAR(L(4,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,3), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,4), 1.08875, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_cholesky_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior cholesky with result from GPML package:
	// L =
	// 1.22980   0.00000   0.00000   0.00100   0.00002
	// 0.00000   1.22911   0.00000   0.00680   0.00000
	// 0.00000   0.00000   1.22970   0.00005  -0.00000
	// 0.00000   0.00000   0.00000   1.22917   0.00000
	// 0.00000   0.00000   0.00000   0.00000   1.22971
	SGMatrix<float64_t> L=inf->get_cholesky();

	EXPECT_NEAR(L(0,0), 1.22980, 1E-5);
	EXPECT_NEAR(L(0,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(0,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(0,3), 0.00100, 1E-5);
	EXPECT_NEAR(L(0,4), 0.00002, 1E-5);

	EXPECT_NEAR(L(1,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(1,1), 1.22911, 1E-5);
	EXPECT_NEAR(L(1,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(1,3), 0.00680, 1E-5);
	EXPECT_NEAR(L(1,4), 0.00000, 1E-5);

	EXPECT_NEAR(L(2,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(2,2), 1.22970, 1E-5);
	EXPECT_NEAR(L(2,3), 0.00005, 1E-5);
	EXPECT_NEAR(L(2,4), 0.00000, 1E-5);

	EXPECT_NEAR(L(3,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(3,3), 1.22917, 1E-5);
	EXPECT_NEAR(L(3,4), 0.00000, 1E-5);

	EXPECT_NEAR(L(4,0), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,1), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,2), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,3), 0.00000, 1E-5);
	EXPECT_NEAR(L(4,4), 1.22971, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha_gaussian_likelihood)
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
	CGaussianLikelihood* likelihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

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

TEST(LaplacianInferenceMethod,get_alpha_t_likelihood)
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

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(1, 3);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior alpha with result from GPML package:
	// alpha =
	// 0.124677
	// -0.011322
	// 0.291186
	// 0.414107
	// 0.710853
	SGVector<float64_t> alpha=inf->get_alpha();

	EXPECT_NEAR(alpha[0], 0.124677, 1E-6);
	EXPECT_NEAR(alpha[1], -0.011322, 1E-6);
	EXPECT_NEAR(alpha[2], 0.291186, 1E-6);
	EXPECT_NEAR(alpha[3], 0.414107, 1E-6);
	EXPECT_NEAR(alpha[4], 0.710853, 1E-6);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior alpha with result from GPML package:
	// alpha =
	// 0.45082
	// -0.32691
	// 0.43705
	// -0.38239
	// -0.34563
	SGVector<float64_t> alpha=inf->get_alpha();

	EXPECT_NEAR(alpha[0], 0.45082, 1E-5);
	EXPECT_NEAR(alpha[1], -0.32691, 1E-5);
	EXPECT_NEAR(alpha[2], 0.43705, 1E-5);
	EXPECT_NEAR(alpha[3], -0.38239, 1E-5);
	EXPECT_NEAR(alpha[4], -0.34563, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior alpha with result from GPML package:
	// alpha =
	// -0.50646
	// 0.50327
	// 0.50604
	// 0.50366
	// -0.50605
	SGVector<float64_t> alpha=inf->get_alpha();

	EXPECT_NEAR(alpha[0], -0.50646, 1E-5);
	EXPECT_NEAR(alpha[1], 0.50327, 1E-5);
	EXPECT_NEAR(alpha[2], 0.50604, 1E-5);
	EXPECT_NEAR(alpha[3], 0.50366, 1E-5);
	EXPECT_NEAR(alpha[4], -0.50605, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_gaussian_likelihood)
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
	CGaussianLikelihood* likelihood=new CGaussianLikelihood();

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior negative marginal likelihood with
	// result from GPML package:
	// nlZ =
	// 6.8615
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nml, 6.8615, 1E-4);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_t_likelihood)
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

	// Student's-T likelihood with sigma = 1, df = 3
	CStudentsTLikelihood* likelihood=new CStudentsTLikelihood(1, 3);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior negative marginal likelihood with
	// result from GPML package:
	// nlZ =
	// 7.4892
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nml, 7.4892, 1E-4);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior negative marginal likelihood with
	// result from GPML package:
	// nlZ =
	// 3.3876
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nml, 3.3876, 1E-4);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, likelihood);

	// comparison of posterior negative marginal likelihood with
	// result from GPML package:
	// nlZ =
	// 3.4990
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nml, 3.4990, 1E-4);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_gaussian_likelihood)
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

	float64_t ell=0.1;

	// choose Gaussian kernel with width = 2 * ell^2 = 0.02 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.25
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.25);

	// specify GP regression with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, lik);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");

	float64_t dnlZ_ell=4*ell*ell*(gradient->get_element(width_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison of partial derivatives of negative marginal likelihood with
	// result from GPML package:
	// lik =  0.0074073
	// cov =
	// -0.85103
	// -0.57052
	EXPECT_NEAR(dnlZ_lik, 0.0074073, 1E-7);
	EXPECT_NEAR(dnlZ_ell, -0.85103, 1E-5);
	EXPECT_NEAR(dnlZ_sf2, -0.57052, 1E-5);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_t_likelihood)
{
	// create some easy regression data: 1d noisy sine wave
	index_t ntr=5;

	SGMatrix<float64_t> feat_train(1, ntr);
	SGVector<float64_t> lab_train(ntr);

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

	float64_t ell=0.1;

	// choose Gaussian kernel with width = 2 * ell^2 = 0.02 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);
	CZeroMean* mean=new CZeroMean();

	// Student's-T likelihood with sigma = 0.25, df = 3
	CStudentsTLikelihood* lik=new CStudentsTLikelihood(0.25, 3);

	// specify GP regression with exact inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
		features_train,	mean, labels_train, lik);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");
	TParameter* df_param=lik->m_gradient_parameters->get_parameter("df");

	float64_t dnlZ_ell=4*ell*ell*(gradient->get_element(width_param))[0];
	float64_t dnlZ_df=(gradient->get_element(df_param))[0];
	float64_t dnlZ_sigma=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison of partial derivatives of negative marginal likelihood with
	// result from GPML package:
	// lik =
	// -0.64932
	// -0.15567
	// cov =
	// -0.84364
	// -0.30177
	EXPECT_NEAR(dnlZ_df, -0.64932, 1E-5);
	EXPECT_NEAR(dnlZ_sigma, -0.15567, 1E-5);
	EXPECT_NEAR(dnlZ_ell, -0.84364, 1E-5);
	EXPECT_NEAR(dnlZ_sf2, -0.30177, 1E-5);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_logit_likelihood)
{
	// create some easy classification data:
	// y=sign(sqrt(x1.^2+x2.^2)-1)
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=0.8822936;
	feat_train(0, 1)=-0.7160792;
	feat_train(0, 2)=0.9178174;
	feat_train(0, 3)=-0.0135544;
	feat_train(0, 4)=-0.5275911;

	feat_train(1, 0)=-0.9597321;
	feat_train(1, 1)=0.0231289;
	feat_train(1, 2)=0.8284935;
	feat_train(1, 3)=0.0023812;
	feat_train(1, 4)=-0.7218931;

	lab_train[0]=1.0;
	lab_train[1]=-1.0;
	lab_train[2]=1.0;
	lab_train[3]=-1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// logit likelihood
	CLogitLikelihood* likelihood=new CLogitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");

	float64_t dnlZ_ell=4*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison of partial derivatives of negative marginal likelihood with
	// result from GPML package:
	// cov =
	// 0.266464
	// -0.068637
	EXPECT_NEAR(dnlZ_ell, 0.266464, 1E-6);
	EXPECT_NEAR(dnlZ_sf2, -0.068637, 1E-6);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");

	float64_t dnlZ_ell=4*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];

	// comparison of partial derivatives of negative marginal likelihood with
	// result from GPML package:
	// cov =
	// -0.034178
	// 0.108246
	EXPECT_NEAR(dnlZ_ell, -0.034178, 1E-6);
	EXPECT_NEAR(dnlZ_sf2, 0.108246, 1E-6);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_posterior_mean_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	// comparison of the mode with result from GPML package
	SGVector<float64_t> approx_mean=inf->get_posterior_mean();
	EXPECT_NEAR(approx_mean[0], -0.50527, 1E-5);
	EXPECT_NEAR(approx_mean[1], 0.51150, 1E-5);
	EXPECT_NEAR(approx_mean[2], 0.50609, 1E-5);
	EXPECT_NEAR(approx_mean[3], 0.51073, 1E-5);
	EXPECT_NEAR(approx_mean[4], -0.50607, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_posterior_covariance_probit_likelihood)
{
	// create some easy random classification data
	index_t n=5;

	SGMatrix<float64_t> feat_train(2, n);
	SGVector<float64_t> lab_train(n);

	feat_train(0, 0)=-1.07932;
	feat_train(0, 1)=1.15768;
	feat_train(0, 2)=3.26631;
	feat_train(0, 3)=1.79009;
	feat_train(0, 4)=-3.66051;

	feat_train(1, 0)=-1.83544;
	feat_train(1, 1)=2.91702;
	feat_train(1, 2)=-3.85663;
	feat_train(1, 3)=0.11949;
	feat_train(1, 4)=1.75159;

	lab_train[0]=-1.0;
	lab_train[1]=1.0;
	lab_train[2]=1.0;
	lab_train[3]=1.0;
	lab_train[4]=-1.0;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CBinaryLabels* labels_train=new CBinaryLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// probit likelihood
	CProbitLikelihood* likelihood=new CProbitLikelihood();

	// specify GP classification with Laplacian inference
	CLaplacianInferenceMethod* inf=new CLaplacianInferenceMethod(kernel,
			features_train,	mean, labels_train, likelihood);

	SGMatrix<float64_t> approx_cov=inf->get_posterior_covariance();

	// comparison of the covariance with result from GPML package
	EXPECT_NEAR(approx_cov(0,0), 6.6120e-01, 1E-5);
	EXPECT_NEAR(approx_cov(0,1), -5.3908e-06, 1E-5);
	EXPECT_NEAR(approx_cov(0,2), 4.4528e-06, 1E-5);
	EXPECT_NEAR(approx_cov(0,3), 1.0552e-03, 1E-5);
	EXPECT_NEAR(approx_cov(0,4), 2.5118e-05, 1E-5);

	EXPECT_NEAR(approx_cov(1,0), -5.3908e-06, 1E-5);
	EXPECT_NEAR(approx_cov(1,1), 6.6190e-01, 1E-5);
	EXPECT_NEAR(approx_cov(1,2), -3.0048e-07, 1E-5);
	EXPECT_NEAR(approx_cov(1,3), 7.1667e-03, 1E-5);
	EXPECT_NEAR(approx_cov(1,4), 2.0193e-06, 1E-5);

	EXPECT_NEAR(approx_cov(2,0), 4.4528e-06, 1E-5);
	EXPECT_NEAR(approx_cov(2,1), -3.0048e-07, 1E-5);
	EXPECT_NEAR(approx_cov(2,2), 6.6130e-01, 1E-5);
	EXPECT_NEAR(approx_cov(2,3), 5.4317e-05, 1E-5);
	EXPECT_NEAR(approx_cov(2,4), -8.7921e-11, 1E-5);

	EXPECT_NEAR(approx_cov(3,0), 1.0552e-03, 1E-5);
	EXPECT_NEAR(approx_cov(3,1), 7.1667e-03, 1E-5);
	EXPECT_NEAR(approx_cov(3,2), 5.4317e-05, 1E-5);
	EXPECT_NEAR(approx_cov(3,3), 6.6181e-01, 1E-5);
	EXPECT_NEAR(approx_cov(3,4), 9.1741e-09, 1E-5);

	EXPECT_NEAR(approx_cov(4,0), 2.5118e-05, 1E-5);
	EXPECT_NEAR(approx_cov(4,1), 2.0193e-06, 1E-5);
	EXPECT_NEAR(approx_cov(4,2), -8.7921e-11, 1E-5);
	EXPECT_NEAR(approx_cov(4,3), 9.1741e-09, 1E-5);
	EXPECT_NEAR(approx_cov(4,4), 6.6130e-01, 1E-5);

	// clean up
	SG_UNREF(inf);
}

#endif /* HAVE_EIGEN3 */
