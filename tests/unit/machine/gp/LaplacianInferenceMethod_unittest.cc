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
#include <shogun/mathematics/Math.h>

using namespace shogun;
float64_t get_abs_tolorance(float64_t true_value, float64_t rel_tolorance){
  rel_tolorance = CMath::abs(rel_tolorance);
  return true_value==0.0 ? rel_tolorance : CMath::abs(true_value * rel_tolorance);
}
/* corresponding Matlab/Ocatve code in GPML for 
 * get_negative_marginal_likelihood_logit_likelihood
 * get_marginal_likelihood_derivatives_logit_likelihood
 * get_cholesky_logit_likelihood
 * get_alpha_logit_likelihood
 */
//clear all; close all;
//meanfunc = @meanZero; hyp.mean=[];
//%where hyp.cov=log([sqrt(width/2.0) 1.0]) CGaussianKernel(cache_size, width)  
//covfunc = @covSEiso; hyp.cov = log([1.0 1.0]);
//likfunc = @likLogistic; hyp.lik=[];
//x1=[0.8822936
//-0.7160792
//0.9178174
//-0.0135544
//-0.5275911];
//x2=[-0.9597321
//0.0231289
//0.8284935
//0.0023812
//-0.7218931];
//x=[x1 x2];
//y=[1
//-1
//1
//-1
//-1];
//inf = @infLaplace;
//covfunc = {covfunc};
//meanfunc = {meanfunc};
//likfunc = {likfunc};
//[post nlZ dnlZ] = infLaplace(hyp, meanfunc, covfunc, likfunc, x, y);
//post.alpha %get_alpha_logit_likelihood
//post.L %get_cholesky_logit_likelihood
//nlZ %get_negative_marginal_likelihood_logit_likelihood
//dnlZ %get_marginal_likelihood_derivatives_logit_likelihood

/* corresponding Matlab/Ocatve code in GPML for 
 * get_negative_marginal_likelihood_gaussian_likelihood
 * get_cholesky_gaussian_likelihood
 * get_alpha_gaussian_likelihood
 * get_marginal_likelihood_derivatives_gaussian_likelihood
 */
//clear all; close all;
//meanfunc = @meanZero; hyp.mean=[];
//%for get_marginal_likelihood_derivatives_gaussian_likelihood
//covfunc = @covSEiso; hyp.cov = log([0.1 1.0]); 
//%for others
//covfunc = @covSEiso; hyp.cov = log([1.0 1.0]);
//%hyp.lik=log(sigma) where CGaussianLikelihood(sigma);
//likfunc = @likGauss; hyp.lik=log(1.0);
//x=[
//0.27815
//1.12759
//1.26760
//1.52883
//2.96195];
//y=[
//0.59787
//0.86969
//1.14778
//1.31794
//1.52609];
//inf = @infLaplace;
//covfunc = {covfunc};
//meanfunc = {meanfunc};
//likfunc = {likfunc};
//[post nlZ dnlZ] = infLaplace(hyp, meanfunc, covfunc, likfunc, x, y);
//post.alpha %get_alpha_gaussian_likelihood
//post.L %get_cholesky_gaussian_likelihood
//nlZ %get_negative_marginal_likelihood_gaussian_likelihood
//dnlZ %get_marginal_likelihood_derivatives_gaussian_likelihood

/* corresponding Matlab/Ocatve code in GPML for 
 * get_negative_marginal_likelihood_probit_likelihood
 * get_marginal_likelihood_derivatives_probit_likelihood
 * get_cholesky_probit_likelihood
 * get_alpha_probit_likelihood
 * get_posterior_mean_probit_likelihood
 * get_posterior_covariance_probit_likelihood
 */
//clear all; close all;
//meanfunc = @meanZero; hyp.mean=[];
//covfunc = @covSEiso; hyp.cov = log([1.0 1.0]);
//likfunc = @likErf; hyp.lik=[];
//x1=[-1.07932
//1.15768
//3.26631
//1.79009
//-3.66051];
//x2=[-1.83544
//2.91702
//-3.85663
//0.11949
//1.75159];
//x=[x1 x2];
//y=[-1
//1
//1
//1
//-1];
//inf = @infLaplace;
//covfunc = {covfunc};
//meanfunc = {meanfunc};
//likfunc = {likfunc};
//[post nlZ dnlZ] = infLaplace(hyp, meanfunc, covfunc, likfunc, x, y);
//m = feval(meanfunc{:}, hyp.mean, x);
//K = feval(covfunc{:},  hyp.cov,  x);
//V = (post.L)'\(diag(post.sW)*K);
//post_mean = K*(post.alpha)+m
//post_cov = K - V'*V
//post.alpha %get_alpha_probit_likelihood
//post.L %get_cholesky_probit_likelihood
//nlZ %get_negative_marginal_likelihood_probit_likelihood
//dnlZ %get_marginal_likelihood_derivatives_probit_likelihood
//post_mean %get_posterior_mean_probit_likelihood
//post_cov %get_posterior_covariance_probit_likelihood

/* corresponding Matlab/Ocatve code in GPML for 
 * get_negative_marginal_likelihood_t_likelihood
 * get_cholesky_t_likelihood
 * get_alpha_t_likelihood
 * get_marginal_likelihood_derivatives_t_likelihood
 */
//clear all; close all;
//meanfunc = @meanZero; hyp.mean=[];
//covfunc = @covSEiso; hyp.cov = log([0.1 1.0]);
//%for get_marginal_likelihood_derivatives_t_likelihood
//covfunc = @covSEiso; hyp.cov = log([0.1 1.0]); 
//%for others
//covfunc = @covSEiso; hyp.cov = log([1.0 1.0]);
// %where hyp.lik=log([(df-1), sigma])  CStudentsTLikelihood(sigma, df);
//likfunc = @likT; hyp.lik=log([3-1,0.25]);
//x=[0.27815
//1.12759
//1.26760
//1.52883
//2.96195];
//y=[0.59787
//0.86969
//1.14778
//1.31794
//1.52609];
//inf = @infLaplace;
//covfunc = {covfunc};
//meanfunc = {meanfunc};
//likfunc = {likfunc};
//[post nlZ dnlZ] = infLaplace(hyp, meanfunc, covfunc, likfunc, x, y);
//post.alpha %get_alpha_t_likelihood
//post.L %get_cholesky_t_likelihood
//nlZ %get_negative_marginal_likelihood_t_likelihood
//dnlZ %get_marginal_likelihood_derivatives_t_likelihood


TEST(LaplacianInferenceMethod,get_cholesky_gaussian_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(1.41421, rel_tolorance);
  EXPECT_NEAR(L(0,0),  1.41421,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.49295, rel_tolorance);
  EXPECT_NEAR(L(0,1),  0.49295,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.43341, rel_tolorance);
  EXPECT_NEAR(L(0,2),  0.43341,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.32346, rel_tolorance);
  EXPECT_NEAR(L(0,3),  0.32346,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.01929, rel_tolorance);
  EXPECT_NEAR(L(0,4),  0.01929,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(1,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.32552, rel_tolorance);
  EXPECT_NEAR(L(1,1),  1.32552,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.58588, rel_tolorance);
  EXPECT_NEAR(L(1,2),  0.58588,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.57578, rel_tolorance);
  EXPECT_NEAR(L(1,3),  0.57578,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.13309, rel_tolorance);
  EXPECT_NEAR(L(1,4),  0.13309,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.21198, rel_tolorance);
  EXPECT_NEAR(L(2,2),  1.21198,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.40341, rel_tolorance);
  EXPECT_NEAR(L(2,3),  0.40341,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.12515, rel_tolorance);
  EXPECT_NEAR(L(2,4),  0.12515,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.18369, rel_tolorance);
  EXPECT_NEAR(L(3,3),  1.18369,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.18988, rel_tolorance);
  EXPECT_NEAR(L(3,4),  0.18988,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,3),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.38932, rel_tolorance);
  EXPECT_NEAR(L(4,4),  1.38932,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_cholesky_t_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(1.52370, rel_tolorance);
  EXPECT_NEAR(L(0,0),  1.52370,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.60734, rel_tolorance);
  EXPECT_NEAR(L(0,1),  0.60734,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.52104, rel_tolorance);
  EXPECT_NEAR(L(0,2),  0.52104,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.37860, rel_tolorance);
  EXPECT_NEAR(L(0,3),  0.37860,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.01995, rel_tolorance);
  EXPECT_NEAR(L(0,4),  0.01995,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(1,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.40156, rel_tolorance);
  EXPECT_NEAR(L(1,1),  1.40156,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.69336, rel_tolorance);
  EXPECT_NEAR(L(1,2),  0.69336,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.66974, rel_tolorance);
  EXPECT_NEAR(L(1,3),  0.66974,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.13981, rel_tolorance);
  EXPECT_NEAR(L(1,4),  0.13981,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.23173, rel_tolorance);
  EXPECT_NEAR(L(2,2),  1.23173,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.43255, rel_tolorance);
  EXPECT_NEAR(L(2,3),  0.43255,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.12387, rel_tolorance);
  EXPECT_NEAR(L(2,4),  0.12387,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.19342, rel_tolorance);
  EXPECT_NEAR(L(3,3),  1.19342,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.18933, rel_tolorance);
  EXPECT_NEAR(L(3,4),  0.18933,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,3),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.36684, rel_tolorance);
  EXPECT_NEAR(L(4,4),  1.36684,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_cholesky_logit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(1.11695, rel_tolorance);
  EXPECT_NEAR(L(0,0),  1.11695,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.03594, rel_tolorance);
  EXPECT_NEAR(L(0,1),  0.03594,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.04463, rel_tolorance);
  EXPECT_NEAR(L(0,2),  0.04463,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.09123, rel_tolorance);
  EXPECT_NEAR(L(0,3),  0.09123,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.07623, rel_tolorance);
  EXPECT_NEAR(L(0,4),  0.07623,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(1,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.10397, rel_tolorance);
  EXPECT_NEAR(L(1,1),  1.10397,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.03866, rel_tolorance);
  EXPECT_NEAR(L(1,2),  0.03866,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.15833, rel_tolorance);
  EXPECT_NEAR(L(1,3),  0.15833,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.14793, rel_tolorance);
  EXPECT_NEAR(L(1,4),  0.14793,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.11470, rel_tolorance);
  EXPECT_NEAR(L(2,2),  1.11470,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.09049, rel_tolorance);
  EXPECT_NEAR(L(2,3),  0.09049,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.01420, rel_tolorance);
  EXPECT_NEAR(L(2,4),  0.01420,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.09297, rel_tolorance);
  EXPECT_NEAR(L(3,3),  1.09297,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.11357, rel_tolorance);
  EXPECT_NEAR(L(3,4),  0.11357,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,3),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.08875, rel_tolorance);
  EXPECT_NEAR(L(4,4),  1.08875,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_cholesky_probit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(1.22980, rel_tolorance);
  EXPECT_NEAR(L(0,0),  1.22980,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(0,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(0,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00100, rel_tolorance);
  EXPECT_NEAR(L(0,3),  0.00100,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00002, rel_tolorance);
  EXPECT_NEAR(L(0,4),  0.00002,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(1,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.22911, rel_tolorance);
  EXPECT_NEAR(L(1,1),  1.22911,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(1,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00680, rel_tolorance);
  EXPECT_NEAR(L(1,3),  0.00680,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(1,4),  0.00000,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.22970, rel_tolorance);
  EXPECT_NEAR(L(2,2),  1.22970,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00005, rel_tolorance);
  EXPECT_NEAR(L(2,3),  0.00005,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(2,4),  0.00000,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.22917, rel_tolorance);
  EXPECT_NEAR(L(3,3),  1.22917,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(3,4),  0.00000,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,0),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,1),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,2),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.00000, rel_tolorance);
  EXPECT_NEAR(L(4,3),  0.00000,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.22971, rel_tolorance);
  EXPECT_NEAR(L(4,4),  1.22971,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha_gaussian_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(0.112590, rel_tolorance);
  EXPECT_NEAR(alpha[0],  0.112590,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.030952, rel_tolorance);
  EXPECT_NEAR(alpha[1],  0.030952,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.265522, rel_tolorance);
  EXPECT_NEAR(alpha[2],  0.265522,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.372392, rel_tolorance);
  EXPECT_NEAR(alpha[3],  0.372392,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.660354, rel_tolorance);
  EXPECT_NEAR(alpha[4],  0.660354,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha_t_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(0.124677, rel_tolorance);
  EXPECT_NEAR(alpha[0],  0.124677,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.011322, rel_tolorance);
  EXPECT_NEAR(alpha[1],  -0.011322,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.291186, rel_tolorance);
  EXPECT_NEAR(alpha[2],  0.291186,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.414107, rel_tolorance);
  EXPECT_NEAR(alpha[3],  0.414107,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.710853, rel_tolorance);
  EXPECT_NEAR(alpha[4],  0.710853,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha_logit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(0.45082, rel_tolorance);
  EXPECT_NEAR(alpha[0],  0.45082,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.32691, rel_tolorance);
  EXPECT_NEAR(alpha[1],  -0.32691,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.43705, rel_tolorance);
  EXPECT_NEAR(alpha[2],  0.43705,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.38239, rel_tolorance);
  EXPECT_NEAR(alpha[3],  -0.38239,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.34563, rel_tolorance);
  EXPECT_NEAR(alpha[4],  -0.34563,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_alpha_probit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(-0.50646, rel_tolorance);
  EXPECT_NEAR(alpha[0],  -0.50646,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.50327, rel_tolorance);
  EXPECT_NEAR(alpha[1],  0.50327,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.50604, rel_tolorance);
  EXPECT_NEAR(alpha[2],  0.50604,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.50366, rel_tolorance);
  EXPECT_NEAR(alpha[3],  0.50366,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.50605, rel_tolorance);
  EXPECT_NEAR(alpha[4],  -0.50605,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_gaussian_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(6.8615, rel_tolorance);
  EXPECT_NEAR(nml,  6.8615,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_t_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(7.4892, rel_tolorance);
  EXPECT_NEAR(nml,  7.4892,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_logit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(3.3876, rel_tolorance);
  EXPECT_NEAR(nml,  3.3876,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_negative_marginal_likelihood_probit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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

  abs_tolorance = get_abs_tolorance(3.4990, rel_tolorance);
  EXPECT_NEAR(nml,  3.4990,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_gaussian_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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
  abs_tolorance = get_abs_tolorance(0.0074073, rel_tolorance);
  EXPECT_NEAR(dnlZ_lik,  0.0074073,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.85103, rel_tolorance);
  EXPECT_NEAR(dnlZ_ell,  -0.85103,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.57052, rel_tolorance);
  EXPECT_NEAR(dnlZ_sf2,  -0.57052,  abs_tolorance);

  // clean up
  SG_UNREF(gradient);
  SG_UNREF(parameter_dictionary);
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_t_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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
  abs_tolorance = get_abs_tolorance(-0.64932, rel_tolorance);
  EXPECT_NEAR(dnlZ_df,  -0.64932,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.15567, rel_tolorance);
  EXPECT_NEAR(dnlZ_sigma,  -0.15567,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.84364, rel_tolorance);
  EXPECT_NEAR(dnlZ_ell,  -0.84364,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.30177, rel_tolorance);
  EXPECT_NEAR(dnlZ_sf2,  -0.30177,  abs_tolorance);

  // clean up
  SG_UNREF(gradient);
  SG_UNREF(parameter_dictionary);
  SG_UNREF(inf);
}




TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_logit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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
  abs_tolorance = get_abs_tolorance(0.266464, rel_tolorance);
  EXPECT_NEAR(dnlZ_ell,  0.266464,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.068637, rel_tolorance);
  EXPECT_NEAR(dnlZ_sf2,  -0.068637,  abs_tolorance);

  // clean up
  SG_UNREF(gradient);
  SG_UNREF(parameter_dictionary);
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_marginal_likelihood_derivatives_probit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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
  abs_tolorance = get_abs_tolorance(-0.034178, rel_tolorance);
  EXPECT_NEAR(dnlZ_ell,  -0.034178,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.108246, rel_tolorance);
  EXPECT_NEAR(dnlZ_sf2,  0.108246,  abs_tolorance);

  // clean up
  SG_UNREF(gradient);
  SG_UNREF(parameter_dictionary);
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_posterior_mean_probit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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
  abs_tolorance = get_abs_tolorance(-0.50527, rel_tolorance);
  EXPECT_NEAR(approx_mean[0],  -0.50527,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.51150, rel_tolorance);
  EXPECT_NEAR(approx_mean[1],  0.51150,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.50609, rel_tolorance);
  EXPECT_NEAR(approx_mean[2],  0.50609,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(0.51073, rel_tolorance);
  EXPECT_NEAR(approx_mean[3],  0.51073,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-0.50607, rel_tolorance);
  EXPECT_NEAR(approx_mean[4],  -0.50607,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

TEST(LaplacianInferenceMethod,get_posterior_covariance_probit_likelihood)
{

  float64_t rel_tolorance = 1e-3;
  float64_t abs_tolorance;
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
  abs_tolorance = get_abs_tolorance(6.6120e-01, rel_tolorance);
  EXPECT_NEAR(approx_cov(0,0),  6.6120e-01,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-5.3908e-06, rel_tolorance);
  EXPECT_NEAR(approx_cov(0,1),  -5.3908e-06,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(4.4528e-06, rel_tolorance);
  EXPECT_NEAR(approx_cov(0,2),  4.4528e-06,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(1.0552e-03, rel_tolorance);
  EXPECT_NEAR(approx_cov(0,3),  1.0552e-03,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(2.5118e-05, rel_tolorance);
  EXPECT_NEAR(approx_cov(0,4),  2.5118e-05,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(-5.3908e-06, rel_tolorance);
  EXPECT_NEAR(approx_cov(1,0),  -5.3908e-06,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(6.6190e-01, rel_tolorance);
  EXPECT_NEAR(approx_cov(1,1),  6.6190e-01,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-3.0048e-07, rel_tolorance);
  EXPECT_NEAR(approx_cov(1,2),  -3.0048e-07,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(7.1667e-03, rel_tolorance);
  EXPECT_NEAR(approx_cov(1,3),  7.1667e-03,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(2.0193e-06, rel_tolorance);
  EXPECT_NEAR(approx_cov(1,4),  2.0193e-06,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(4.4528e-06, rel_tolorance);
  EXPECT_NEAR(approx_cov(2,0),  4.4528e-06,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-3.0048e-07, rel_tolorance);
  EXPECT_NEAR(approx_cov(2,1),  -3.0048e-07,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(6.6130e-01, rel_tolorance);
  EXPECT_NEAR(approx_cov(2,2),  6.6130e-01,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(5.4317e-05, rel_tolorance);
  EXPECT_NEAR(approx_cov(2,3),  5.4317e-05,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-8.7921e-11, rel_tolorance);
  EXPECT_NEAR(approx_cov(2,4),  -8.7921e-11,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(1.0552e-03, rel_tolorance);
  EXPECT_NEAR(approx_cov(3,0),  1.0552e-03,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(7.1667e-03, rel_tolorance);
  EXPECT_NEAR(approx_cov(3,1),  7.1667e-03,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(5.4317e-05, rel_tolorance);
  EXPECT_NEAR(approx_cov(3,2),  5.4317e-05,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(6.6181e-01, rel_tolorance);
  EXPECT_NEAR(approx_cov(3,3),  6.6181e-01,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(9.1741e-09, rel_tolorance);
  EXPECT_NEAR(approx_cov(3,4),  9.1741e-09,  abs_tolorance);

  abs_tolorance = get_abs_tolorance(2.5118e-05, rel_tolorance);
  EXPECT_NEAR(approx_cov(4,0),  2.5118e-05,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(2.0193e-06, rel_tolorance);
  EXPECT_NEAR(approx_cov(4,1),  2.0193e-06,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(-8.7921e-11, rel_tolorance);
  EXPECT_NEAR(approx_cov(4,2),  -8.7921e-11,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(9.1741e-09, rel_tolorance);
  EXPECT_NEAR(approx_cov(4,3),  9.1741e-09,  abs_tolorance);
  abs_tolorance = get_abs_tolorance(6.6130e-01, rel_tolorance);
  EXPECT_NEAR(approx_cov(4,4),  6.6130e-01,  abs_tolorance);

  // clean up
  SG_UNREF(inf);
}

#endif /* HAVE_EIGEN3 */
