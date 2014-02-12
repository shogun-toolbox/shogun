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
#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
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

	// specify GP regression with FITC inference
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

TEST(FITCInferenceMethod,get_cholesky_sparse)
{
	// create some easy regression data with latent features:
	// y approximately equals to x^sin(x)
	index_t n=6, m=3;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> lat_feat_train(1, m);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.81263;
	feat_train[1]=0.99976;
	feat_train[2]=1.17037;
	feat_train[3]=1.51752;
	feat_train[4]=1.57765;
	feat_train[5]=3.89440;

	lat_feat_train[0]=0.00000;
	lat_feat_train[1]=2.00000;
	lat_feat_train[2]=4.00000;

	lab_train[0]=0.86015;
	lab_train[1]=0.99979;
	lab_train[2]=1.15589;
	lab_train[3]=1.51662;
	lab_train[4]=1.57764;
	lab_train[5]=0.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 8.0 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 8.0);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(0.5);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, latent_features_train);
	inf->set_scale(2.5);

	// comparison of posterior cholesky with result from GPML package:
	SGMatrix<float64_t> L=inf->get_cholesky();

	EXPECT_NEAR(L(0,0), -0.174157, 1E-6);
	EXPECT_NEAR(L(0,1), 0.108204, 1E-6);
	EXPECT_NEAR(L(0,2), -0.037098, 1E-6);

	EXPECT_NEAR(L(1,0), 0.108204, 1E-6);
	EXPECT_NEAR(L(1,1), -0.296029, 1E-6);
	EXPECT_NEAR(L(1,2), 0.155598, 1E-6);

	EXPECT_NEAR(L(2,0), -0.037098, 1E-6);
	EXPECT_NEAR(L(2,1), 0.155598, 1E-6);
	EXPECT_NEAR(L(2,2), -0.237840, 1E-6);

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

	// specify GP regression with FITC inference
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

TEST(FITCInferenceMethod,get_alpha_sparse)
{
	// create some easy regression data with latent features:
	// y approximately equals to x^sin(x)
	index_t n=6, m=3;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> lat_feat_train(1, m);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.81263;
	feat_train[1]=0.99976;
	feat_train[2]=1.17037;
	feat_train[3]=1.51752;
	feat_train[4]=1.57765;
	feat_train[5]=3.89440;

	lat_feat_train[0]=0.00000;
	lat_feat_train[1]=2.00000;
	lat_feat_train[2]=4.00000;

	lab_train[0]=0.86015;
	lab_train[1]=0.99979;
	lab_train[2]=1.15589;
	lab_train[3]=1.51662;
	lab_train[4]=1.57764;
	lab_train[5]=0.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 8.0 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 8.0);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(0.5);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, latent_features_train);
	inf->set_scale(2.5);

	// comparison of posterior alpha with result from GPML package:
	SGVector<float64_t> alpha=inf->get_alpha();

	EXPECT_NEAR(alpha[0], -0.20898, 1E-5);
	EXPECT_NEAR(alpha[1], 0.48719, 1E-5);
	EXPECT_NEAR(alpha[2], -0.20836, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_negative_log_marginal_likelihood)
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

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, latent_features_train);

	// comparison of posterior negative marginal likelihood with
	// result from GPML package:
	// nlZ =
	// 0.84354
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nml, 0.84354, 1E-5);

	// clean up
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_negative_log_marginal_likelihood_sparse)
{
	// create some easy regression data with latent features:
	// y approximately equals to x^sin(x)
	index_t n=6, m=3;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> lat_feat_train(1, m);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.81263;
	feat_train[1]=0.99976;
	feat_train[2]=1.17037;
	feat_train[3]=1.51752;
	feat_train[4]=1.57765;
	feat_train[5]=3.89440;

	lat_feat_train[0]=0.00000;
	lat_feat_train[1]=2.00000;
	lat_feat_train[2]=4.00000;

	lab_train[0]=0.86015;
	lab_train[1]=0.99979;
	lab_train[2]=1.15589;
	lab_train[3]=1.51662;
	lab_train[4]=1.57764;
	lab_train[5]=0.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 8.0 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 8.0);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(0.5);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, latent_features_train);
	inf->set_scale(2.5);

	// comparison of posterior negative marginal likelihood with result from
	// GPML package
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nml, 6.6776, 1E-4);

	// clean up
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_marginal_likelihood_derivatives)
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
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.1);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

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

	float64_t dnlZ_ell=4*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];

	// comparison of partial derivatives of negative log marginal likelihood
	// with result from GPML package:
	// lik =  2.1930
	// cov =
	// -1.67233
	// 0.55979
	EXPECT_NEAR(dnlZ_lik, 2.1930, 1E-4);
	EXPECT_NEAR(dnlZ_ell, -1.67233, 1E-5);
	EXPECT_NEAR(dnlZ_sf2, 0.55979, 1E-5);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_marginal_likelihood_derivatives_sparse)
{
	// create some easy regression data with latent features:
	// y approximately equals to x^sin(x)
	index_t n=6, m=3;

	SGMatrix<float64_t> feat_train(1, n);
	SGMatrix<float64_t> lat_feat_train(1, m);
	SGVector<float64_t> lab_train(n);

	feat_train[0]=0.81263;
	feat_train[1]=0.99976;
	feat_train[2]=1.17037;
	feat_train[3]=1.51752;
	feat_train[4]=1.57765;
	feat_train[5]=3.89440;

	lat_feat_train[0]=0.00000;
	lat_feat_train[1]=2.00000;
	lat_feat_train[2]=4.00000;

	lab_train[0]=0.86015;
	lab_train[1]=0.99979;
	lab_train[2]=1.15589;
	lab_train[3]=1.51662;
	lab_train[4]=1.57764;
	lab_train[5]=0.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(
			feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	float64_t ell=2.0;

	// choose Gaussian kernel with sigma = 2*ell*ell = 8.0 and zero mean
	// function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*CMath::sq(ell));
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.5);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, lik, latent_features_train);
	inf->set_scale(2.5);

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

	float64_t dnlZ_ell=4*CMath::sq(ell)*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=2.5*(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];

	// comparison of partial derivatives of negative log marginal likelihood
	// with result from GPML package:
	EXPECT_NEAR(dnlZ_lik, 2.4566, 1E-4);
	EXPECT_NEAR(dnlZ_ell, -2.5158, 1E-4);
	EXPECT_NEAR(dnlZ_sf2, 2.7023, 1E-4);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

#endif /* HAVE_EIGEN3 */
