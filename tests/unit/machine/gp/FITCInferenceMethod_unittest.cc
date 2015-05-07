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

#include <shogun/labels/RegressionLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/ConstMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <shogun/machine/gp/GaussianARDFITCKernel.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(FITCInferenceMethod,get_cholesky)
{
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	float64_t sigma=0.1;
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, inducing_features_train);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

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
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 8.0 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 8.0);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	float64_t sigma=0.5;
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, inducing_features_train);
	inf->set_scale(2.5);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

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
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	float64_t sigma=0.1;
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, inducing_features_train);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

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
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 8.0 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 8.0);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	float64_t sigma=0.5;
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, inducing_features_train);
	inf->set_scale(2.5);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

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
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	float64_t sigma=0.1;
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, inducing_features_train);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

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
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 8.0 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 8.0);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	float64_t sigma=0.5;
	CGaussianLikelihood* liklihood=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, liklihood, inducing_features_train);
	inf->set_scale(2.5);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

	// comparison of posterior negative marginal likelihood with result from
	// GPML package
	float64_t nml=inf->get_negative_log_marginal_likelihood();

	EXPECT_NEAR(nml, 6.6776, 1E-4);

	// clean up
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_marginal_likelihood_derivatives)
{
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2);
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	float64_t sigma=0.1;
	CGaussianLikelihood* lik=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		mean, labels_train, lik, inducing_features_train);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("log_width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");

	float64_t dnlZ_ell=(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];

	TParameter* noise_param=inf->m_gradient_parameters->get_parameter("inducing_noise");
	float64_t dnlZ_noise=(gradient->get_element(noise_param))[0];
	dnlZ_lik+=dnlZ_noise;
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
	// create some easy regression data with inducing features:
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
	CDenseFeatures<float64_t>* inducing_features_train=new CDenseFeatures<float64_t>(
			lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	float64_t ell=2.0;

	// choose Gaussian kernel with sigma = 2*ell*ell = 8.0 and zero mean
	// function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*CMath::sq(ell));
	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.5
	float64_t sigma=0.5;
	CGaussianLikelihood* lik=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		   mean, labels_train, lik, inducing_features_train);
	inf->set_scale(2.5);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* width_param=kernel->m_gradient_parameters->get_parameter("log_width");
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");

	float64_t dnlZ_ell=(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=2.5*(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];

	TParameter* noise_param=inf->m_gradient_parameters->get_parameter("inducing_noise");
	float64_t dnlZ_noise=(gradient->get_element(noise_param))[0];
	dnlZ_lik+=dnlZ_noise;

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
#ifdef HAVE_LINALG_LIB
TEST(FITCInferenceMethod,get_marginal_likelihood_derivatives_for_ARD_kernel1)
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

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=1.0;
	CLinearARDKernel* kernel=new CGaussianARDFITCKernel(10, 2*ell*ell);

	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	float64_t weight1=0.02;
	float64_t weight2=-0.4;
	float64_t weight3=0;
	float64_t weight4=0.01;
	weights(0,0)=weight1;
	weights(0,1)=weight2;
	weights(1,0)=weight3;
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

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");
	TParameter* noise_param=inf->m_gradient_parameters->get_parameter("inducing_noise");
	TParameter* weights_param=kernel->m_gradient_parameters->get_parameter("weights");
		TParameter* mean_param=mean->m_gradient_parameters->get_parameter("mean");

	float64_t dnlZ_sf2=scale*(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_noise=(gradient->get_element(noise_param))[0];
	float64_t dnlZ_mean=(gradient->get_element(mean_param))[0];

	SGVector<float64_t> dnlz_weights_vec=gradient->get_element(weights_param);
	SGMatrix<float64_t> dnlz_weights(dnlz_weights_vec.vector,t_dim, dim, false);
	//diagonal elements are in the log domain
	dnlz_weights(0,0)*=weight1;
	dnlz_weights(1,1)*=weight4;

	dnlZ_lik+=dnlZ_noise;

	// comparison of partial derivatives of negative log marginal likelihood
	// with result from GPML 3.5 package:
	//mean: 0.110630684381327
	//cov: 4.756295043167128
	//lik: 0.260126513715885
	//weights=[
	//-0.326900947549438  -6.119973491296843
	//0                    0.002268754179272]
	//Note that in GPML 3.5, weights is a upper-triangular matrix
	//diagonal elements of the matrix are in the log domain
	//while off-diagonal elements are in natrual domain

	EXPECT_NEAR(dnlz_weights(0,0), -0.326900947549438, 1E-10);
	EXPECT_NEAR(dnlz_weights(0,1), -6.119973491296843, 1E-10);
	//dnlz_weights(1,0) should be different (non-zero)
	//since the kernel matrix in the Shogun's implementation does not
	//have the "upper triangular matrix" constraint
	EXPECT_NEAR(dnlz_weights(1,1), 0.002268754179272, 1E-10);

	EXPECT_NEAR(dnlZ_lik, 0.260126513715885,1E-10);
	EXPECT_NEAR(dnlZ_mean, 0.110630684381327, 1E-10);
	EXPECT_NEAR(dnlZ_sf2, 4.756295043167128, 1E-10);

	//Note that in the latest GPML3.5, derivatives wrt xu (covSEfact) does not support yet
	//In Shohun's implementation, the derivatives wrt xu are supported.
	//TParameter* lat_param=inf->m_gradient_parameters->get_parameter("inducing_features");
	//SGVector<float64_t> tmp=gradient->get_element(lat_param);
	//SGMatrix<float64_t> deriv_lat(tmp.vector, dim, m, false);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_marginal_likelihood_derivatives_for_ARD_kernel2)
{
	index_t n=6;
	index_t dim=3;
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

	feat_train(2,0)=1.5;
	feat_train(2,1)=-0.4576;
	feat_train(2,2)=-4.17637;
	feat_train(2,3)=12.56752;
	feat_train(2,4)=6.57765;
	feat_train(2,5)=-2.89440;

	lat_feat_train(0,0)=1.00000;
	lat_feat_train(0,1)=23.00000;
	lat_feat_train(0,2)=4.00000;

	lat_feat_train(1,0)=3.00000;
	lat_feat_train(1,1)=2.00000;
	lat_feat_train(1,2)=-5.00000;

	lat_feat_train(2,0)=-1.00000;
	lat_feat_train(2,1)=3.00000;
	lat_feat_train(2,2)=-5.00000;

	lab_train[0]=0.46015;
	lab_train[1]=0.69979;
	lab_train[2]=2.15589;
	lab_train[3]=1.51672;
	lab_train[4]=3.59764;
	lab_train[5]=2.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=1.0;
	CLinearARDKernel* kernel=new CGaussianARDFITCKernel(10, 2*ell*ell);

	int32_t t_dim=2;
	SGMatrix<float64_t> weights(t_dim,dim);
	//the weights is a upper triangular matrix since GPML 3.5 only supports this type
	//weights =
	//0.02 -0.40 -0.01
	//0    0.03  -0.20
	float64_t weight00=0.02;
	float64_t weight11=0.03;
	weights(0,0)=weight00;
	weights(0,1)=-0.40;
	weights(0,2)=-0.01;

	weights(1,0)=0;
	weights(1,1)=weight11;
	weights(1,2)=-0.20;

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

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	TParameter* scale_param=inf->m_gradient_parameters->get_parameter("scale");
	TParameter* sigma_param=lik->m_gradient_parameters->get_parameter("sigma");
	TParameter* noise_param=inf->m_gradient_parameters->get_parameter("inducing_noise");
	TParameter* weights_param=kernel->m_gradient_parameters->get_parameter("weights");
	TParameter* mean_param=mean->m_gradient_parameters->get_parameter("mean");

	float64_t dnlZ_sf2=scale*(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];
	float64_t dnlZ_noise=(gradient->get_element(noise_param))[0];
	float64_t dnlZ_mean=(gradient->get_element(mean_param))[0];

	SGVector<float64_t> dnlz_weights_vec=gradient->get_element(weights_param);
	SGMatrix<float64_t> dnlz_weights(dnlz_weights_vec.vector,t_dim, dim, false);
	//diagonal elements are in the log domain
	dnlz_weights(0,0)*=weight00;
	dnlz_weights(1,1)*=weight11;

	dnlZ_lik+=dnlZ_noise;

	// comparison of partial derivatives of negative log marginal likelihood
	// with result from GPML 3.5 package:
	//
	//mean: 0.080187488475807
	//cov: 5.226913608720613
	//lik: 0.095062798184398
	//
	//weights=[
	// 0.014310121599512  -1.258335093159427   0.021881344855798
	// 0                  0.004443216429718    -1.253522960589666]
	//
	//Note that in GPML 3.5, weights is a upper-triangular matrix
	//diagonal elements of the matrix are in the log domain
	//while off-diagonal elements are in natrual domain

	EXPECT_NEAR(dnlz_weights(0,0), 0.014310121599512, 1E-10);
	EXPECT_NEAR(dnlz_weights(0,1), -1.258335093159427, 1E-10);
	EXPECT_NEAR(dnlz_weights(0,2), 0.021881344855798, 1E-10);

	//dnlz_weights(1,0) should be different (non-zero)
	//since the kernel matrix in the Shogun's implementation does not
	//have the "upper triangular matrix" constraint
	EXPECT_NEAR(dnlz_weights(1,1), 0.004443216429718, 1E-10);
	EXPECT_NEAR(dnlz_weights(1,2), -1.253522960589666, 1E-10);


	EXPECT_NEAR(dnlZ_lik, 0.095062798184398,1E-10);
	EXPECT_NEAR(dnlZ_mean, 0.080187488475807, 1E-10);
	EXPECT_NEAR(dnlZ_sf2, 5.226913608720613, 1E-10);

	//Note that in the latest GPML3.5, derivatives wrt xu (covSEfact) does not support yet
	//In Shohun's implementation, the derivatives wrt xu are supported.
	//TParameter* lat_param=inf->m_gradient_parameters->get_parameter("inducing_features");
	//SGVector<float64_t> tmp=gradient->get_element(lat_param);
	//SGMatrix<float64_t> deriv_lat(tmp.vector, dim, m, false);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(FITCInferenceMethod,get_marginal_likelihood_derivatives_for_inducing_features)
{
	index_t n=6;
	index_t dim=2;
	index_t m=3;
	float64_t rel_tolorance=1e-5;
	float64_t abs_tolorance;

	SGMatrix<float64_t> feat_train(dim, n);
	SGMatrix<float64_t> lat_feat_train(dim, m);
	SGVector<float64_t> lab_train(n);

	feat_train(0,0)=0.81263;
	feat_train(0,1)=0.99976;
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
	lat_feat_train(1,2)=5.00000;

	lab_train[0]=0.46015;
	lab_train[1]=0.69979;
	lab_train[2]=2.15589;
	lab_train[3]=1.51672;
	lab_train[4]=3.59764;
	lab_train[5]=2.39475;

	// shogun representation of features and labels
	CDenseFeatures<float64_t>* features_train=new CDenseFeatures<float64_t>(feat_train);
	CDenseFeatures<float64_t>* latent_features_train=new CDenseFeatures<float64_t>(lat_feat_train);
	CRegressionLabels* labels_train=new CRegressionLabels(lab_train);

	// choose Gaussian kernel with sigma = 2 and zero mean function
	float64_t ell=1.0;
	CLinearARDKernel* kernel=new CGaussianARDFITCKernel(10, 2*ell*ell);
	float64_t weight1=3.0;
	float64_t weight2=2.0;
	SGVector<float64_t> weights(2);
	weights[0]=1.0/weight1;
	weights[1]=1.0/weight2;//bug missing parameter
	kernel->set_vector_weights(weights);

	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.1
	float64_t sigma=0.1;
	CGaussianLikelihood* lik=new CGaussianLikelihood(sigma);

	// specify GP regression with FITC inference
	CFITCInferenceMethod* inf=new CFITCInferenceMethod(kernel, features_train,
		mean, labels_train, lik, latent_features_train);

	float64_t ind_noise=1e-6*CMath::sq(sigma);
	inf->set_inducing_noise(ind_noise);

	float64_t scale=3.0;
	inf->set_scale(scale);

	// build parameter dictionary
	CMap<TParameter*, CSGObject*>* parameter_dictionary=new CMap<TParameter*, CSGObject*>();
	inf->build_gradient_parameter_dictionary(parameter_dictionary);

	// compute derivatives wrt parameters
	CMap<TParameter*, SGVector<float64_t> >* gradient=
		inf->get_negative_log_marginal_likelihood_derivatives(parameter_dictionary);

	// get parameters to compute derivatives
	// comparison of partial derivatives of negative log marginal likelihood
	// with result from GPML package:

 	// lik = 0.010184026598875
	// cov =
	// -1.486266337025983
	// -1.655475869620671
	// 4.122329314701688
	// xu = (for kernel covSEard)
	// -0.002696862602213   0.155565442024638   0.182559104195480
	// -0.122912900116072   0.243421751386864   0.069745929240874

	TParameter* lat_param=inf->m_gradient_parameters->get_parameter("inducing_features");
	SGVector<float64_t> tmp=gradient->get_element(lat_param);
	SGMatrix<float64_t> deriv_lat(tmp.vector, dim, m, false);

	abs_tolorance = CMath::get_abs_tolerance(-0.002696862602213, rel_tolorance);
	EXPECT_NEAR(deriv_lat(0,0),  -0.002696862602213,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.155565442024638, rel_tolorance);
	EXPECT_NEAR(deriv_lat(0,1),  0.155565442024638,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.182559104195480, rel_tolorance);
	EXPECT_NEAR(deriv_lat(0,2),  0.182559104195480,  abs_tolorance);

	abs_tolorance = CMath::get_abs_tolerance(-0.122912900116072, rel_tolorance);
	EXPECT_NEAR(deriv_lat(1,0),  -0.122912900116072,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.243421751386864, rel_tolorance);
	EXPECT_NEAR(deriv_lat(1,1),  0.243421751386864,  abs_tolorance);
	abs_tolorance = CMath::get_abs_tolerance(0.069745929240874, rel_tolorance);
	EXPECT_NEAR(deriv_lat(1,2),  0.069745929240874,  abs_tolorance);

	// clean up
	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}
#endif /* HAVE_LINALG_LIB */

#endif /* HAVE_EIGEN3 */
