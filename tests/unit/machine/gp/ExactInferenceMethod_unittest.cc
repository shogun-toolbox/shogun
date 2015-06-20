/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 *
 * Written (W) 2013 Roman Votyakov
 * Written (w) 2014 Wu Lin
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
#include <shogun/machine/gp/ExactInferenceMethod.h>
#include <shogun/machine/gp/ZeroMean.h>
#include <shogun/machine/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/Math.h>
#include <shogun/machine/gp/ConstMean.h>

using namespace shogun;

TEST(ExactInferenceMethod,get_cholesky)
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

	/* do some checks against gpml toolbox*/
	// L =
	// 1.414213562373095   0.386132930109494   0.062877078699608
	//                 0   1.360478357154224   0.383538270389077
	//                 0                   0   1.359759121359794
	SGMatrix<float64_t> L=inf->get_cholesky();
	EXPECT_LE(CMath::abs(L(0,0)-1.414213562373095), 10E-15);
	EXPECT_LE(CMath::abs(L(0,1)-0.386132930109494), 10E-15);
	EXPECT_LE(CMath::abs(L(0,2)-0.062877078699608), 10E-15);
	EXPECT_LE(CMath::abs(L(1,0)-0), 10E-15);
	EXPECT_LE(CMath::abs(L(1,1)-1.360478357154224), 10E-15);
	EXPECT_LE(CMath::abs(L(1,2)-0.383538270389077), 10E-15);
	EXPECT_LE(CMath::abs(L(2,0)-0), 10E-15);
	EXPECT_LE(CMath::abs(L(2,1)-0), 10E-15);
	EXPECT_LE(CMath::abs(L(2,2)-1.359759121359794), 10E-15);

	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_alpha)
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

	/* do some checks against gpml toolbox*/

	// alpha =
	// -0.121668320184276
	// 0.396533145765454
	// 0.301389368713216
	SGVector<float64_t> alpha=inf->get_alpha();
	EXPECT_LE(CMath::abs(alpha[0]+0.121668320184276), 10E-15);
	EXPECT_LE(CMath::abs(alpha[1]-0.396533145765454), 10E-15);
	EXPECT_LE(CMath::abs(alpha[2]-0.301389368713216), 10E-15);

	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_negative_marginal_likelihood)
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

	/* do some checks against gpml toolbox*/
	// nlZ =
	// 4.017065867797999
	float64_t nml=inf->get_negative_log_marginal_likelihood();
	EXPECT_LE(CMath::abs(nml-4.017065867797999), 10E-15);

	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_negative_log_marginal_likelihood_derivatives)
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

	float64_t ell=0.1;

	// choose Gaussian kernel with width = 2 * ell^2 = 0.02 and zero mean function
	CGaussianKernel* kernel=new CGaussianKernel(10, 2*ell*ell);

	CZeroMean* mean=new CZeroMean();

	// Gaussian likelihood with sigma = 0.25
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.25);

	// specify GP regression with exact inference
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, features_train,
			mean, labels_train, lik);

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

	// in GPML package: dK(i,j)/dell = sf2 * exp(-(x(i) - x(j))^2/(2*ell^2)) *
	// (x(i) - x(j))^2 / (ell^2), but in SHOGUN we compute: dK(i,j)/dw = sf2 *
	// exp(-(x(i) - x(j))^2/w) * (x(i) - x(j))^2 / (w^2), so if w = 2 * ell^2,
	// then dK(i,j)/dell = 4 * ell^2 * dK(i,j)/dw.
	float64_t dnlZ_ell=4*ell*ell*(gradient->get_element(width_param))[0];
	float64_t dnlZ_sf2=(gradient->get_element(scale_param))[0];
	float64_t dnlZ_lik=(gradient->get_element(sigma_param))[0];

	// comparison of partial derivatives of negative marginal likelihood with
	// result from GPML package:
	// lik =  0.10638
	// cov =
	// -0.015133
	// 1.699483
	EXPECT_NEAR(dnlZ_lik, 0.10638, 1E-5);
	EXPECT_NEAR(dnlZ_ell, -0.015133, 1E-6);
	EXPECT_NEAR(dnlZ_sf2, 1.699483, 1E-6);

	SG_UNREF(gradient);
	SG_UNREF(parameter_dictionary);
	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_posterior_mean)
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

	// comparison of posterior mean with result from GPML package
	// 0.3613824450357372430886471
	// 0.0032614946619217077480868
	// -1.0762845465175703285609643
	// 1.2348344945975837649854157
	// -0.0750001215533616788500026
	SGVector<float64_t> mu=inf->get_posterior_mean();

	EXPECT_NEAR(mu[0], 0.36138244503573730, 1E-15);
	EXPECT_NEAR(mu[1], 0.00326149466192171, 1E-15);
	EXPECT_NEAR(mu[2], -1.07628454651757055, 1E-15);
	EXPECT_NEAR(mu[3], 1.23483449459758354, 1E-15);
	EXPECT_NEAR(mu[4], -0.07500012155336166, 1E-15);

	// clean up
	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_posterior_covariance)
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

	float64_t rel_tolerance = 1e-2;
	float64_t abs_tolerance;

	// comparison of posterior approximation covariance with result from GPML
	// package
	//0.0569394684731059849691626 0.0000000000000000000053289 0.0000000000000131879791500 0.0000136088380251857096650 -0.0000000002307207284608970
	//0.0000000000000000000053289 0.0569395017793594276911406 -0.0000000000000000000000000 -0.0000000000000000000000130 0.0000000000000000000000000
	//0.0000000000000131879791500 -0.0000000000000000000000000 0.0569395017611918560773709 -0.0000000000053885704468227 0.0000003178376535342790039
	//0.0000136088380251857249116 -0.0000000000000000000000130 -0.0000000000053885704468227 0.0569394684715077217807000 0.0000000942718277538447049
	//-0.0000000002307207284608972 0.0000000000000000000000000 0.0000003178376535342790568 0.0000000942718277538447182 0.0569395017595935928889084
	SGMatrix<float64_t> Sigma=inf->get_posterior_covariance();

	abs_tolerance = CMath::get_abs_tolerance(0.0569394684731059849691626, rel_tolerance);
	EXPECT_NEAR(Sigma(0,0),  0.0569394684731059849691626,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000053289, rel_tolerance);
	EXPECT_NEAR(Sigma(0,1),  0.0000000000000000000053289,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000131879791500, rel_tolerance);
	EXPECT_NEAR(Sigma(0,2),  0.0000000000000131879791500,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000136088380251857096650, rel_tolerance);
	EXPECT_NEAR(Sigma(0,3),  0.0000136088380251857096650,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000002307207284608970, rel_tolerance);
	EXPECT_NEAR(Sigma(0,4),  -0.0000000002307207284608970,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000053289, rel_tolerance);
	EXPECT_NEAR(Sigma(1,0),  0.0000000000000000000053289,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0569395017793594276911406, rel_tolerance);
	EXPECT_NEAR(Sigma(1,1),  0.0569395017793594276911406,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000000000000000000, rel_tolerance);
	EXPECT_NEAR(Sigma(1,2),  -0.0000000000000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000000000000000130, rel_tolerance);
	EXPECT_NEAR(Sigma(1,3),  -0.0000000000000000000000130,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000000000, rel_tolerance);
	EXPECT_NEAR(Sigma(1,4),  0.0000000000000000000000000,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000131879791500, rel_tolerance);
	EXPECT_NEAR(Sigma(2,0),  0.0000000000000131879791500,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000000000000000000, rel_tolerance);
	EXPECT_NEAR(Sigma(2,1),  -0.0000000000000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0569395017611918560773709, rel_tolerance);
	EXPECT_NEAR(Sigma(2,2),  0.0569395017611918560773709,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000053885704468227, rel_tolerance);
	EXPECT_NEAR(Sigma(2,3),  -0.0000000000053885704468227,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000003178376535342790039, rel_tolerance);
	EXPECT_NEAR(Sigma(2,4),  0.0000003178376535342790039,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(0.0000136088380251857249116, rel_tolerance);
	EXPECT_NEAR(Sigma(3,0),  0.0000136088380251857249116,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000000000000000130, rel_tolerance);
	EXPECT_NEAR(Sigma(3,1),  -0.0000000000000000000000130,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.0000000000053885704468227, rel_tolerance);
	EXPECT_NEAR(Sigma(3,2),  -0.0000000000053885704468227,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0569394684715077217807000, rel_tolerance);
	EXPECT_NEAR(Sigma(3,3),  0.0569394684715077217807000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000942718277538447049, rel_tolerance);
	EXPECT_NEAR(Sigma(3,4),  0.0000000942718277538447049,  abs_tolerance);

	abs_tolerance = CMath::get_abs_tolerance(-0.0000000002307207284608972, rel_tolerance);
	EXPECT_NEAR(Sigma(4,0),  -0.0000000002307207284608972,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000000000000000000000, rel_tolerance);
	EXPECT_NEAR(Sigma(4,1),  0.0000000000000000000000000,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000003178376535342790568, rel_tolerance);
	EXPECT_NEAR(Sigma(4,2),  0.0000003178376535342790568,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0000000942718277538447182, rel_tolerance);
	EXPECT_NEAR(Sigma(4,3),  0.0000000942718277538447182,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.0569395017595935928889084, rel_tolerance);
	EXPECT_NEAR(Sigma(4,4),  0.0569395017595935928889084,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

TEST(ExactInferenceMethod,get_posterior_mean2)
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
	CConstMean* mean=new CConstMean(1.0);

	// Gaussian likelihood with sigma = 0.25
	CGaussianLikelihood* lik=new CGaussianLikelihood(0.25);

	// specify GP regression with exact inference
	CExactInferenceMethod* inf=new CExactInferenceMethod(kernel, features_train,
			mean, labels_train, lik);
	inf->set_scale(0.8);

	float64_t rel_tolerance = 1e-3;
	float64_t abs_tolerance;

	// comparison of posterior mean with result from GPML package
	// -0.5498667882510408499996402
	// -0.9077705338078291275039078
	// -1.9873216600130905185039865
	// 0.3235837493820302168678893
	// -0.9860387397670280495987072
	SGVector<float64_t> mu=inf->get_posterior_mean();

	abs_tolerance = CMath::get_abs_tolerance(-0.5498667882510408499996402, rel_tolerance);
	EXPECT_NEAR(mu[0],  -0.5498667882510408499996402,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.9077705338078291275039078, rel_tolerance);
	EXPECT_NEAR(mu[1],  -0.9077705338078291275039078,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-1.9873216600130905185039865, rel_tolerance);
	EXPECT_NEAR(mu[2],  -1.9873216600130905185039865,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(0.3235837493820302168678893, rel_tolerance);
	EXPECT_NEAR(mu[3],  0.3235837493820302168678893,  abs_tolerance);
	abs_tolerance = CMath::get_abs_tolerance(-0.9860387397670280495987072, rel_tolerance);
	EXPECT_NEAR(mu[4],  -0.9860387397670280495987072,  abs_tolerance);

	// clean up
	SG_UNREF(inf);
}

#endif /* HAVE_EIGEN3 */
