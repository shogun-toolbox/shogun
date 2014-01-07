/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <lib/config.h>

#ifdef HAVE_EIGEN3

#include <labels/RegressionLabels.h>
#include <features/DenseFeatures.h>
#include <kernel/GaussianKernel.h>
#include <machine/gp/ExactInferenceMethod.h>
#include <machine/gp/ZeroMean.h>
#include <machine/gp/GaussianLikelihood.h>
#include <gtest/gtest.h>

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

	// comparison of posterior approximation covariance with result from GPML
	// package
	SGMatrix<float64_t> Sigma=inf->get_posterior_covariance();

	EXPECT_NEAR(Sigma(0,0), 6.03558716779569e-01, 1E-15);
	EXPECT_NEAR(Sigma(0,1), 6.31493961602982e-19, 1E-15);
	EXPECT_NEAR(Sigma(0,2), 8.24248696877457e-16, 1E-15);
	EXPECT_NEAR(Sigma(0,3), 1.61269208627034e-03, 1E-15);
	EXPECT_NEAR(Sigma(0,4), -1.44168641713900e-11, 1E-15);

	EXPECT_NEAR(Sigma(1,0), 6.31493961602982e-19, 1E-15);
	EXPECT_NEAR(Sigma(1,1), 6.03558718861210e-01, 1E-15);
	EXPECT_NEAR(Sigma(1,2), -7.89915715089744e-34, 1E-15);
	EXPECT_NEAR(Sigma(1,3), -8.15123749995260e-25, 1E-15);
	EXPECT_NEAR(Sigma(1,4), 1.38193977359810e-29, 1E-15);

	EXPECT_NEAR(Sigma(2,0), 8.24248696877457e-16, 1E-15);
	EXPECT_NEAR(Sigma(2,1), -7.89915715089744e-34, 1E-15);
	EXPECT_NEAR(Sigma(2,2), 6.03558718860075e-01, 1E-15);
	EXPECT_NEAR(Sigma(2,3), -3.36784805037017e-13, 1E-15);
	EXPECT_NEAR(Sigma(2,4), 3.76650331606094e-05, 1E-15);

	EXPECT_NEAR(Sigma(3,0), 1.61269208627034e-03, 1E-15);
	EXPECT_NEAR(Sigma(3,1), -8.15123749995260e-25, 1E-15);
	EXPECT_NEAR(Sigma(3,2), -3.36784805037017e-13, 1E-15);
	EXPECT_NEAR(Sigma(3,3), 6.03558716779469e-01, 1E-15);
	EXPECT_NEAR(Sigma(3,4), 1.11715217566074e-05, 1E-15);

	EXPECT_NEAR(Sigma(4,0), -1.44168641713900e-11, 1E-15);
	EXPECT_NEAR(Sigma(4,1), 1.38193977359810e-29, 1E-15);
	EXPECT_NEAR(Sigma(4,2), 3.76650331606094e-05, 1E-15);
	EXPECT_NEAR(Sigma(4,3), 1.11715217566074e-05, 1E-15);
	EXPECT_NEAR(Sigma(4,4), 6.03558718859975e-01, 1E-15);

	// clean up
	SG_UNREF(inf);
}

#endif /* HAVE_EIGEN3 */
