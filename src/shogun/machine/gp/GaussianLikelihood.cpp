/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 */

#include <shogun/machine/gp/GaussianLikelihood.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianLikelihood::CGaussianLikelihood() : CLikelihoodModel()
{
	init();
}

CGaussianLikelihood::CGaussianLikelihood(float64_t sigma) : CLikelihoodModel()
{
	REQUIRE(sigma>0.0, "Standard deviation must be greater than zero\n")
	init();
	m_sigma=sigma;
}

void CGaussianLikelihood::init()
{
	m_sigma=1.0;
	SG_ADD(&m_sigma, "sigma", "Observation Noise.", MS_AVAILABLE);
}

CGaussianLikelihood::~CGaussianLikelihood()
{
}

CGaussianLikelihood* CGaussianLikelihood::obtain_from_generic(CLikelihoodModel* lik)
{
	ASSERT(lik!=NULL);

	if (lik->get_model_type()!=LT_GAUSSIAN)
		SG_SERROR("Provided likelihood is not of type CGaussianLikelihood!\n")

	SG_REF(lik);
	return (CGaussianLikelihood*)lik;
}

SGVector<float64_t> CGaussianLikelihood::evaluate_means(SGVector<float64_t> mu,
		SGVector<float64_t> s2)
{
	return SGVector<float64_t>(mu);
}

SGVector<float64_t> CGaussianLikelihood::evaluate_variances(SGVector<float64_t> mu,
		SGVector<float64_t> s2)
{
	SGVector<float64_t> result(s2);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	eigen_result+=CMath::sq(m_sigma)*VectorXd::Ones(result.vlen);

	return result;
}

float64_t CGaussianLikelihood::get_log_probability_f(CLabels* lab,
		SGVector<float64_t> func)
{
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
		"Labels must be type of CRegressionLabels\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute log probability: lp=-(y-f).^2./sigma^2/2-log(2*pi*sigma^2)/2
	eigen_result=eigen_y-eigen_f;
	eigen_result=-eigen_result.cwiseProduct(eigen_result)/(2*CMath::sq(m_sigma))-
		VectorXd::Ones(result.vlen)*log(2*CMath::PI*CMath::sq(m_sigma))/2.0;

	return eigen_result.sum();
}

SGVector<float64_t> CGaussianLikelihood::get_log_probability_derivative_f(
		CLabels* lab, SGVector<float64_t> func, index_t i)
{
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
		"Labels must be type of CRegressionLabels\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// set result=y-f
	eigen_result=eigen_y-eigen_f;

	// compute derivatives of log probability wrt f
	if (i == 1)
		eigen_result/=CMath::sq(m_sigma);

	else if (i == 2)
		eigen_result=-VectorXd::Ones(result.vlen)/CMath::sq(m_sigma);

	else if (i == 3)
		eigen_result=VectorXd::Zero(result.vlen);

	else
		SG_ERROR("Invalid Index for Likelihood Derivative\n")

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_first_derivative(CLabels* lab,
		TParameter* param,  CSGObject* obj, SGVector<float64_t> func)
{
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
		"Labels must be type of CRegressionLabels\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	if (strcmp(param->m_name, "sigma") || obj!=this)
		return SGVector<float64_t>();

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of log probability wrt sigma:
	// lp_dsigma=(y-f).^2/sigma^2-1
	eigen_result=eigen_y-eigen_f;
	eigen_result=eigen_result.cwiseProduct(eigen_result)/CMath::sq(m_sigma);
	eigen_result-=VectorXd::Ones(result.vlen);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_second_derivative(CLabels* lab,
		TParameter* param,  CSGObject* obj, SGVector<float64_t> func)
{
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
		"Labels must be type of CRegressionLabels\n")

	if (strcmp(param->m_name, "sigma") || obj!=this)
		return SGVector<float64_t>();

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of the first derivative of log probability wrt sigma:
	// dlp_dsigma=2*(f-y)/sigma^2
	eigen_result=2*(eigen_f-eigen_y)/CMath::sq(m_sigma);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_third_derivative(CLabels* lab,
		TParameter* param, CSGObject* obj, SGVector<float64_t> func)
{
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
		"Labels must be type of CRegressionLabels\n")

	if (strcmp(param->m_name, "sigma") || obj!=this)
		return SGVector<float64_t>();

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	// compute derivative of the second derivative of log probability wrt sigma:
	// d2lp_dsigma=1/sigma^2
	eigen_result=2*VectorXd::Ones(result.vlen)/CMath::sq(m_sigma);

	return result;
}

#endif //HAVE_EIGEN3
