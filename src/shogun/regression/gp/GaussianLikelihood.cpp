/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/GaussianLikelihood.h>
#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/mathematics/Math.h>
#include <shogun/base/Parameter.h>

using namespace shogun;
using namespace Eigen;

CGaussianLikelihood::CGaussianLikelihood() : CLikelihoodModel()
{
	init();
}

CGaussianLikelihood::CGaussianLikelihood(float64_t sigma) : CLikelihoodModel()
{
	init();
	m_sigma=sigma;
}

void CGaussianLikelihood::init()
{
	m_sigma = 1;
	SG_ADD(&m_sigma, "sigma", "Observation Noise.", MS_AVAILABLE);
}

CGaussianLikelihood::~CGaussianLikelihood()
{
}

CGaussianLikelihood* CGaussianLikelihood::obtain_from_generic(CLikelihoodModel* likelihood)
{
	ASSERT(likelihood!=NULL);

	if (likelihood->get_model_type()!=LT_GAUSSIAN)
		SG_SERROR("CGaussianLikelihood::obtain_from_generic(): provided likelihood is "
			"not of type CGaussianLikelihood!\n")

	SG_REF(likelihood);
	return (CGaussianLikelihood*)likelihood;
}

SGVector<float64_t> CGaussianLikelihood::evaluate_means(
		SGVector<float64_t>& means)
{
	return SGVector<float64_t>(means);
}

SGVector<float64_t> CGaussianLikelihood::evaluate_variances(
		SGVector<float64_t>& vars)
{
	SGVector<float64_t> result(vars);

	for (index_t i=0; i<result.vlen; i++)
		result[i]+=CMath::sq(m_sigma);

	return result;
}

float64_t CGaussianLikelihood::get_log_probability_f(CRegressionLabels* labels,
		SGVector<float64_t> m_function)
{
	Map<VectorXd> eigen_function(m_function.vector, m_function.vlen);

	SGVector<float64_t> result(m_function.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=labels->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute log probability: lp=-(y-f).^2./sigma^2/2-log(2*pi*sigma^2)/2
	eigen_result=eigen_y-eigen_function;
	eigen_result=-eigen_result.cwiseProduct(eigen_result)/(2*CMath::sq(m_sigma))-
		VectorXd::Ones(result.vlen)*log(2*CMath::PI*CMath::sq(m_sigma))/2.0;

	return eigen_result.sum();
}

SGVector<float64_t> CGaussianLikelihood::get_log_probability_derivative_f(
		CRegressionLabels* labels, SGVector<float64_t> m_function, index_t j)
{
	Map<VectorXd> eigen_function(m_function.vector, m_function.vlen);

	SGVector<float64_t> result(m_function.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=labels->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// set result=y-f
	eigen_result=eigen_y-eigen_function;

	// compute derivatives of log probability wrt f
	if (j == 1)
		eigen_result/=CMath::sq(m_sigma);

	else if (j == 2)
		eigen_result=-VectorXd::Ones(result.vlen)/CMath::sq(m_sigma);

	else if (j == 3)
		eigen_result=VectorXd::Zero(result.vlen);

	else
		SG_ERROR("Invalid Index for Likelihood Derivative\n")

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_first_derivative(CRegressionLabels* labels,
		TParameter* param,  CSGObject* obj, SGVector<float64_t> m_function)
{
	Map<VectorXd> eigen_function(m_function.vector, m_function.vlen);

	SGVector<float64_t> result(m_function.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	if (strcmp(param->m_name, "sigma") || obj != this)
	{
		result[0] = CMath::INFTY;
		return result;
	}

	SGVector<float64_t> y=labels->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of log probability wrt sigma:
	// lp_dsigma=(y-f).^2/sigma^2-1
	eigen_result=eigen_y-eigen_function;
	eigen_result=eigen_result.cwiseProduct(eigen_result)/CMath::sq(m_sigma);
	eigen_result-=VectorXd::Ones(result.vlen);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_second_derivative(CRegressionLabels* labels,
		TParameter* param,  CSGObject* obj, SGVector<float64_t> m_function)
{
	Map<VectorXd> eigen_function(m_function.vector, m_function.vlen);

	SGVector<float64_t> result(m_function.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	if (strcmp(param->m_name, "sigma") || obj != this)
	{
		result[0] = CMath::INFTY;
		return result;
	}

	SGVector<float64_t> y=labels->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of the first derivative of log probability wrt sigma:
	// dlp_dsigma=2*(f-y)/sigma^2
	eigen_result=2*(eigen_function-eigen_y)/CMath::sq(m_sigma);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_third_derivative(CRegressionLabels* labels,
		TParameter* param, CSGObject* obj, SGVector<float64_t> m_function)
{
	Map<VectorXd> eigen_function(m_function.vector, m_function.vlen);

	SGVector<float64_t> result(m_function.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	if (strcmp(param->m_name, "sigma") || obj != this)
	{
		result[0] = CMath::INFTY;
		return result;
	}

	// compute derivative of the second derivative of log probability wrt sigma:
	// d2lp_dsigma=1/sigma^2
	eigen_result=2*VectorXd::Ones(result.vlen)/CMath::sq(m_sigma);

	return result;
}

#endif //HAVE_EIGEN3
