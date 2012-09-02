/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */


#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/mathematics/eigen3.h>

#include <shogun/base/Parameter.h>

using namespace shogun;
using namespace Eigen;

CGaussianLikelihood::CGaussianLikelihood() : CLikelihoodModel()
{
	init();
}

void CGaussianLikelihood::init()
{
	m_sigma = 0.01;
	SG_ADD(&m_sigma, "sigma", "Observation Noise.", MS_AVAILABLE);
}

CGaussianLikelihood::~CGaussianLikelihood()
{
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

	for (index_t i = 0; i < result.vlen; i++)
		result[i] += (m_sigma*m_sigma);

	return result;
}

float64_t CGaussianLikelihood::get_log_probability_f(CRegressionLabels* labels,
		SGVector<float64_t> m_function)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);

	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = labels->get_labels()[i] - function[i];

	result = result.cwiseProduct(result);

	result /= -2*m_sigma*m_sigma;

	for (index_t i = 0; i < function.rows(); i++)
		result[i] -= log(2*CMath::PI*m_sigma*m_sigma)/2.0;

	return result.sum();
}

SGVector<float64_t> CGaussianLikelihood::get_log_probability_derivative_f(
		CRegressionLabels* labels, SGVector<float64_t> m_function, index_t j)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);
	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = labels->get_labels()[i] - function[i];

	if (j == 1)
		result = result/(m_sigma*m_sigma);

	else if (j == 2)
		result = -VectorXd::Ones(result.rows())/(m_sigma*m_sigma);

	else if (j == 3)
		result = VectorXd::Zero(result.rows());

	else
		SG_ERROR("Invalid Index for Likelihood Derivative\n");

	SGVector<float64_t> sgresult(result.rows());
	
	for (index_t i = 0; i < result.rows(); i++)
		sgresult[i] = result[i];

	return sgresult;
}

SGVector<float64_t> CGaussianLikelihood::get_first_derivative(CRegressionLabels* labels,
		TParameter* param,  CSGObject* obj, SGVector<float64_t> m_function)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);

	VectorXd result(function.rows());

	SGVector<float64_t> sgresult(result.rows());

	if (strcmp(param->m_name, "sigma") || obj != this)
	{
		sgresult[0] = CMath::INFTY;
		return sgresult;
	}

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = labels->get_labels()[i] - function[i];

	result = result.cwiseProduct(result);

	result /= m_sigma*m_sigma;

	for (index_t i = 0; i < function.rows(); i++)
		result[i] -= 1;
	
	for (index_t i = 0; i < result.rows(); i++)
		sgresult[i] = result[i];

	return sgresult;
}

SGVector<float64_t> CGaussianLikelihood::get_second_derivative(CRegressionLabels* labels,
		TParameter* param, CSGObject* obj, SGVector<float64_t> m_function)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);
	VectorXd result(function.rows());

	SGVector<float64_t> sgresult(result.rows());

	if (strcmp(param->m_name, "sigma") || obj != this)
	{
		sgresult[0] = CMath::INFTY;
		return sgresult;
	}

	result = 2*VectorXd::Ones(function.rows())/(m_sigma*m_sigma);

	for (index_t i = 0; i < result.rows(); i++)
		sgresult[i] = result[i];

	return sgresult;
}

#endif //HAVE_EIGEN3


