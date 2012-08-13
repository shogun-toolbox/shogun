/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/GaussianLikelihood.h>
#include <shogun/modelselection/ParameterCombination.h>

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
		Eigen::VectorXd function)
{
	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = labels->get_labels()[i] - function[i];

	result = result.cwiseProduct(result);

	result /= -2*m_sigma*m_sigma;

	for (index_t i = 0; i < function.rows(); i++)
		result[i] -= log(2*CMath::PI*m_sigma*m_sigma)/2.0;

	return result.sum();
}

VectorXd CGaussianLikelihood::get_log_probability_derivative_f(
		CRegressionLabels* labels, Eigen::VectorXd function, index_t j)
{
	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = labels->get_labels()[i] - function[i];

	if (j == 1)
		return result/(m_sigma*m_sigma);

	else if (j == 2)
		return -VectorXd::Ones(result.rows())/(m_sigma*m_sigma);

	else if (j == 3)
		return VectorXd::Zero(result.rows());

	else
		SG_ERROR("Invalid Index for Likelihood Derivative\n");
}

VectorXd CGaussianLikelihood::get_first_derivative(CRegressionLabels* labels,
		TParameter* param,  CSGObject* obj, Eigen::VectorXd function)
{
	VectorXd result(function.rows());

	if (strcmp(param->m_name, "sigma") || obj != this)
	{
		result(0) = CMath::INFTY;
		return result;
	}

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = labels->get_labels()[i] - function[i];

	result = result.cwiseProduct(result);

	result /= m_sigma*m_sigma;

	for (index_t i = 0; i < function.rows(); i++)
		result[i] -= 1;

	return result;
}

VectorXd CGaussianLikelihood::get_second_derivative(CRegressionLabels* labels,
		TParameter* param, CSGObject* obj, Eigen::VectorXd function)
{

	if (strcmp(param->m_name, "sigma") || obj != this)
	{
		VectorXd result(function.rows());
		result(0) = CMath::INFTY;
		return result;
	}

	return 2*VectorXd::Ones(function.rows())/(m_sigma*m_sigma);
}


