/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 *
 * Adapted from the GPML toolbox, specifically likT.m
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 *
 */

#ifdef HAVE_EIGEN3

#include <shogun/regression/gp/StudentsTLikelihood.h>
#ifdef HAVE_EIGEN3
#include <shogun/modelselection/ParameterCombination.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CStudentsTLikelihood::CStudentsTLikelihood() : CLikelihoodModel()
{
	init();
}

void CStudentsTLikelihood::init()
{
	m_df = 3.0;
	m_sigma = 0.01;
	SG_ADD(&m_sigma, "sigma", "Observation Noise.", MS_AVAILABLE);
}

CStudentsTLikelihood::~CStudentsTLikelihood()
{
}


SGVector<float64_t> CStudentsTLikelihood::evaluate_means(
		SGVector<float64_t>& means)
{
	return SGVector<float64_t>(means);
}

SGVector<float64_t> CStudentsTLikelihood::evaluate_variances(
		SGVector<float64_t>& vars)
{
	SGVector<float64_t> result(vars);

	for (index_t i = 0; i < result.vlen; i++)
	{
		if (m_df < 2)
			result[i] = CMath::INFTY;
		else
			result[i] += m_df*(m_sigma*m_sigma)/float64_t(m_df-2);

	}

	return result;
}

float64_t CStudentsTLikelihood::get_log_probability_f(
		CRegressionLabels* labels, SGVector<float64_t> m_function)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);

	float64_t temp = lgamma(m_df/2.0+0.5) -
			lgamma(m_df/2.0) - log(m_df*CMath::PI*m_sigma*m_sigma)/2.0;

	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = labels->get_labels()[i] - function[i];

	result = result.cwiseProduct(result);

	result /= m_df*m_sigma*m_sigma;

	for (index_t i = 0; i < function.rows(); i++)
	{
		result[i] = -(m_df+1)*log(1.0+float64_t(result[i]))/2.0;

		result[i] += temp;
	}

	return result.sum();
}

SGVector<float64_t>  CStudentsTLikelihood::get_log_probability_derivative_f(
		CRegressionLabels* labels, SGVector<float64_t> m_function, index_t j)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);
	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = (labels->get_labels()[i] - function[i]);

	VectorXd result_squared = result.cwiseProduct(result);

	VectorXd a(function.rows());
	VectorXd b(function.rows());
	VectorXd c(function.rows());
	VectorXd d(function.rows());

	SGVector<float64_t> sgresult(result.rows());

	for (index_t i = 0; i < function.rows(); i++)
		a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

	if (j == 1)
	{
		result = (m_df+1)*result.cwiseQuotient(a);
		for (index_t i = 0; i < result.rows(); i++)
			sgresult[i] = result[i];
		return sgresult;
	}

	for (index_t i = 0; i < function.rows(); i++)
		b[i] = result_squared[i] - m_df*m_sigma*m_sigma;

	if (j == 2)
	{
		result = (m_df+1)*b.cwiseQuotient(a.cwiseProduct(a));
		for (index_t i = 0; i < result.rows(); i++)
			sgresult[i] = result[i];
		return sgresult;
	}

	for (index_t i = 0; i < function.rows(); i++)
		c[i] = result_squared[i] - 3*m_df*m_sigma*m_sigma;

	d = a.cwiseProduct(a);

	if (j == 3)
	{
		result = (m_df+1)*2*result.cwiseProduct(c).cwiseQuotient(d.cwiseProduct(a));
		for (index_t i = 0; i < result.rows(); i++)
			sgresult[i] = result[i];
		return sgresult;

	}

	else
	{
		SG_ERROR("Invalid index for derivative\n");
		return sgresult;
	}
}

//Taken in log space then converted back to direct derivative
SGVector<float64_t> CStudentsTLikelihood::get_first_derivative(
		CRegressionLabels* labels, TParameter* param,
		CSGObject* obj, SGVector<float64_t> m_function)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);

	SGVector<float64_t> sgresult(function.rows());

	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = (labels->get_labels()[i] - function[i]);


	VectorXd result_squared = result.cwiseProduct(result);

	VectorXd a(function.rows());
	VectorXd b(function.rows());
	VectorXd c(function.rows());
	VectorXd d(function.rows());


	if (strcmp(param->m_name, "df") == 0 && obj == this)
	{
		for (index_t i = 0; i < function.rows(); i++)
			a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

		a = result_squared.cwiseQuotient(a);

		a *= (m_df/2.0+.5);

		for (index_t i = 0; i < function.rows(); i++)
			a[i] += m_df*( CStatistics::dlgamma(m_df/2.0+1/2.0)-
			CStatistics::dlgamma(m_df/2.0) )/2.0 - 1/2.0
			-m_df*log(1+result_squared[i]/(m_df*m_sigma*m_sigma))/2.0;

		a *= (1-1/m_df);

		result = a/(m_df-1);

		for (index_t i = 0; i < result.rows(); i++)
			sgresult[i] = result[i];
		return sgresult;
	}


	if (strcmp(param->m_name, "sigma") == 0 && obj == this)
	{
		for (index_t i = 0; i < function.rows(); i++)
			a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

		a = (m_df+1)*result_squared.cwiseQuotient(a);

		for (index_t i = 0; i < function.rows(); i++)
			a[i] -= 1.0;

		result = a/(m_sigma);

		for (index_t i = 0; i < result.rows(); i++)
			sgresult[i] = result[i];
 
		return sgresult;
	}


	sgresult[0] = CMath::INFTY;

	return sgresult;
}

//Taken in log space then converted back to direct derivative
SGVector<float64_t> CStudentsTLikelihood::get_second_derivative(
		CRegressionLabels* labels, TParameter* param,
		CSGObject* obj, SGVector<float64_t> m_function)
{
	Map<VectorXd> function(m_function.vector, m_function.vlen);

	SGVector<float64_t> sgresult(function.rows());
	VectorXd result(function.rows());

	for (index_t i = 0; i < function.rows(); i++)
		result[i] = (labels->get_labels()[i] - function[i]);

	VectorXd result_squared = result.cwiseProduct(result);

	VectorXd a(function.rows());
	VectorXd b(function.rows());
	VectorXd c(function.rows());
	VectorXd d(function.rows());

	if (strcmp(param->m_name, "df") == 0 && obj == this)
	{
		for (index_t i = 0; i < function.rows(); i++)
			a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

		b = result_squared.cwiseProduct(result_squared);

		b = b - 3*result_squared*m_sigma*m_sigma*(m_df+1);

		for (index_t i = 0; i < function.rows(); i++)
			b[i] = result_squared[i] - 3*m_sigma*m_sigma*(1+m_df);

		b = b.cwiseProduct(result_squared);

		for (index_t i = 0; i < function.rows(); i++)
			b[i] = b[i] + pow(m_sigma, 4)*m_df;

		b *= m_df;

		c = a.cwiseProduct(a);

		c = c.cwiseProduct(a);

		result = b.cwiseQuotient(c);

		result = result/(m_df-1);
		
		for (index_t i = 0; i < result.rows(); i++)
			sgresult[i] = result[i];
		return sgresult;
	}

	if (strcmp(param->m_name, "sigma") == 0 && obj == this)
	{
		for (index_t i = 0; i < function.rows(); i++)
			a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

		c = a.cwiseProduct(a);

		c = c.cwiseProduct(a);

		for (index_t i = 0; i < function.rows(); i++)
			b[i] = m_df*m_sigma*m_sigma - 3*result_squared[i];

		b *= m_sigma*m_sigma*m_df*2.0*(m_df+1);

		result = b.cwiseQuotient(c)/m_sigma;

		for (index_t i = 0; i < result.rows(); i++)
			sgresult[i] = result[i];

		return sgresult;

	}


	sgresult[0] = CMath::INFTY;
	return sgresult;

}

#endif //HAVE_EIGEN3



