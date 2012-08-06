/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2012 Jacob Walker
 */

#include <shogun/regression/gp/StudentsTLikelihood.h>
#include <shogun/modelselection/ParameterCombination.h>


#include <shogun/base/Parameter.h>

using namespace shogun;
using namespace Eigen;

CStudentsTLikelihood::CStudentsTLikelihood() : CLikelihoodModel()
{
	init();
}

void CStudentsTLikelihood::init()
{
	m_sigma = 0.01;
	m_df = 3;
	SG_ADD(&m_sigma, "sigma", "Observation Noise.", MS_AVAILABLE);
	SG_ADD(&m_df, "nu", "Degrees of Freedom.", MS_NOT_AVAILABLE);
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

float64_t CStudentsTLikelihood::get_log_probability_f(CRegressionLabels* labels, Eigen::VectorXd function)
{
	float64_t temp = lgamma(m_df/2+1/2) - lgamma(m_df/2) - log(m_df*CMath::PI*m_sigma*m_sigma)/2.0;

    VectorXd result(function.rows());

    for (index_t i = 0; i < function.rows(); i++)
    	result[i] = labels->get_labels()[i] - function[i];

    result = result.cwiseProduct(result);

    result /= m_df*m_sigma*m_sigma;

    for (index_t i = 0; i < function.rows(); i++)
    {
    	result[i] -= (m_df+1)*log(1.0+float64_t(result[i]))/2.0;
    	result[i] += temp;
    }

    return result.sum();
}

VectorXd CStudentsTLikelihood::get_log_probability_derivative_f(CRegressionLabels* labels, Eigen::VectorXd function, index_t j)
{
    VectorXd result(function.rows());

    for (index_t i = 0; i < function.rows(); i++)
    	result[i] = (labels->get_labels()[i] - function[i]);

    VectorXd result_squared = result.cwiseProduct(result);

    VectorXd a;
    VectorXd b;
    VectorXd c;
    VectorXd d;


    for (index_t i = 0; i < function.rows(); i++)
    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

    if (j == 1)
    	return (m_df+1)*result.cwiseQuotient(a);

    for (index_t i = 0; i < function.rows(); i++)
    	b[i] = result_squared[i] - m_df*m_sigma*m_sigma;

    if (j == 2)
    	return (m_df+1)*b.cwiseQuotient(a.cwiseProduct(a));

    for (index_t i = 0; i < function.rows(); i++)
        c[i] = result_squared[i] - 3*m_df*m_sigma*m_sigma;

    d = a.cwiseProduct(a);

    if (j == 3)
    	return (m_df+1)*2*result.cwiseProduct(c).cwiseQuotient(a.cwiseProduct(a));
}

VectorXd CStudentsTLikelihood::get_first_derivative(CRegressionLabels* labels, TParameter* param,  CSGObject* obj, Eigen::VectorXd function)
{
    VectorXd result(function.rows());

    for (index_t i = 0; i < function.rows(); i++)
    	result[i] = (labels->get_labels()[i] - function[i]);

    VectorXd result_squared = result.cwiseProduct(result);

    VectorXd a;
    VectorXd b;
    VectorXd c;
    VectorXd d;


    for (index_t i = 0; i < function.rows(); i++)
    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

    for (index_t i = 0; i < function.rows(); i++)
     	b[i] = log(1.0+float64_t(result_squared[i])/m_df*m_sigma*m_sigma);

    result = (m_df/2.0+0.5)*result_squared.cwiseQuotient(a);

    result = result - m_df*b/2.0;

    for (index_t i = 0; i < function.rows(); i++)
     	result[i] += m_df*(lgamma(m_df/2+1/2)-lgamma(m_df/2) )/2.0 - 0.5;

    result = (1-1/m_df)*result;

    return result;

    result = (m_df+1)*result_squared.cwiseProduct(a);

    for (index_t i = 0; i < function.rows(); i++)
     	result[i] - 1.0;

    return result;
}

VectorXd CStudentsTLikelihood::get_second_derivative(CRegressionLabels* labels, TParameter* param, CSGObject* obj, Eigen::VectorXd function)
{

    VectorXd result(function.rows());

    for (index_t i = 0; i < function.rows(); i++)
    	result[i] = (labels->get_labels()[i] - function[i]);

    VectorXd result_squared = result.cwiseProduct(result);

    VectorXd a;
    VectorXd b;
    VectorXd c;
    VectorXd d;


    for (index_t i = 0; i < function.rows(); i++)
    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

    for (index_t i = 0; i < function.rows(); i++)
     	b[i] = result_squared[i] - 3*(1+m_df)*m_sigma*m_sigma;

    c = a.cwiseProduct(a);

    result = result_squared.cwiseProduct(b);

    for (index_t i = 0; i < function.rows(); i++)
     	result[i] = m_df*(result[i]+m_df*pow(m_sigma,4));

    result = result.cwiseQuotient(c.cwiseProduct(a));

    result = (1-1/m_df)*result;

    return result;

    for (index_t i = 0; i < function.rows(); i++)
     	d[i] =  m_df*m_sigma*m_sigma-3*result_squared[i];

    result = (m_df+1)*2*m_df*m_sigma*m_sigma*d.cwiseQuotient(c.cwiseProduct(a));

    return result;
}



