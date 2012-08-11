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
#include <iostream>

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
	SG_ADD(&m_df, "df", "Degrees of Freedom.", MS_AVAILABLE);
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
	float64_t temp = lgamma(m_df/2.0+0.5) - lgamma(m_df/2.0) - log(m_df*CMath::PI*m_sigma*m_sigma)/2.0;
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

VectorXd CStudentsTLikelihood::get_log_probability_derivative_f(CRegressionLabels* labels, Eigen::VectorXd function, index_t j)
{
    VectorXd result(function.rows());

    for (index_t i = 0; i < function.rows(); i++)
    	result[i] = (labels->get_labels()[i] - function[i]);

    VectorXd result_squared = result.cwiseProduct(result);

    VectorXd a(function.rows());
    VectorXd b(function.rows());
    VectorXd c(function.rows());
    VectorXd d(function.rows());


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
    	return (m_df+1)*2*result.cwiseProduct(c).cwiseQuotient(d.cwiseProduct(a));
}

VectorXd CStudentsTLikelihood::get_first_derivative(CRegressionLabels* labels, TParameter* param,  CSGObject* obj, Eigen::VectorXd function)
{
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

	//	-.5/x +.5dlgamma(.5x+.5)-dloggamma(x) - .5log(r/(sx)+1) + (.5*r(x+1))/(x(r+s*x));
 /*   for (index_t i = 0; i < function.rows(); i++)
    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

    a *= m_df;

    a = result_squared.cwiseQuotient(a);

    a *= (m_df+1)/2.0;


    for (index_t i = 0; i < function.rows(); i++)
     	a[i] = a[i] - 1/(2*m_df) + .5*dlgamma(.5*m_df+.5)-dlgamma(m_df)-.5*log(result_squared[i]/(m_sigma*m_sigma*m_df) + 1.0);

    result = a;

    return result;*/

		for (index_t i = 0; i < function.rows(); i++)
		    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

		a = result_squared.cwiseQuotient(a);

		a *= (m_df/2.0+.5);

		for (index_t i = 0; i < function.rows(); i++)
			a[i] += m_df*( dlgamma(m_df/2.0+1/2.0)-dlgamma(m_df/2.0) )/2.0 - 1/2.0
			-m_df*log(1+result_squared[i]/(m_df*m_sigma*m_sigma))/2.0;

        a *= (1-1/m_df);

        return a/(m_df-1);
	}

	if (strcmp(param->m_name, "sigma") == 0 && obj == this)
	{
	//	-1/x + (n+1)*r/(x*(nx^2+r))

/*	for (index_t i = 0; i < function.rows(); i++)
	    a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

	a *= m_sigma;

    a = result_squared.cwiseQuotient(a);

    a *= (m_df+1);

    for (index_t i = 0; i < function.rows(); i++)
     	a[i] = a[i] - 1/(m_sigma);

    result = a;*/

		for (index_t i = 0; i < function.rows(); i++)
			    a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

	    a = (m_df+1)*result_squared.cwiseQuotient(a);

		for (index_t i = 0; i < function.rows(); i++)
			    a[i] -= 1.0;

    return a/(m_sigma);
}
	else
	{
		result(0) = CMath::INFTY;
		return result;
	}

}

VectorXd CStudentsTLikelihood::get_second_derivative(CRegressionLabels* labels, TParameter* param, CSGObject* obj, Eigen::VectorXd function)
{

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
	//	(r^2-3rs(x+1)+s^2x)/(r+sx)^3

  /*  for (index_t i = 0; i < function.rows(); i++)
    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

    b = result_squared.cwiseProduct(result_squared);

    b = b - 3*result_squared*m_sigma*m_sigma*(m_df+1);

    for (index_t i = 0; i < function.rows(); i++)
     	b[i] = b[i] + pow(m_sigma, 4)*m_df;

    c = a.cwiseProduct(a);

    c = c.cwiseProduct(a);

    result = b.cwiseQuotient(c);

    return result;*/

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

		    return result/(m_df-1);
	}

	if (strcmp(param->m_name, "sigma") == 0 && obj == this)
	{
		//2*n(n+1)x(nx^2-3r)/((nx^2+r)^3)
	/*    for (index_t i = 0; i < function.rows(); i++)
	    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

	    c = a.cwiseProduct(a);

	    c = c.cwiseProduct(a);

    for (index_t i = 0; i < function.rows(); i++)
     	d[i] =  m_df*pow(m_sigma, 4)-3*result_squared[i];

    d *= m_sigma*(m_df+1)*m_df*2;

    result = d.cwiseQuotient(c);

    return result;*/

		 for (index_t i = 0; i < function.rows(); i++)
			    	a[i] = result_squared[i] + m_df*m_sigma*m_sigma;

			    c = a.cwiseProduct(a);

			    c = c.cwiseProduct(a);

	     for (index_t i = 0; i < function.rows(); i++)
			     b[i] = m_df*m_sigma*m_sigma - 3*result_squared[i];

	     b *= m_sigma*m_sigma*m_df*2.0*(m_df+1);

	     return b.cwiseQuotient(c)/m_sigma;

	}

	else
	{
		result(0) = CMath::INFTY;
		return result;
	}

}



