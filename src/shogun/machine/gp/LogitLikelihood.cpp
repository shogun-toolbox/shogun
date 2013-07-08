/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/machine/gp/LogitLikelihood.h>

#ifdef HAVE_EIGEN3

#include <shogun/labels/BinaryLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CLogitLikelihood::CLogitLikelihood() : CLikelihoodModel()
{
}

CLogitLikelihood::~CLogitLikelihood()
{
}

SGVector<float64_t> CLogitLikelihood::evaluate_log_probabilities(SGVector<float64_t> mu,
		SGVector<float64_t> s2, CLabels* lab)
{
	SG_NOTIMPLEMENTED

	return SGVector<float64_t>();
}

SGVector<float64_t> CLogitLikelihood::evaluate_means(SGVector<float64_t> mu,
		SGVector<float64_t> s2, CLabels* lab)
{
	SG_NOTIMPLEMENTED

	return SGVector<float64_t>();
}

SGVector<float64_t> CLogitLikelihood::evaluate_variances(SGVector<float64_t> mu,
		SGVector<float64_t> s2, CLabels* lab)
{
	SG_NOTIMPLEMENTED

	return SGVector<float64_t>();
}

SGVector<float64_t> CLogitLikelihood::get_log_probability_f(CLabels* lab,
		SGVector<float64_t> func)
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_BINARY,
		"Labels must be type of CBinaryLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match " \
		"length of the function vector\n")

	SGVector<float64_t> y=((CBinaryLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute log probability: -log(1+exp(-f.*y))
	eigen_r=-(1.0+(-eigen_y.array()*eigen_f.array()).exp()).log();

	return r;
}

SGVector<float64_t> CLogitLikelihood::get_log_probability_derivative_f(
	CLabels* lab, SGVector<float64_t> func, index_t i)
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_BINARY,
		"Labels must be type of CBinaryLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match " \
		"length of the function vector\n")
	REQUIRE(i>=1 && i<=3, "Index for derivative should be 1, 2 or 3\n")

	SGVector<float64_t> y=((CBinaryLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> r(func.vlen);
	Map<VectorXd> eigen_r(r.vector, r.vlen);

	// compute s(f)=1./(1+exp(-f))
	VectorXd eigen_s=1.0/(1.0+(-eigen_f).array().exp());

	// compute derivatives of log probability wrt f
	if (i == 1)
	{
		// compute the first derivative: dlp=(y+1)/2-s(f)
		eigen_r=(eigen_y.array()+1.0)/2.0-eigen_s.array();
	}
	else if (i == 2)
	{
		// compute the second derivative: d2lp=-s(f).*(1-s(f))
		eigen_r=-eigen_s.array()*(1.0-eigen_s.array());
	}
	else if (i == 3)
	{
		// compute the third derivative: d2lp=-s(f).*(1-s(f)).*(1-2*s(f))
		eigen_r=-eigen_s.array()*(1.0-eigen_s.array())*(1.0-2*eigen_s.array());
	}
	else
	{
		SG_ERROR("Invalid index for derivative\n")
	}

	return r;
}

SGVector<float64_t> CLogitLikelihood::get_first_derivative(CLabels* lab,
	TParameter* param,  CSGObject* obj, SGVector<float64_t> f)
{
	return SGVector<float64_t>();
}

SGVector<float64_t> CLogitLikelihood::get_second_derivative(CLabels* lab,
	TParameter* param,  CSGObject* obj, SGVector<float64_t> func)
{
	return SGVector<float64_t>();
}

SGVector<float64_t> CLogitLikelihood::get_third_derivative(CLabels* lab,
	TParameter* param, CSGObject* obj, SGVector<float64_t> func)
{
	return SGVector<float64_t>();
}

#endif /* HAVE_EIGEN3 */
