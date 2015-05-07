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
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

CGaussianLikelihood::CGaussianLikelihood() : CLikelihoodModel()
{
	init();
}

CGaussianLikelihood::CGaussianLikelihood(float64_t sigma) : CLikelihoodModel()
{
	init();
	set_sigma(sigma);
}

void CGaussianLikelihood::init()
{
	m_log_sigma=0.0;
	SG_ADD(&m_log_sigma, "log_sigma", "Observation noise in log domain", MS_AVAILABLE, GRADIENT_AVAILABLE);
}

CGaussianLikelihood::~CGaussianLikelihood()
{
}

CGaussianLikelihood* CGaussianLikelihood::obtain_from_generic(
		CLikelihoodModel* lik)
{
	ASSERT(lik!=NULL);

	if (lik->get_model_type()!=LT_GAUSSIAN)
		SG_SERROR("Provided likelihood is not of type CGaussianLikelihood!\n")

	SG_REF(lik);
	return (CGaussianLikelihood*)lik;
}

SGVector<float64_t> CGaussianLikelihood::get_predictive_means(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	return SGVector<float64_t>(mu);
}

SGVector<float64_t> CGaussianLikelihood::get_predictive_variances(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels* lab) const
{
	SGVector<float64_t> result(s2);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	eigen_result=eigen_result.array()+CMath::exp(m_log_sigma*2.0);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_log_probability_f(const CLabels* lab,
		SGVector<float64_t> func) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute log probability: lp=-(y-f).^2./sigma^2/2-log(2*pi*sigma^2)/2
	eigen_result=eigen_y-eigen_f;
	eigen_result=-eigen_result.cwiseProduct(eigen_result)/(2.0*CMath::exp(m_log_sigma*2.0))-
		VectorXd::Ones(result.vlen)*log(2.0*CMath::PI*CMath::exp(m_log_sigma*2.0))/2.0;

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_log_probability_derivative_f(
		const CLabels* lab, SGVector<float64_t> func, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")
	REQUIRE(i>=1 && i<=3, "Index for derivative should be 1, 2 or 3\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// set result=y-f
	eigen_result=eigen_y-eigen_f;

	// compute derivatives of log probability wrt f
	if (i == 1)
		eigen_result/=CMath::exp(m_log_sigma*2.0);
	else if (i == 2)
		eigen_result=-VectorXd::Ones(result.vlen)/CMath::exp(m_log_sigma*2.0);
	else if (i == 3)
		eigen_result=VectorXd::Zero(result.vlen);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_first_derivative(const CLabels* lab,
		SGVector<float64_t> func, const TParameter* param) const
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	if (strcmp(param->m_name, "log_sigma"))
		return SGVector<float64_t>();

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of log probability wrt log_sigma:
	// dlp_dlogsigma
	// lp_dsigma=(y-f).^2/sigma^2-1
	eigen_result=eigen_y-eigen_f;
	eigen_result=eigen_result.cwiseProduct(eigen_result)/CMath::exp(m_log_sigma*2.0);
	eigen_result-=VectorXd::Ones(result.vlen);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_second_derivative(const CLabels* lab,
		SGVector<float64_t> func, const TParameter* param) const
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	if (strcmp(param->m_name, "log_sigma"))
		return SGVector<float64_t>();

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of (the first log_sigma derivative of log probability) wrt f:
	// d2lp_dlogsigma_df == d2lp_df_dlogsigma
	// dlp_dsigma=2*(f-y)/sigma^2
	eigen_result=2.0*(eigen_f-eigen_y)/CMath::exp(m_log_sigma*2.0);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_third_derivative(const CLabels* lab,
		SGVector<float64_t> func, const TParameter* param) const
{
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")
	REQUIRE(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector\n")

	if (strcmp(param->m_name, "log_sigma"))
		return SGVector<float64_t>();

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	// compute derivative of (the derivative of the first log_sigma derivative of log probability) wrt f:
	// d3lp_dlogsigma_df_df == d3lp_df_df_dlogsigma
	// d2lp_dsigma=2/sigma^2
	eigen_result=2.0*VectorXd::Ones(result.vlen)/CMath::exp(m_log_sigma*2.0);

	return result;
}

SGVector<float64_t> CGaussianLikelihood::get_log_zeroth_moments(
		SGVector<float64_t> mu, SGVector<float64_t> s2, const CLabels *lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means (%d), length of the vector of "
				"variances (%d) and number of labels (%d) should be the same\n",
				mu.vlen, s2.vlen, lab->get_num_labels())
		REQUIRE(lab->get_label_type()==LT_REGRESSION,
				"Labels must be type of CRegressionLabels\n")

		y=((CRegressionLabels*)lab)->get_labels();
	}
	else
	{
		REQUIRE(mu.vlen==s2.vlen, "Length of the vector of means (%d) and "
				"length of the vector of variances (%d) should be the same\n",
				mu.vlen, s2.vlen)

		y=SGVector<float64_t>(mu.vlen);
		y.set_const(1.0);
	}

	// create eigen representation of y, mu and s2
	Map<VectorXd> eigen_mu(mu.vector, mu.vlen);
	Map<VectorXd> eigen_s2(s2.vector, s2.vlen);
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	SGVector<float64_t> result(mu.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	// compule lZ=-(y-mu).^2./(sn2+s2)/2-log(2*pi*(sn2+s2))/2
	eigen_s2=eigen_s2.array()+CMath::exp(m_log_sigma*2.0);
	eigen_result=-(eigen_y-eigen_mu).array().square()/(2.0*eigen_s2.array())-
		(2.0*CMath::PI*eigen_s2.array()).log()/2.0;

	return result;
}

float64_t CGaussianLikelihood::get_first_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels *lab, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means (%d), length of the vector of "
			"variances (%d) and number of labels (%d) should be the same\n",
			mu.vlen, s2.vlen, lab->get_num_labels())
	REQUIRE(i>=0 && i<=mu.vlen, "Index (%d) out of bounds!\n", i)
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")

	SGVector<float64_t> y=((CRegressionLabels*)lab)->get_labels();

	// compute 1st moment
	float64_t Ex=mu[i]+s2[i]*(y[i]-mu[i])/(CMath::exp(m_log_sigma*2.0)+s2[i]);

	return Ex;
}

float64_t CGaussianLikelihood::get_second_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, const CLabels *lab, index_t i) const
{
	// check the parameters
	REQUIRE(lab, "Labels are required (lab should not be NULL)\n")
	REQUIRE((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means (%d), length of the vector of "
			"variances (%d) and number of labels (%d) should be the same\n",
			mu.vlen, s2.vlen, lab->get_num_labels())
	REQUIRE(i>=0 && i<=mu.vlen, "Index (%d) out of bounds!\n", i)
	REQUIRE(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of CRegressionLabels\n")

	// compute 2nd moment
	float64_t Var=s2[i]-CMath::sq(s2[i])/(CMath::exp(m_log_sigma*2.0)+s2[i]);

	return Var;
}

#endif /* HAVE_EIGEN3 */
