/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2012 Jacob Walker
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */
#include <shogun/machine/gp/GaussianLikelihood.h>


#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/eigen3.h>

using namespace shogun;
using namespace Eigen;

GaussianLikelihood::GaussianLikelihood() : LikelihoodModel()
{
	init();
}

GaussianLikelihood::GaussianLikelihood(float64_t sigma) : LikelihoodModel()
{
	init();
	set_sigma(sigma);
}

void GaussianLikelihood::init()
{
	m_log_sigma=0.0;
	SG_ADD(&m_log_sigma, "log_sigma", "Observation noise in log domain", ParameterProperties::HYPER | ParameterProperties::GRADIENT);
}

GaussianLikelihood::~GaussianLikelihood()
{
}

std::shared_ptr<GaussianLikelihood> GaussianLikelihood::obtain_from_generic(
		const std::shared_ptr<LikelihoodModel>& lik)
{
	ASSERT(lik!=NULL);

	if (lik->get_model_type()!=LT_GAUSSIAN)
		error("Provided likelihood is not of type GaussianLikelihood!");

	return lik->as<GaussianLikelihood>();
}

SGVector<float64_t> GaussianLikelihood::get_predictive_means(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	return SGVector<float64_t>(mu);
}

SGVector<float64_t> GaussianLikelihood::get_predictive_variances(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels> lab) const
{
	SGVector<float64_t> result(s2);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	eigen_result = eigen_result.array() + std::exp(m_log_sigma * 2.0);

	return result;
}

SGVector<float64_t> GaussianLikelihood::get_log_probability_f(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute log probability: lp=-(y-f).^2./sigma^2/2-log(2*pi*sigma^2)/2
	eigen_result=eigen_y-eigen_f;
	eigen_result = -eigen_result.cwiseProduct(eigen_result) /
	                   (2.0 * std::exp(m_log_sigma * 2.0)) -
	               VectorXd::Ones(result.vlen) *
	                   log(2.0 * Math::PI * std::exp(m_log_sigma * 2.0)) / 2.0;

	return result;
}

SGVector<float64_t> GaussianLikelihood::get_log_probability_derivative_f(
		std::shared_ptr<const Labels> lab, SGVector<float64_t> func, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");
	require(i>=1 && i<=3, "Index for derivative should be 1, 2 or 3");

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// set result=y-f
	eigen_result=eigen_y-eigen_f;

	// compute derivatives of log probability wrt f
	if (i == 1)
		eigen_result /= std::exp(m_log_sigma * 2.0);
	else if (i == 2)
		eigen_result =
		    -VectorXd::Ones(result.vlen) / std::exp(m_log_sigma * 2.0);
	else if (i == 3)
		eigen_result=VectorXd::Zero(result.vlen);

	return result;
}

SGVector<float64_t> GaussianLikelihood::get_first_derivative(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func, Parameters::const_reference param) const
{
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	if (param.first != "log_sigma")
		return SGVector<float64_t>();

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of log probability wrt log_sigma:
	// dlp_dlogsigma
	// lp_dsigma=(y-f).^2/sigma^2-1
	eigen_result=eigen_y-eigen_f;
	eigen_result =
	    eigen_result.cwiseProduct(eigen_result) / std::exp(m_log_sigma * 2.0);
	eigen_result-=VectorXd::Ones(result.vlen);

	return result;
}

SGVector<float64_t> GaussianLikelihood::get_second_derivative(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func, Parameters::const_reference param) const
{
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	if (param.first != "log_sigma")
		return SGVector<float64_t>();

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();
	Map<VectorXd> eigen_y(y.vector, y.vlen);

	// compute derivative of (the first log_sigma derivative of log probability) wrt f:
	// d2lp_dlogsigma_df == d2lp_df_dlogsigma
	// dlp_dsigma=2*(f-y)/sigma^2
	eigen_result = 2.0 * (eigen_f - eigen_y) / std::exp(m_log_sigma * 2.0);

	return result;
}

SGVector<float64_t> GaussianLikelihood::get_third_derivative(std::shared_ptr<const Labels> lab,
		SGVector<float64_t> func, Parameters::const_reference param) const
{
	require(lab, "Labels are required (lab should not be NULL)");
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");
	require(lab->get_num_labels()==func.vlen, "Number of labels must match "
			"length of the function vector");

	if (param.first != "log_sigma")
		return SGVector<float64_t>();

	Map<VectorXd> eigen_f(func.vector, func.vlen);

	SGVector<float64_t> result(func.vlen);
	Map<VectorXd> eigen_result(result.vector, result.vlen);

	// compute derivative of (the derivative of the first log_sigma derivative of log probability) wrt f:
	// d3lp_dlogsigma_df_df == d3lp_df_df_dlogsigma
	// d2lp_dsigma=2/sigma^2
	eigen_result =
	    2.0 * VectorXd::Ones(result.vlen) / std::exp(m_log_sigma * 2.0);

	return result;
}

SGVector<float64_t> GaussianLikelihood::get_log_zeroth_moments(
		SGVector<float64_t> mu, SGVector<float64_t> s2, std::shared_ptr<const Labels >lab) const
{
	SGVector<float64_t> y;

	if (lab)
	{
		require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
				"Length of the vector of means ({}), length of the vector of "
				"variances ({}) and number of labels ({}) should be the same",
				mu.vlen, s2.vlen, lab->get_num_labels());
		require(lab->get_label_type()==LT_REGRESSION,
				"Labels must be type of RegressionLabels");

		y=lab->as<RegressionLabels>()->get_labels();
	}
	else
	{
		require(mu.vlen==s2.vlen, "Length of the vector of means ({}) and "
				"length of the vector of variances ({}) should be the same",
				mu.vlen, s2.vlen);

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
	eigen_s2 = eigen_s2.array() + std::exp(m_log_sigma * 2.0);
	eigen_result=-(eigen_y-eigen_mu).array().square()/(2.0*eigen_s2.array())-
		(2.0*Math::PI*eigen_s2.array()).log()/2.0;

	return result;
}

float64_t GaussianLikelihood::get_first_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels >lab, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());
	require(i>=0 && i<=mu.vlen, "Index ({}) out of bounds!", i);
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");

	SGVector<float64_t> y=lab->as<RegressionLabels>()->get_labels();

	// compute 1st moment
	float64_t Ex =
	    mu[i] + s2[i] * (y[i] - mu[i]) / (std::exp(m_log_sigma * 2.0) + s2[i]);

	return Ex;
}

float64_t GaussianLikelihood::get_second_moment(SGVector<float64_t> mu,
		SGVector<float64_t> s2, std::shared_ptr<const Labels >lab, index_t i) const
{
	// check the parameters
	require(lab, "Labels are required (lab should not be NULL)");
	require((mu.vlen==s2.vlen) && (mu.vlen==lab->get_num_labels()),
			"Length of the vector of means ({}), length of the vector of "
			"variances ({}) and number of labels ({}) should be the same",
			mu.vlen, s2.vlen, lab->get_num_labels());
	require(i>=0 && i<=mu.vlen, "Index ({}) out of bounds!", i);
	require(lab->get_label_type()==LT_REGRESSION,
			"Labels must be type of RegressionLabels");

	// compute 2nd moment
	float64_t Var =
	    s2[i] - Math::sq(s2[i]) / (std::exp(m_log_sigma * 2.0) + s2[i]);

	return Var;
}

