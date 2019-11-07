/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Jacob Walker, Roman Votyakov, Sergey Lisitsyn, Soeren Sonnenburg,
 *          Heiko Strathmann, Wu Lin
 */


#include <shogun/regression/GaussianProcessRegression.h>
#include <shogun/io/SGIO.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>

using namespace shogun;

GaussianProcessRegression::GaussianProcessRegression()
		: GaussianProcessMachine()
{
}

GaussianProcessRegression::GaussianProcessRegression(const std::shared_ptr<Inference>& method)
		: GaussianProcessMachine(method)
{
	// set labels
	m_labels=method->get_labels();
}

GaussianProcessRegression::~GaussianProcessRegression()
{
}

std::shared_ptr<RegressionLabels> GaussianProcessRegression::apply_regression(std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	auto lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());


	std::shared_ptr<RegressionLabels> result;

	// if regression data equals to NULL, then apply regression on training
	// features
	if (!data)
	{
		std::shared_ptr<Features> feat;

		// use inducing features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC_REGRESSION)
		{
			auto fitc_method = m_method->as<FITCInferenceMethod>();
			feat=fitc_method->get_inducing_features();
		}
		else
			feat=m_method->get_features();

		result=std::make_shared<RegressionLabels>(get_mean_vector(feat));
		if (m_compute_variance)
			result->put("current_values", get_variance_vector(feat));


	}
	else
	{
		result=std::make_shared<RegressionLabels>(get_mean_vector(data));
		if (m_compute_variance)
			result->put("current_values", get_variance_vector(data));
	}

	return result;
}

bool GaussianProcessRegression::train_machine(std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	auto lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());
	if (data)
	{
		// set inducing features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC_REGRESSION)
		{
			auto fitc_method = m_method->as<FITCInferenceMethod>();
			fitc_method->set_inducing_features(data);

		}
		else
			m_method->set_features(data);
	}

	// perform inference
	m_method->update();

	return true;
}

SGVector<float64_t> GaussianProcessRegression::get_mean_vector(const std::shared_ptr<Features>& data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	auto lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());

	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);

	// evaluate mean
	lik=m_method->get_model();
	mu=lik->get_predictive_means(mu, s2);


	return mu;
}

SGVector<float64_t> GaussianProcessRegression::get_variance_vector(
		const std::shared_ptr<Features>& data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	auto lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());

	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);

	// evaluate variance
	s2=lik->get_predictive_variances(mu, s2);


	return s2;
}
