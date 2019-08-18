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

CGaussianProcessRegression::CGaussianProcessRegression()
		: CGaussianProcessMachine()
{
}

CGaussianProcessRegression::CGaussianProcessRegression(CInference* method)
		: CGaussianProcessMachine(method)
{
	// set labels
	m_labels=method->get_labels();
}

CGaussianProcessRegression::~CGaussianProcessRegression()
{
}

CRegressionLabels* CGaussianProcessRegression::apply_regression(CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	CLikelihoodModel* lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());
	SG_UNREF(lik);

	CRegressionLabels* result;

	// if regression data equals to NULL, then apply regression on training
	// features
	if (!data)
	{
		CFeatures* feat;

		// use inducing features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC_REGRESSION)
		{
			CFITCInferenceMethod* fitc_method = m_method->as<CFITCInferenceMethod>();
			feat=fitc_method->get_inducing_features();
		}
		else
			feat=m_method->get_features();

		result=new CRegressionLabels(get_mean_vector(feat));
		if (m_compute_variance)
			result->put("current_values", get_variance_vector(feat));

		SG_UNREF(feat);
	}
	else
	{
		result=new CRegressionLabels(get_mean_vector(data));
		if (m_compute_variance)
			result->put("current_values", get_variance_vector(data));
	}

	return result;
}

bool CGaussianProcessRegression::train_machine(CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	CLikelihoodModel* lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());
	SG_UNREF(lik);

	if (data)
	{
		// set inducing features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC_REGRESSION)
		{
			CFITCInferenceMethod* fitc_method = m_method->as<CFITCInferenceMethod>();
			fitc_method->set_inducing_features(data);
			SG_UNREF(fitc_method);
		}
		else
			m_method->set_features(data);
	}

	// perform inference
	m_method->update();

	return true;
}

SGVector<float64_t> CGaussianProcessRegression::get_mean_vector(CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	CLikelihoodModel* lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());
	SG_UNREF(lik);

	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);

	// evaluate mean
	lik=m_method->get_model();
	mu=lik->get_predictive_means(mu, s2);
	SG_UNREF(lik);

	return mu;
}

SGVector<float64_t> CGaussianProcessRegression::get_variance_vector(
		CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	require(m_method, "Inference method should not be NULL");
	CLikelihoodModel* lik=m_method->get_model();
	require(m_method->supports_regression(), "{} with {} doesn't support "
			"regression",	m_method->get_name(), lik->get_name());

	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);

	// evaluate variance
	s2=lik->get_predictive_variances(mu, s2);
	SG_UNREF(lik);

	return s2;
}
