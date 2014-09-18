/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 *
 * Code adapted from Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 */

#include <shogun/regression/GaussianProcessRegression.h>

#ifdef HAVE_EIGEN3

#include <shogun/io/SGIO.h>
#include <shogun/machine/gp/FITCInferenceMethod.h>

using namespace shogun;

CGaussianProcessRegression::CGaussianProcessRegression()
		: CGaussianProcessMachine()
{
}

CGaussianProcessRegression::CGaussianProcessRegression(CInferenceMethod* method)
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
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support "
			"regression\n",	m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	CRegressionLabels* result;

	// if regression data equals to NULL, then apply regression on training
	// features
	if (!data)
	{
		CFeatures* feat;

		// use latent features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC)
		{
			CFITCInferenceMethod* fitc_method=
				CFITCInferenceMethod::obtain_from_generic(m_method);
			feat=fitc_method->get_latent_features();
			SG_UNREF(fitc_method);
		}
		else
			feat=m_method->get_features();

		result=new CRegressionLabels(get_mean_vector(feat));

		SG_UNREF(feat);
	}
	else
	{
		result=new CRegressionLabels(get_mean_vector(data));
	}

	return result;
}

bool CGaussianProcessRegression::train_machine(CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports regression
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support "
			"regression\n",	m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	if (data)
	{
		// set latent features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC)
		{
			CFITCInferenceMethod* fitc_method=
				CFITCInferenceMethod::obtain_from_generic(m_method);
			fitc_method->set_latent_features(data);
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
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support "
			"regression\n",	m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

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
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support "
			"regression\n",	m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

	// evaluate variance
	s2=lik->get_predictive_variances(mu, s2);
	SG_UNREF(lik);

	return s2;
}
SGMatrix<float64_t> CGaussianProcessRegression::get_covariance_matrix(CFeatures* data)
{
	// check whether given combination of inference method and likelihood function
	// supports regression
	REQUIRE(m_method, "Inference method must be attached\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_regression(), "%s with %s doesn't support regression\n",
			m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGMatrix<float64_t> Sigma=get_posterior_covariance(data);
	SG_UNREF(data);

	REQUIRE(lik->get_model_type() == LT_GAUSSIAN, "Restricted to gaussian likelihood");
	SG_UNREF(lik);

	return Sigma;
}
#endif
