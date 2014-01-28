/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/classifier/GaussianProcessBinaryClassification.h>

using namespace shogun;

CGaussianProcessBinaryClassification::CGaussianProcessBinaryClassification()
	: CGaussianProcessMachine()
{
}

CGaussianProcessBinaryClassification::CGaussianProcessBinaryClassification(
		CInferenceMethod* method) : CGaussianProcessMachine(method)
{
	// set labels
	m_labels=method->get_labels();
}

CGaussianProcessBinaryClassification::~CGaussianProcessBinaryClassification()
{
}

CBinaryLabels* CGaussianProcessBinaryClassification::apply_binary(
		CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary(), "%s with %s doesn't support "
			"binary classification\n", m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	// if regression data equals to NULL, then apply classification on training
	// features
	if (!data)
		data=m_method->get_features();
	else
		SG_REF(data);

	CBinaryLabels* result=new CBinaryLabels(get_mean_vector(data));
	SG_UNREF(data);

	return result;
}

bool CGaussianProcessBinaryClassification::train_machine(CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary(), "%s with %s doesn't support "
			"binary classification\n", m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	if (data)
		m_method->set_features(data);

	// perform inference
	if (m_method->update_parameter_hash())
		m_method->update();

	return true;
}

SGVector<float64_t> CGaussianProcessBinaryClassification::get_mean_vector(
		CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary(), "%s with %s doesn't support "
			"binary classification\n", m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

	// evaluate mean
	mu=lik->get_predictive_means(mu, s2);
	SG_UNREF(lik);

	return mu;
}

SGVector<float64_t> CGaussianProcessBinaryClassification::get_variance_vector(
		CFeatures* data)
{
	// check whether given combination of inference method and
	// likelihood function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary(), "%s with %s doesn't support "
			"binary classification\n", m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

	// evaluate variance
	s2=lik->get_predictive_variances(mu, s2);
	SG_UNREF(lik);

	return s2;
}

SGVector<float64_t> CGaussianProcessBinaryClassification::get_probabilities(
		CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary(), "%s with %s doesn't support "
			"binary classification\n", m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

	// evaluate log probabilities
	SGVector<float64_t> p=lik->get_predictive_log_probabilities(mu, s2);
	SG_UNREF(lik);

	// evaluate probabilities
	p.exp();

	return p;
}

#endif /* HAVE_EIGEN3 */
