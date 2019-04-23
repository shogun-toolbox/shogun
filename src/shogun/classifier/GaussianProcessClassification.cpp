/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 Wu Lin
 * Written (W) 2013 Roman Votyakov
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
 * Code adapted from
 * Gaussian Process Machine Learning Toolbox
 * http://www.gaussianprocess.org/gpml/code/matlab/doc/
 * and
 * https://gist.github.com/yorkerlin/8a36e8f9b298aa0246a4
 */


#include <shogun/lib/config.h>
#include <shogun/classifier/GaussianProcessClassification.h>
#include <shogun/mathematics/Math.h>
#ifdef USE_GPL_SHOGUN
#include <shogun/machine/gp/SingleFITCLaplaceInferenceMethod.h>
#endif //USE_GPL_SHOGUN

using namespace shogun;

GaussianProcessClassification::GaussianProcessClassification()
	: GaussianProcessMachine()
{
}

GaussianProcessClassification::GaussianProcessClassification(
		std::shared_ptr<Inference> method) : GaussianProcessMachine(method)
{
	// set labels
	m_labels=method->get_labels();
}

GaussianProcessClassification::~GaussianProcessClassification()
{
}

std::shared_ptr<MulticlassLabels> GaussianProcessClassification::apply_multiclass(std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	auto lik=m_method->get_model();
	REQUIRE(m_method->supports_multiclass(), "%s with %s doesn't support "
			"multi classification\n", m_method->get_name(), lik->get_name())


	// if regression data equals to NULL, then apply classification on training
	// features
	if (!data)
	{
		if (m_method->get_inference_type()==INF_SPARSE)
		{
			SG_NOTIMPLEMENTED
		}
		else
			data=m_method->get_features();
	}

	const index_t n=data->get_num_vectors();
	SGVector<float64_t> mean=get_mean_vector(data);
	const index_t C=mean.vlen/n;
	SGVector<index_t> lab(n);
	for (index_t idx=0; idx<n; idx++)
	{
		int32_t cate=Math::arg_max(mean.vector+idx*C, 1, C);
		lab[idx]=cate;
	}
	auto result=std::make_shared<MulticlassLabels>();
	result->set_int_labels(lab);



	return result;
}

std::shared_ptr<BinaryLabels> GaussianProcessClassification::apply_binary(
		std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	auto lik=m_method->get_model();
	REQUIRE(m_method->supports_binary(), "%s with %s doesn't support "
			"binary classification\n", m_method->get_name(), lik->get_name())


	// if regression data equals to NULL, then apply classification on training
	// features
	if (!data)
	{
		if (m_method->get_inference_type()== INF_FITC_LAPLACE_SINGLE)
		{
#ifdef USE_GPL_SHOGUN
			auto fitc_method = m_method->as<SingleFITCLaplaceInferenceMethod>();
			data=fitc_method->get_inducing_features();
#else
			SG_GPL_ONLY
#endif //USE_GPL_SHOGUN
		}
		else
			data=m_method->get_features();
	}

	auto result=std::make_shared<BinaryLabels>(get_mean_vector(data));
	if (m_compute_variance)
		result->put("current_values", get_variance_vector(data));


	return result;
}

bool GaussianProcessClassification::train_machine(std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	auto lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(), "%s with %s doesn't support "
			"classification\n", m_method->get_name(), lik->get_name())


	if (data)
	{
		// set inducing features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC_LAPLACE_SINGLE)
		{
#ifdef USE_GPL_SHOGUN
			auto fitc_method = m_method->as<SingleFITCLaplaceInferenceMethod>();
			fitc_method->set_inducing_features(data);
#else
			SG_ERROR("Single FITC Laplace inference only supported under GPL.\n")
#endif //USE_GPL_SHOGUN
		}
		else
			m_method->set_features(data);
	}

	// perform inference
	m_method->update();

	return true;
}

SGVector<float64_t> GaussianProcessClassification::get_mean_vector(
		std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	auto lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(),
		"%s with %s doesn't support classification\n", m_method->get_name(), lik->get_name())


	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);


	// evaluate mean
	mu=lik->get_predictive_means(mu, s2);


	return mu;
}

SGVector<float64_t> GaussianProcessClassification::get_variance_vector(
		std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and
	// likelihood function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	auto lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(),
		"%s with %s doesn't support classification\n", m_method->get_name(), lik->get_name())


	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);


	// evaluate variance
	s2=lik->get_predictive_variances(mu, s2);


	return s2;
}

SGVector<float64_t> GaussianProcessClassification::get_probabilities(
		std::shared_ptr<Features> data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	auto lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(),
		"%s with %s doesn't support classification\n", m_method->get_name(), lik->get_name())


	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);


	// evaluate log probabilities
	SGVector<float64_t> p=lik->get_predictive_log_probabilities(mu, s2);


	// evaluate probabilities
	for (index_t idx=0; idx<p.vlen; idx++)
		p[idx] = std::exp(p[idx]);

	return p;
}
