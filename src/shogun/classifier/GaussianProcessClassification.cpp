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
#include <shogun/machine/gp/SingleFITCLaplacianInferenceMethod.h>

using namespace shogun;

CGaussianProcessClassification::CGaussianProcessClassification()
	: CGaussianProcessMachine()
{
}

CGaussianProcessClassification::CGaussianProcessClassification(
		CInferenceMethod* method) : CGaussianProcessMachine(method)
{
	// set labels
	m_labels=method->get_labels();
}

CGaussianProcessClassification::~CGaussianProcessClassification()
{
}

CMulticlassLabels* CGaussianProcessClassification::apply_multiclass(CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_multiclass(), "%s with %s doesn't support "
			"multi classification\n", m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

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
	else
		SG_REF(data);

	const index_t n=data->get_num_vectors();
	SGVector<float64_t> mean=get_mean_vector(data);
	const index_t C=mean.vlen/n;
	SGVector<index_t> lab(n);
	for (index_t idx=0; idx<n; idx++)
	{
		int32_t cate=CMath::arg_max(mean.vector+idx*C, 1, C);
		lab[idx]=cate;
	}
	CMulticlassLabels *result=new CMulticlassLabels();
	result->set_int_labels(lab);

	SG_UNREF(data);

	return result;
}

CBinaryLabels* CGaussianProcessClassification::apply_binary(
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
	{
		if (m_method->get_inference_type()==INF_FITC_LAPLACIAN_SINGLE)
		{
			CSingleFITCLaplacianInferenceMethod* fitc_method=
				CSingleFITCLaplacianInferenceMethod::obtain_from_generic(m_method);
			data=fitc_method->get_inducing_features();
			SG_UNREF(fitc_method);
		}
		else
			data=m_method->get_features();
	}
	else
		SG_REF(data);

	CBinaryLabels* result=new CBinaryLabels(get_mean_vector(data));
	SG_UNREF(data);

	return result;
}

bool CGaussianProcessClassification::train_machine(CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(), "%s with %s doesn't support "
			"classification\n", m_method->get_name(), lik->get_name())
	SG_UNREF(lik);

	if (data)
	{
		// set inducing features for FITC inference method
		if (m_method->get_inference_type()==INF_FITC_LAPLACIAN_SINGLE)
		{
			CSingleFITCLaplacianInferenceMethod* fitc_method=
				CSingleFITCLaplacianInferenceMethod::obtain_from_generic(m_method);
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

SGVector<float64_t> CGaussianProcessClassification::get_mean_vector(
		CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(),
		"%s with %s doesn't support classification\n", m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

	// evaluate mean
	mu=lik->get_predictive_means(mu, s2);
	SG_UNREF(lik);

	return mu;
}

SGVector<float64_t> CGaussianProcessClassification::get_variance_vector(
		CFeatures* data)
{
	// check whether given combination of inference method and
	// likelihood function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(),
		"%s with %s doesn't support classification\n", m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

	// evaluate variance
	s2=lik->get_predictive_variances(mu, s2);
	SG_UNREF(lik);

	return s2;
}

SGVector<float64_t> CGaussianProcessClassification::get_probabilities(
		CFeatures* data)
{
	// check whether given combination of inference method and likelihood
	// function supports classification
	REQUIRE(m_method, "Inference method should not be NULL\n")
	CLikelihoodModel* lik=m_method->get_model();
	REQUIRE(m_method->supports_binary() || m_method->supports_multiclass(),
		"%s with %s doesn't support classification\n", m_method->get_name(), lik->get_name())

	SG_REF(data);
	SGVector<float64_t> mu=get_posterior_means(data);
	SGVector<float64_t> s2=get_posterior_variances(data);
	SG_UNREF(data);

	// evaluate log probabilities
	SGVector<float64_t> p=lik->get_predictive_log_probabilities(mu, s2);
	SG_UNREF(lik);

	// evaluate probabilities
	for (index_t idx=0; idx<p.vlen; idx++)
		p[idx]=CMath::exp(p[idx]);

	return p;
}

