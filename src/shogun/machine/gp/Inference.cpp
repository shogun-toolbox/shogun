/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2013 Heiko Strathmann
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
#include <shogun/lib/config.h>


#include <shogun/machine/gp/Inference.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

Inference::Inference()
{
	init();
}

float64_t Inference::get_scale() const
{
	return std::exp(m_log_scale);
}

void Inference::set_scale(float64_t scale)
{
	require(scale>0, "Scale ({}) must be positive", scale);
	m_log_scale = std::log(scale);
}

SGMatrix<float64_t> Inference::get_multiclass_E()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_E);
}

Inference::Inference(std::shared_ptr<Kernel> kernel, std::shared_ptr<Features> features,
	std::shared_ptr<MeanFunction> mean, std::shared_ptr<Labels> labels, std::shared_ptr<LikelihoodModel> model)
{
	init();

	set_kernel(kernel);
	set_features(features);
	set_labels(labels);
	set_model(model);
	set_mean(mean);
}

Inference::~Inference()
{






}

void Inference::init()
{
	SG_ADD(&m_kernel, "kernel", "Kernel", ParameterProperties::HYPER);
	SG_ADD(&m_log_scale, "log_scale", "Kernel log scale", ParameterProperties::HYPER | ParameterProperties::GRADIENT);
	SG_ADD(&m_model, "likelihood_model", "Likelihood model", ParameterProperties::HYPER);
	SG_ADD(&m_mean, "mean_function", "Mean function", ParameterProperties::HYPER);
	SG_ADD(&m_labels, "labels", "Labels");
	SG_ADD(&m_features, "features", "Features");
	SG_ADD(&m_gradient_update, "gradient_update", "Whether gradients are updated");


	m_kernel=NULL;
	m_model=NULL;
	m_labels=NULL;
	m_features=NULL;
	m_mean=NULL;
	m_log_scale=0.0;
	m_gradient_update=false;
	m_minimizer=NULL;

	SG_ADD((std::shared_ptr<SGObject>*)&m_minimizer, "Inference__m_minimizer", "minimizer in Inference");
	SG_ADD(&m_alpha, "alpha", "alpha vector used in process mean calculation");
	SG_ADD(&m_L, "L", "upper triangular factor of Cholesky decomposition");
	SG_ADD(&m_E, "E", "the matrix used for multi classification");
}

void Inference::register_minimizer(std::shared_ptr<Minimizer> minimizer)
{
	require(minimizer, "Minimizer must set");
	if(minimizer!=m_minimizer)
	{


		m_minimizer=minimizer;
	}
}

float64_t Inference::get_marginal_likelihood_estimate(
		int32_t num_importance_samples, float64_t ridge_size)
{
	/* sample from Gaussian approximation to q(f|y) */
	SGMatrix<float64_t> cov=get_posterior_covariance();

	/* add ridge */
	for (index_t i=0; i<cov.num_rows; ++i)
		cov(i,i)+=ridge_size;

	SGVector<float64_t> mean=get_posterior_mean();

	auto post_approx=std::make_shared<GaussianDistribution>(mean, cov);
	SGMatrix<float64_t> samples=post_approx->sample(num_importance_samples);

	/* evaluate q(f^i|y), p(f^i|\theta), p(y|f^i), i.e.,
	 * log pdf of approximation, prior and likelihood */

	/* log pdf q(f^i|y) */
	SGVector<float64_t> log_pdf_post_approx=post_approx->log_pdf_multiple(samples);

	/* dont need gaussian anymore, free memory */

	post_approx=NULL;

	/* log pdf p(f^i|\theta) and free memory afterwise. Scale kernel before */
	SGMatrix<float64_t> scaled_kernel(m_ktrtr.num_rows, m_ktrtr.num_cols);
	sg_memcpy(scaled_kernel.matrix, m_ktrtr.matrix,
			sizeof(float64_t)*m_ktrtr.num_rows*m_ktrtr.num_cols);
	for (index_t i=0; i<m_ktrtr.num_rows*m_ktrtr.num_cols; ++i)
		scaled_kernel.matrix[i] *= std::exp(m_log_scale * 2.0);

	/* add ridge */
	for (index_t i=0; i<m_ktrtr.num_rows; ++i)
		scaled_kernel(i,i)+=ridge_size;

	auto prior=std::make_shared<GaussianDistribution>(
			m_mean->get_mean_vector(m_features), scaled_kernel);
	SGVector<float64_t> log_pdf_prior=prior->log_pdf_multiple(samples);

	prior=NULL;

	/* p(y|f^i) */
	SGVector<float64_t> log_likelihood=m_model->get_log_probability_fmatrix(
			m_labels, samples);

	/* combine probabilities */
	ASSERT(log_likelihood.vlen==num_importance_samples);
	ASSERT(log_likelihood.vlen==log_pdf_prior.vlen);
	ASSERT(log_likelihood.vlen==log_pdf_post_approx.vlen);
	SGVector<float64_t> sum(log_likelihood);
	for (index_t i=0; i<log_likelihood.vlen; ++i)
		sum[i]=log_likelihood[i]+log_pdf_prior[i]-log_pdf_post_approx[i];

	/* use log-sum-exp (in particular, log-mean-exp) trick to combine values */
	return Math::log_mean_exp(sum);
}

std::shared_ptr<CMap<TParameter*, SGVector<float64_t> >> Inference::
get_negative_log_marginal_likelihood_derivatives(std::shared_ptr<CMap<TParameter*, SGObject*>> params)
{
	require(params->get_num_elements(), "Number of parameters should be greater "
			"than zero");

	compute_gradient();

	// get number of derivatives
	const index_t num_deriv=params->get_num_elements();

	// create map of derivatives
	auto result=
		std::make_shared<CMap<TParameter*, SGVector<float64_t>>>(num_deriv, num_deriv);



	#pragma omp parallel for
	for (index_t i=0; i<num_deriv; i++)
	{
        CMapNode<TParameter*, SGObject*>* node=params->get_node_ptr(i);
        SGVector<float64_t> gradient;

		if(node->data == this)
		{
			// try to find dervative wrt InferenceMethod.parameter
			gradient=this->get_derivative_wrt_inference_method(node->key);
		}
        else if (node->data == this->m_model.get())
		{
			// try to find derivative wrt LikelihoodModel.parameter
			gradient=this->get_derivative_wrt_likelihood_model(node->key);
		}
		else if (node->data ==this->m_kernel.get())
		{
			// try to find derivative wrt Kernel.parameter
			gradient=this->get_derivative_wrt_kernel(node->key);
		}
		else if (node->data ==this->m_mean.get())
		{
			// try to find derivative wrt MeanFunction.parameter
			gradient=this->get_derivative_wrt_mean(node->key);
		}
		else
		{
			error("Can't compute derivative of negative log marginal "
					"likelihood wrt {}.{}", node->data->get_name(), node->key->m_name);
		}

		#pragma omp critical
		{
			result->add(node->key, gradient);
		}
	}

	return result;
}

void Inference::update()
{
	check_members();
	update_train_kernel();
}

void Inference::check_members() const
{
	require(m_features, "Training features should not be NULL");
	require(m_features->get_num_vectors(),
			"Number of training features must be greater than zero");
	require(m_labels, "Labels should not be NULL");
	require(m_labels->get_num_labels(),
			"Number of labels must be greater than zero");
	require(m_labels->get_num_labels()==m_features->get_num_vectors(),
			"Number of training vectors ({}) must match number of labels ({})",
			m_labels->get_num_labels(), m_features->get_num_vectors());
	require(m_kernel, "Kernel should not be NULL");
	require(m_mean, "Mean function should not be NULL");
}

void Inference::update_train_kernel()
{
	m_kernel->init(m_features, m_features);
	m_ktrtr=m_kernel->get_kernel_matrix();
}

void Inference::compute_gradient()
{
	if (parameter_hash_changed())
		update();
}
