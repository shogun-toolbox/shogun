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

CInference::CInference()
{
	init();
}

float64_t CInference::get_scale() const
{
	return CMath::exp(m_log_scale);
}

void CInference::set_scale(float64_t scale)
{
	REQUIRE(scale>0, "Scale (%f) must be positive", scale);
	m_log_scale = std::log(scale);
}

SGMatrix<float64_t> CInference::get_multiclass_E()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_E);
}

CInference::CInference(CKernel* kernel, CFeatures* features,
	CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model)
{
	init();

	set_kernel(kernel);
	set_features(features);
	set_labels(labels);
	set_model(model);
	set_mean(mean);
}

CInference::~CInference()
{
	SG_UNREF(m_kernel);
	SG_UNREF(m_features);
	SG_UNREF(m_labels);
	SG_UNREF(m_model);
	SG_UNREF(m_mean);
	SG_UNREF(m_minimizer);
}

void CInference::init()
{
	SG_ADD((CSGObject**)&m_kernel, "kernel", "Kernel", MS_AVAILABLE);
	SG_ADD(&m_log_scale, "log_scale", "Kernel log scale", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD((CSGObject**)&m_model, "likelihood_model", "Likelihood model",
		MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_mean, "mean_function", "Mean function", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_labels, "labels", "Labels", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_features, "features", "Features", MS_NOT_AVAILABLE);
	SG_ADD(&m_gradient_update, "gradient_update", "Whether gradients are updated", MS_NOT_AVAILABLE);
	

	m_kernel=NULL;
	m_model=NULL;
	m_labels=NULL;
	m_features=NULL;
	m_mean=NULL;
	m_log_scale=0.0;
	m_gradient_update=false;
	m_minimizer=NULL;

	SG_ADD((CSGObject**)&m_minimizer, "Inference__m_minimizer", "minimizer in Inference", MS_NOT_AVAILABLE);
	SG_ADD(&m_alpha, "alpha", "alpha vector used in process mean calculation", MS_NOT_AVAILABLE);
	SG_ADD(&m_L, "L", "upper triangular factor of Cholesky decomposition", MS_NOT_AVAILABLE);
	SG_ADD(&m_E, "E", "the matrix used for multi classification", MS_NOT_AVAILABLE);
}

void CInference::register_minimizer(Minimizer* minimizer)
{
	REQUIRE(minimizer, "Minimizer must set\n");
	if(minimizer!=m_minimizer)
	{
		SG_REF(minimizer);
		SG_UNREF(m_minimizer);
		m_minimizer=minimizer;
	}
}

float64_t CInference::get_marginal_likelihood_estimate(
		int32_t num_importance_samples, float64_t ridge_size)
{
	/* sample from Gaussian approximation to q(f|y) */
	SGMatrix<float64_t> cov=get_posterior_covariance();

	/* add ridge */
	for (index_t i=0; i<cov.num_rows; ++i)
		cov(i,i)+=ridge_size;

	SGVector<float64_t> mean=get_posterior_mean();

	CGaussianDistribution* post_approx=new CGaussianDistribution(mean, cov);
	SGMatrix<float64_t> samples=post_approx->sample(num_importance_samples);

	/* evaluate q(f^i|y), p(f^i|\theta), p(y|f^i), i.e.,
	 * log pdf of approximation, prior and likelihood */

	/* log pdf q(f^i|y) */
	SGVector<float64_t> log_pdf_post_approx=post_approx->log_pdf_multiple(samples);

	/* dont need gaussian anymore, free memory */
	SG_UNREF(post_approx);
	post_approx=NULL;

	/* log pdf p(f^i|\theta) and free memory afterwise. Scale kernel before */
	SGMatrix<float64_t> scaled_kernel(m_ktrtr.num_rows, m_ktrtr.num_cols);
	sg_memcpy(scaled_kernel.matrix, m_ktrtr.matrix,
			sizeof(float64_t)*m_ktrtr.num_rows*m_ktrtr.num_cols);
	for (index_t i=0; i<m_ktrtr.num_rows*m_ktrtr.num_cols; ++i)
		scaled_kernel.matrix[i]*=CMath::exp(m_log_scale*2.0);

	/* add ridge */
	for (index_t i=0; i<m_ktrtr.num_rows; ++i)
		scaled_kernel(i,i)+=ridge_size;

	CGaussianDistribution* prior=new CGaussianDistribution(
			m_mean->get_mean_vector(m_features), scaled_kernel);
	SGVector<float64_t> log_pdf_prior=prior->log_pdf_multiple(samples);
	SG_UNREF(prior);
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
	return CMath::log_mean_exp(sum);
}

CMap<TParameter*, SGVector<float64_t> >* CInference::
get_negative_log_marginal_likelihood_derivatives(CMap<TParameter*, CSGObject*>* params)
{
	REQUIRE(params->get_num_elements(), "Number of parameters should be greater "
			"than zero\n")

	compute_gradient();

	// get number of derivatives
	const index_t num_deriv=params->get_num_elements();

	// create map of derivatives
	CMap<TParameter*, SGVector<float64_t> >* result=
		new CMap<TParameter*, SGVector<float64_t> >(num_deriv, num_deriv);

	SG_REF(result);

	#pragma omp parallel for
	for (index_t i=0; i<num_deriv; i++)
	{
        CMapNode<TParameter*, CSGObject*>* node=params->get_node_ptr(i);
        SGVector<float64_t> gradient;

		if(node->data == this)
		{
			// try to find dervative wrt InferenceMethod.parameter
			gradient=this->get_derivative_wrt_inference_method(node->key);
		}
        else if (node->data == this->m_model)
		{
			// try to find derivative wrt LikelihoodModel.parameter
			gradient=this->get_derivative_wrt_likelihood_model(node->key);
		}
		else if (node->data ==this->m_kernel)
		{
			// try to find derivative wrt Kernel.parameter
			gradient=this->get_derivative_wrt_kernel(node->key);
		}
		else if (node->data ==this->m_mean)
		{
			// try to find derivative wrt MeanFunction.parameter
			gradient=this->get_derivative_wrt_mean(node->key);
		}
		else
		{
			SG_SERROR("Can't compute derivative of negative log marginal "
					"likelihood wrt %s.%s", node->data->get_name(), node->key->m_name);
		}

		#pragma omp critical
		{
			result->add(node->key, gradient);
		}
	}

	return result;
}

void CInference::update()
{
	check_members();
	update_train_kernel();
}

void CInference::check_members() const
{
	REQUIRE(m_features, "Training features should not be NULL\n")
	REQUIRE(m_features->get_num_vectors(),
			"Number of training features must be greater than zero\n")
	REQUIRE(m_labels, "Labels should not be NULL\n")
	REQUIRE(m_labels->get_num_labels(),
			"Number of labels must be greater than zero\n")
	REQUIRE(m_labels->get_num_labels()==m_features->get_num_vectors(),
			"Number of training vectors (%d) must match number of labels (%d)\n",
			m_labels->get_num_labels(), m_features->get_num_vectors())
	REQUIRE(m_kernel, "Kernel should not be NULL\n")
	REQUIRE(m_mean, "Mean function should not be NULL\n")
}

void CInference::update_train_kernel()
{
	m_kernel->init(m_features, m_features);
	m_ktrtr=m_kernel->get_kernel_matrix();
}

void CInference::compute_gradient()
{
	if (parameter_hash_changed())
		update();
}
