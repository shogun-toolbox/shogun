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

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/InferenceMethod.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Lock.h>

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct GRADIENT_THREAD_PARAM
{
	CInferenceMethod* inf;
	CMap<TParameter*, SGVector<float64_t> >* grad;
	CSGObject* obj;
	TParameter* param;
	CLock* lock;
};
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CInferenceMethod::CInferenceMethod()
{
	init();
}

float64_t CInferenceMethod::get_scale() const
{
	return CMath::exp(m_log_scale);
}

void CInferenceMethod::set_scale(float64_t scale)
{
	REQUIRE(scale>0, "Scale (%f) must be positive", scale);
	m_log_scale=CMath::log(scale);
}

SGMatrix<float64_t> CInferenceMethod::get_multiclass_E()
{
	if (parameter_hash_changed())
		update();

	return SGMatrix<float64_t>(m_E);
}

CInferenceMethod::CInferenceMethod(CKernel* kernel, CFeatures* features,
	CMeanFunction* mean, CLabels* labels, CLikelihoodModel* model)
{
	init();

	set_kernel(kernel);
	set_features(features);
	set_labels(labels);
	set_model(model);
	set_mean(mean);
}

CInferenceMethod::~CInferenceMethod()
{
	SG_UNREF(m_kernel);
	SG_UNREF(m_features);
	SG_UNREF(m_labels);
	SG_UNREF(m_model);
	SG_UNREF(m_mean);
}

void CInferenceMethod::init()
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

	SG_ADD(&m_alpha, "alpha", "alpha vector used in process mean calculation", MS_NOT_AVAILABLE);
	SG_ADD(&m_L, "L", "upper triangular factor of Cholesky decomposition", MS_NOT_AVAILABLE);
	SG_ADD(&m_E, "E", "the matrix used for multi classification", MS_NOT_AVAILABLE);
}

float64_t CInferenceMethod::get_marginal_likelihood_estimate(
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
	memcpy(scaled_kernel.matrix, m_ktrtr.matrix,
			sizeof(float64_t)*m_ktrtr.num_rows*m_ktrtr.num_cols);
	for (index_t i=0; i<m_ktrtr.num_rows*m_ktrtr.num_cols; ++i)
		scaled_kernel.matrix[i]*=CMath::exp(m_log_scale*2.0);

	/* add ridge */
	for (index_t i=0; i<cov.num_rows; ++i)
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

CMap<TParameter*, SGVector<float64_t> >* CInferenceMethod::
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

	// create lock object
	CLock lock;

#ifdef HAVE_PTHREAD
	if (num_deriv<2)
	{
#endif /* HAVE_PTHREAD */
		for (index_t i=0; i<num_deriv; i++)
		{
			CMapNode<TParameter*, CSGObject*>* node=params->get_node_ptr(i);

			GRADIENT_THREAD_PARAM thread_params;

			thread_params.inf=this;
			thread_params.obj=node->data;
			thread_params.param=node->key;
			thread_params.grad=result;
			thread_params.lock=&lock;

			get_derivative_helper((void*) &thread_params);
		}
#ifdef HAVE_PTHREAD
	}
	else
	{
		pthread_t* threads=SG_MALLOC(pthread_t, num_deriv);
		GRADIENT_THREAD_PARAM* thread_params=SG_MALLOC(GRADIENT_THREAD_PARAM,
				num_deriv);

		for (index_t t=0; t<num_deriv; t++)
		{
			CMapNode<TParameter*, CSGObject*>* node=params->get_node_ptr(t);

			thread_params[t].inf=this;
			thread_params[t].obj=node->data;
			thread_params[t].param=node->key;
			thread_params[t].grad=result;
			thread_params[t].lock=&lock;

			pthread_create(&threads[t], NULL, CInferenceMethod::get_derivative_helper,
					(void*)&thread_params[t]);
		}

		for (index_t t=0; t<num_deriv; t++)
			pthread_join(threads[t], NULL);

		SG_FREE(thread_params);
		SG_FREE(threads);
	}
#endif /* HAVE_PTHREAD */

	return result;
}

void* CInferenceMethod::get_derivative_helper(void *p)
{
	GRADIENT_THREAD_PARAM* thread_param=(GRADIENT_THREAD_PARAM*)p;

	CInferenceMethod* inf=thread_param->inf;
	CSGObject* obj=thread_param->obj;
	CMap<TParameter*, SGVector<float64_t> >* grad=thread_param->grad;
	TParameter* param=thread_param->param;
	CLock* lock=thread_param->lock;

	REQUIRE(param, "Parameter should not be NULL\n");
	REQUIRE(obj, "Object of the parameter should not be NULL\n");

	SGVector<float64_t> gradient;

	if (obj==inf)
	{
		// try to find dervative wrt InferenceMethod.parameter
		gradient=inf->get_derivative_wrt_inference_method(param);
	}
	else if (obj==inf->m_model)
	{
		// try to find derivative wrt LikelihoodModel.parameter
		gradient=inf->get_derivative_wrt_likelihood_model(param);
	}
	else if (obj==inf->m_kernel)
	{
		// try to find derivative wrt Kernel.parameter
		gradient=inf->get_derivative_wrt_kernel(param);
	}
	else if (obj==inf->m_mean)
	{
		// try to find derivative wrt MeanFunction.parameter
		gradient=inf->get_derivative_wrt_mean(param);
	}
	else
	{
		SG_SERROR("Can't compute derivative of negative log marginal "
				"likelihood wrt %s.%s", obj->get_name(), param->m_name);
	}

	lock->lock();
	grad->add(param, gradient);
	lock->unlock();

	return NULL;
}

void CInferenceMethod::update()
{
	check_members();
	update_train_kernel();
}

void CInferenceMethod::check_members() const
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

void CInferenceMethod::update_train_kernel()
{
	m_kernel->init(m_features, m_features);
	m_ktrtr=m_kernel->get_kernel_matrix();
}

void CInferenceMethod::compute_gradient()
{
	if (parameter_hash_changed())
		update();
}
#endif /* HAVE_EIGEN3 */
