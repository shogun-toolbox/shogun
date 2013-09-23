/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Roman Votyakov
 * Written (W) 2013 Heiko Strathmann
 * Copyright (C) 2012 Jacob Walker
 * Copyright (C) 2013 Roman Votyakov
 */

#include <shogun/lib/config.h>

#ifdef HAVE_EIGEN3

#include <shogun/machine/gp/InferenceMethod.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/distributions/classical/GaussianDistribution.h>
#include <shogun/mathematics/Statistics.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct GRADIENT_THREAD_PARAM
{
	CInferenceMethod* inf;
	CMap<TParameter*, SGVector<float64_t> >* grad;
	CSGObject* obj;
	TParameter* param;
};
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CInferenceMethod::CInferenceMethod()
{
	init();
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
	SG_ADD(&m_scale, "scale", "Kernel scale", MS_AVAILABLE, GRADIENT_AVAILABLE);
	SG_ADD((CSGObject**)&m_model, "likelihood_model", "Likelihood model",
			MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_mean, "mean_function", "Mean function", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_labels, "labels", "Labels", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**)&m_features, "features", "Features", MS_NOT_AVAILABLE);

	m_kernel=NULL;
	m_model=NULL;
	m_labels=NULL;
	m_features=NULL;
	m_mean=NULL;
	m_scale=1.0;
}

float64_t CInferenceMethod::get_marginal_likelihood_estimate(
		int32_t num_importance_samples, float64_t ridge_size)
{
	/* sample from Gaussian approximation to q(f|y) */
	SGMatrix<float64_t> cov=get_posterior_approximation_covariance();

	/* add ridge */
	for (index_t i=0; i<cov.num_rows; ++i)
		cov(i,i)+=ridge_size;

	SGVector<float64_t> mean=get_posterior_approximation_mean();

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
		scaled_kernel.matrix[i]*=CMath::sq(m_scale);

	/* add ridge */
	for (index_t i=0; i<cov.num_rows; ++i)
		scaled_kernel(i,i)+=ridge_size;

	CGaussianDistribution* prior=new CGaussianDistribution(
			m_mean->get_mean_vector(m_feat), scaled_kernel);
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

	if (update_parameter_hash())
		update();

	// get number of derivatives
	const index_t num_deriv=params->get_num_elements();

	// create map of derivatives
	CMap<TParameter*, SGVector<float64_t> >* result=
		new CMap<TParameter*, SGVector<float64_t> >(num_deriv, num_deriv);

	SG_REF(result);

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

	REQUIRE(param, "Parameter should not be NULL\n");
	REQUIRE(obj, "Object of the parameter should not be NULL\n");

	if (obj==inf)
	{
		// try to find dervative wrt InferenceMethod.parameter
		grad->add(param, inf->get_derivative_wrt_inference_method(param));
	}
	else if (obj==inf->m_model)
	{
		// try to find derivative wrt LikelihoodModel.parameter
		grad->add(param, inf->get_derivative_wrt_likelihood_model(param));
	}
	else if (obj==inf->m_kernel)
	{
		// try to find derivative wrt Kernel.parameter
		grad->add(param, inf->get_derivative_wrt_kernel(param));
	}
	else if (obj==inf->m_mean)
	{
		// try to find derivative wrt MeanFunction.parameter
		grad->add(param, inf->get_derivative_wrt_mean(param));
	}
	else
	{
		SG_SERROR("Can't compute derivative of negative log marginal "
				"likelihood wrt %s.%s", obj->get_name(), param->m_name);
	}

	return NULL;
}

void CInferenceMethod::update()
{
	check_members();
	update_feature_matrix();
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
			"Number of training vectors must match number of labels, which is "
			"%d, but number of training vectors is %d\n",
			m_labels->get_num_labels(), m_features->get_num_vectors())
	REQUIRE(m_kernel, "Kernel should not be NULL\n")
	REQUIRE(m_mean, "Mean function should not be NULL\n")

	CFeatures* feat=m_features;

	if (m_features->get_feature_class()==C_COMBINED)
		feat=((CCombinedFeatures*)m_features)->get_first_feature_obj();
	else
		SG_REF(m_features);

	REQUIRE(feat->has_property(FP_DOT),
			"Training features must be type of CFeatures\n")
	REQUIRE(feat->get_feature_class()==C_DENSE,
			"Training features must be dense\n")
	REQUIRE(feat->get_feature_type()==F_DREAL,
			"Training features must be real\n")

	SG_UNREF(feat);
}

void CInferenceMethod::update_train_kernel()
{
	m_kernel->cleanup();
	m_kernel->init(m_features, m_features);
	m_ktrtr=m_kernel->get_kernel_matrix();
}

void CInferenceMethod::update_feature_matrix()
{
	CFeatures* feat=m_features;

	if (m_features->get_feature_class()==C_COMBINED)
		feat=((CCombinedFeatures*)m_features)->get_first_feature_obj();
	else
		SG_REF(m_features);

	m_feat=((CDotFeatures*)feat)->get_computed_dot_feature_matrix();

	SG_UNREF(feat);
}

#endif /* HAVE_EIGEN3 */
