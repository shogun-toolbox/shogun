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

using namespace shogun;

CInferenceMethod::CInferenceMethod()
{
	init();
}

CInferenceMethod::CInferenceMethod(CKernel* kern, CFeatures* feat,
		CMeanFunction* m, CLabels* lab, CLikelihoodModel* mod)
{
	init();

	set_kernel(kern);
	set_features(feat);
	set_labels(lab);
	set_model(mod);
	set_mean(m);
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
	SG_ADD(&m_scale, "scale", "Kernel Scale", MS_AVAILABLE);
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

float64_t CInferenceMethod::get_log_ml_estimate(int32_t num_importance_samples,
		float64_t ridge_size)
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

#endif // HAVE_EIGEN3
