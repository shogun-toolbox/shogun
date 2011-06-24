/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Alesis Novik
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include "lib/config.h"

#ifdef HAVE_LAPACK

#include "clustering/GMM.h"
#include "clustering/KMeans.h"
#include "distance/EuclidianDistance.h"
#include "base/Parameter.h"
#include "lib/Mathematics.h"
#include "lib/lapack.h"

using namespace shogun;

CGMM::CGMM() : CDistribution(), m_components(NULL), m_n(0),
				m_coefficients(NULL), m_coef_size(0), m_max_iter(0), m_minimal_change(0)
{
	register_params();
}

CGMM::CGMM(int32_t n_, int32_t max_iter_, float64_t min_change_) : CDistribution(), m_components(NULL), m_n(n_),
						m_coefficients(NULL), m_coef_size(n_), m_max_iter(max_iter_),
						m_minimal_change(min_change_)
{
	register_params();
}

CGMM::~CGMM()
{
	if (m_components)
		cleanup();
}

void CGMM::cleanup()
{
	for (int i = 0; i < m_n; i++)
		SG_UNREF(m_components[i]);

	delete[] m_components;
	delete[] m_coefficients;
}

bool CGMM::train(CFeatures* data)
{
	ASSERT(m_n != 0);
	if (m_components)
		cleanup();

	/** init features with data if necessary and assure type is correct */
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n");		
		set_features(data);
	}

	CDotFeatures* dotdata = (CDotFeatures *) data;
	int32_t num_vectors = dotdata->get_num_vectors();
	int32_t num_dim = dotdata->get_dim_feature_space();

	CEuclidianDistance* dist = new CEuclidianDistance();
	CKMeans* init_k_means = new CKMeans(m_n, dist);
	init_k_means->train(dotdata);
	// sorry ;)
	SGMatrix<float64_t> cluster_centers = init_k_means->get_cluster_centers();
	float64_t* init_means = cluster_centers.matrix;
	int32_t init_mean_dim = cluster_centers.num_rows;
	int32_t init_mean_size = cluster_centers.num_cols;


	float64_t* init_cov;
	int32_t init_cov_rows;
	int32_t init_cov_cols;
	dotdata->get_cov(&init_cov, &init_cov_rows, &init_cov_cols);

	m_coefficients = new float64_t[m_coef_size];
	m_components = new CGaussian*[m_n];

	for (int i=0; i<m_n; i++)
	{
		m_coefficients[i] = 1.0/m_coef_size;
		m_components[i] = new CGaussian(SGVector<float64_t>(&(init_means[i*init_mean_dim]), init_mean_dim),
								        SGMatrix<float64_t>(init_cov, init_cov_rows, init_cov_cols));
	}

	/** question of faster vs. less memory using */
	float64_t* pdfs = new float64_t[num_vectors*m_n];
	float64_t* T = new float64_t[num_vectors*m_n];
	int32_t iter = 0;
	float64_t e_log_likelihood_change = m_minimal_change + 1;
	float64_t e_log_likelihood_old = 0;
	float64_t e_log_likelihood_new = -FLT_MAX;

	while (iter<m_max_iter && e_log_likelihood_change>m_minimal_change)
	{
		e_log_likelihood_old = e_log_likelihood_new;
		e_log_likelihood_new = 0;

		/** Precomputing likelihoods */
		for (int i=0; i<num_vectors; i++)
		{
			SGVector<float64_t> v= dotdata->get_feature_vector(i);
			for (int j=0; j<m_n; j++)
				pdfs[i*m_n+j] = m_components[j]->compute_PDF(v.vector, v.vlen);
			v.free_vector();
		}

		for (int i=0; i<num_vectors; i++)
		{
			float64_t sum = 0;

			for (int j=0; j<m_n; j++)
				sum += m_coefficients[j]*pdfs[i*m_n+j];

			for (int j=0; j<m_n; j++)
			{
				T[i*m_n+j] = (m_coefficients[j]*pdfs[i*m_n+j])/sum;
				e_log_likelihood_new += T[i*m_n+j]*CMath::log(m_coefficients[j]*pdfs[i*m_n+j]);
			}
		}

		/** Not sure if getting the abs value is a good idea */
		e_log_likelihood_change = CMath::abs(e_log_likelihood_new - e_log_likelihood_old);

		/** Updates */
		float64_t T_sum;
		float64_t* mean_sum;
		float64_t* cov_sum;

		for (int i=0; i<m_n; i++)
		{
			T_sum = 0;
			mean_sum = new float64_t[num_dim];
			memset(mean_sum, 0, num_dim*sizeof(float64_t));

			for (int j=0; j<num_vectors; j++)
			{
				T_sum += T[j*m_n+i];
				SGVector<float64_t> v=dotdata->get_feature_vector(j);
				CMath::add<float64_t>(mean_sum, T[j*m_n+i], v.vector, 1, mean_sum, v.vlen);
				v.free_vector();
			}

			m_coefficients[i] = T_sum/num_vectors;

			for (int j=0; j<num_dim; j++)
				mean_sum[j] /= T_sum;
			
			m_components[i]->set_mean(SGVector<float64_t>(mean_sum, num_dim));

			cov_sum = new float64_t[num_dim*num_dim];
			memset(cov_sum, 0, num_dim*num_dim*sizeof(float64_t));

			for (int j=0; j<num_vectors; j++)
			{
				SGVector<float64_t> v=dotdata->get_feature_vector(j);	
				CMath::add<float64_t>(v.vector, 1, v.vector, -1, mean_sum, v.vlen);
				cblas_dger(CblasRowMajor, num_dim, num_dim, T[j*m_n+i], v.vector, 1, v.vector,
                    1, (double*) cov_sum, num_dim);
				v.free_vector();
			}

			for (int j=0; j<num_dim*num_dim; j++)
				cov_sum[j] /= T_sum;

			m_components[i]->set_cov(SGMatrix<float64_t>(cov_sum, num_dim, num_dim));

			delete[] mean_sum;
			delete[] cov_sum;
		}
		iter++;
	}

	delete[] pdfs;
	delete[] T;
	return true;
}

int32_t CGMM::get_num_model_parameters()
{
	return 3;
}

float64_t CGMM::get_log_model_parameter(int32_t num_param)
{
	ASSERT(num_param<3);

	if (num_param==0)
		return CMath::log(m_n);
	else if (num_param==1)
		return CMath::log(m_max_iter);
	else
		return CMath::log(m_minimal_change);
}

float64_t CGMM::get_log_derivative(int32_t num_param, int32_t num_example)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CGMM::get_log_likelihood_example(int32_t num_example)
{
	SG_NOTIMPLEMENTED;
	return 1;
}

void CGMM::register_params()
{
	m_parameters->add_vector((CSGObject***) &m_components,
							 &m_n, "m_components", "Mixture components");
	m_parameters->add_vector(&m_coefficients, &m_coef_size, "m_coefficients", "Mixture coefficients.");
	m_parameters->add(&m_max_iter, "m_max_iter", "Maximum number of iterations.");
	m_parameters->add(&m_minimal_change, "m_minimal_change", "Minimal expected log-likelihood change.");
}

#endif
