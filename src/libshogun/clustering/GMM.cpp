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
#include "features/Labels.h"
#include "classifier/KNN.h"

using namespace shogun;

CGMM::CGMM() : CDistribution(), m_components(NULL), m_n(0),
				m_coefficients(NULL), m_coef_size(0)
{
	register_params();
}

CGMM::CGMM(int32_t n_, ECovType cov_type_) : CDistribution(), m_components(NULL), m_n(n_),
						m_coefficients(NULL), m_coef_size(n_)
{
	m_coefficients=new float64_t[m_coef_size];
	m_components=new CGaussian*[m_n];

	for (int i=0; i<m_n; i++)
	{
		m_components[i]=new CGaussian();
		SG_REF(m_components[i]);
		m_components[i]->set_cov_type(cov_type_);
	}

	register_params();
}

CGMM::CGMM(CGaussian** components, int32_t components_length, float64_t* coefficients, int32_t coefficient_length) : CDistribution()
{
	ASSERT(components_length==coefficient_length);

	m_coef_size=coefficient_length;
	m_n=components_length;
	m_coefficients=new float64_t[m_coef_size];
	m_components=new CGaussian*[m_n];

	for (int i=0; i<m_n; i++)
	{
		m_components[i]=components[i];
		SG_REF(m_components[i]);
	}

	memcpy(m_coefficients, coefficients, m_coef_size*sizeof(float64_t));

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

	/** init features with data if necessary and assure type is correct */
	if (data)
	{
		if (!data->has_property(FP_DOT))
				SG_ERROR("Specified features are not of type CDotFeatures\n");		
		set_features(data);
	}

	return true;
}

bool CGMM::train_em(float64_t min_cov, int32_t max_iter, float64_t min_change)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	CKMeans* init_k_means=new CKMeans(m_n, new CEuclidianDistance());
	init_k_means->train(dotdata);
	// sorry ;)
	SGMatrix<float64_t> cluster_centers = init_k_means->get_cluster_centers();
	float64_t* init_means = cluster_centers.matrix;
	int32_t init_mean_dim = cluster_centers.num_rows;
	int32_t init_mean_size = cluster_centers.num_cols;


	float64_t* alpha;

	alpha=alpha_init(init_means, init_mean_dim, init_mean_size);

<<<<<<< HEAD
	for (int i=0; i<m_n; i++)
	{
		m_coefficients[i] = 1.0/m_coef_size;
		m_components[i] = new CGaussian(SGVector<float64_t>(&(init_means[i*init_mean_dim]), init_mean_dim),
								        SGMatrix<float64_t>(init_cov, init_cov_rows, init_cov_cols));
	}
=======
	SG_UNREF(init_k_means);
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start

	max_likelihood(alpha, num_vectors, m_n, min_cov);

	int32_t iter=0;
	float64_t log_likelihood_prev=0;
	float64_t log_likelihood_cur=0;
	float64_t* logPxy=new float64_t[num_vectors*m_n];
	float64_t* logPx=new float64_t[num_vectors];
	float64_t* logPost=new float64_t[num_vectors*m_n];
	float64_t* point;
	int32_t point_len;

	while (iter<max_iter)
	{
<<<<<<< HEAD
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
=======
		log_likelihood_prev=log_likelihood_cur;
		log_likelihood_cur=0;

		for (int i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			dotdata->get_feature_vector(&point, &point_len, i);
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start

			for (int j=0; j<m_n; j++)
			{
				logPxy[i*m_n+j]=m_components[j]->compute_log_PDF(point, point_len)+CMath::log(m_coefficients[j]);
				logPx[i]+=CMath::exp(logPxy[i*m_n+j]);
			}

			logPx[i]=CMath::log(logPx[i]);
			log_likelihood_cur+=logPx[i];
			delete[] point;

			for (int j=0; j<m_n; j++)
			{
				logPost[i*m_n+j]=logPxy[i*m_n+j]-logPx[i];
				alpha[i*m_n+j]=CMath::exp(logPost[i*m_n+j]);
			}
		}

		if (iter>0 && log_likelihood_cur-log_likelihood_prev<min_change)
			break;

		max_likelihood(alpha, num_vectors, m_n, min_cov);

		iter++;
	}

	delete[] logPxy;
	delete[] logPx;
	delete[] logPost;
	delete[] alpha;

	return true;
}

void CGMM::max_likelihood(float64_t* alpha, int32_t alpha_row, int32_t alpha_col, float64_t min_cov)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_dim=dotdata->get_dim_feature_space();

	float64_t alpha_sum;
	float64_t alpha_sum_sum=0;
	float64_t* mean_sum;
	float64_t* cov_sum;
	float64_t* point;
	int32_t point_len;

	for (int i=0; i<alpha_col; i++)
	{
		alpha_sum=0;
		mean_sum=new float64_t[num_dim];
		memset(mean_sum, 0, num_dim*sizeof(float64_t));

		for (int j=0; j<alpha_row; j++)
		{
			alpha_sum+=alpha[j*alpha_col+i];
			dotdata->get_feature_vector(&point, &point_len, j);
			CMath::add<float64_t>(mean_sum, alpha[j*alpha_col+i], point, 1, mean_sum, point_len);
			delete[] point;
		}

<<<<<<< HEAD
			for (int j=0; j<num_vectors; j++)
			{
				T_sum += T[j*m_n+i];
				SGVector<float64_t> v=dotdata->get_feature_vector(j);
				CMath::add<float64_t>(mean_sum, T[j*m_n+i], v.vector, 1, mean_sum, v.vlen);
				v.free_vector();
			}
=======
		for (int j=0; j<num_dim; j++)
			mean_sum[j]/=alpha_sum;
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start

		m_components[i]->set_mean(mean_sum, num_dim);

<<<<<<< HEAD
			for (int j=0; j<num_dim; j++)
				mean_sum[j] /= T_sum;
			
			m_components[i]->set_mean(SGVector<float64_t>(mean_sum, num_dim));
=======
		ECovType cov_type=m_components[i]->get_cov_type();
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start

		if (cov_type==FULL)
		{
			cov_sum=new float64_t[num_dim*num_dim];
			memset(cov_sum, 0, num_dim*num_dim*sizeof(float64_t));
		}
		else if(cov_type==DIAG)
		{
			cov_sum=new float64_t[num_dim];
			memset(cov_sum, 0, num_dim*sizeof(float64_t));
		}
		else
		{
			cov_sum=new float64_t[1];
			*cov_sum=0;
		}

		for (int j=0; j<alpha_row; j++)
		{
			dotdata->get_feature_vector(&point, &point_len, j);
			CMath::add<float64_t>(point, 1, point, -1, mean_sum, point_len);

			switch (cov_type)
			{
<<<<<<< HEAD
				SGVector<float64_t> v=dotdata->get_feature_vector(j);	
				CMath::add<float64_t>(v.vector, 1, v.vector, -1, mean_sum, v.vlen);
				cblas_dger(CblasRowMajor, num_dim, num_dim, T[j*m_n+i], v.vector, 1, v.vector,
                    1, (double*) cov_sum, num_dim);
				v.free_vector();
=======
				case FULL:
					cblas_dger(CblasRowMajor, num_dim, num_dim, alpha[j*alpha_col+i], point, 1, point,
								 1, (double*) cov_sum, num_dim);

					break;
				case DIAG:
					for (int k=0; k<num_dim; k++)
						cov_sum[k]+=point[k]*alpha[j*alpha_col+i];

					break;
				case SPHERICAL:
					float64_t temp=0;

					for (int k=0; k<num_dim; k++)
						temp+=point[k];

					*cov_sum+=temp*alpha[j*alpha_col+i];
					break;
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start
			}
			
			delete[] point;
		}

		switch (cov_type)
		{
			case FULL:
				for (int j=0; j<num_dim*num_dim; j++)
					cov_sum[j]/=alpha_sum;

				float64_t* d0;
				d0=CMath::compute_eigenvectors(cov_sum, num_dim, num_dim);
				for (int j=0; j<num_dim; j++)
					d0[j]=CMath::max(min_cov, d0[j]);

				m_components[i]->set_d(d0, num_dim);
				m_components[i]->set_u(cov_sum, num_dim, num_dim);
				delete[] d0;

				break;
			case DIAG:
				for (int j=0; j<num_dim; j++)
					cov_sum[j]/=alpha_sum;

<<<<<<< HEAD
			m_components[i]->set_cov(SGMatrix<float64_t>(cov_sum, num_dim, num_dim));
=======
				m_components[i]->set_d(cov_sum, num_dim);
>>>>>>> Rewrote the GMM file to use covariance types and decomposition, optimized start

				break;
			case SPHERICAL:
				*cov_sum/=alpha_sum*num_dim;

				m_components[i]->set_d(cov_sum, 1);

				break;
		}

		m_coefficients[i]=alpha_sum;
		alpha_sum_sum+=alpha_sum;

		delete[] mean_sum;
		delete[] cov_sum;
	}

	for (int i=0; i<alpha_col; i++)
		m_coefficients[i]/=alpha_sum_sum;
}

int32_t CGMM::get_num_model_parameters()
{
	return 1;
}

float64_t CGMM::get_log_model_parameter(int32_t num_param)
{
	ASSERT(num_param==1);

	return CMath::log(m_n);
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

float64_t* CGMM::alpha_init(float64_t* init_means, int32_t init_mean_dim, int32_t init_mean_size)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	float64_t* label_num=new float64_t[init_mean_size];

	for (int i=0; i<init_mean_size; i++)
		label_num[i]=i;

	CKNN* knn=new CKNN(1, new CEuclidianDistance(), new CLabels(label_num, init_mean_size));
	knn->train(new CSimpleFeatures<float64_t>(init_means, init_mean_dim, init_mean_size));
	CLabels* init_labels=knn->apply(features);

	float64_t* alpha=new float64_t[num_vectors*m_n];
	memset(alpha, 0, num_vectors*m_n*sizeof(float64_t));

	for (int i=0; i<num_vectors; i++)
		alpha[i*m_n+init_labels->get_int_label(i)]=1;

	SG_UNREF(knn);
	SG_UNREF(init_labels);
	delete[] label_num;

	return alpha;
}

void CGMM::sample(float64_t** samp, int32_t* samp_length)
{
	ASSERT(m_components);
	float64_t rand_num=CMath::random(float64_t(0), float64_t(1));
	float64_t cum_sum=0;
	for (int i=0; i<m_coef_size; i++)
	{
		cum_sum+=m_coefficients[i];
		if (cum_sum>=rand_num)
		{
			m_components[i]->sample(samp, samp_length);
			return;
		}
	}
}

void CGMM::register_params()
{
	m_parameters->add_vector((CSGObject***) &m_components,
							 &m_n, "m_components", "Mixture components");
	m_parameters->add_vector(&m_coefficients, &m_coef_size, "m_coefficients", "Mixture coefficients.");
}

#endif
