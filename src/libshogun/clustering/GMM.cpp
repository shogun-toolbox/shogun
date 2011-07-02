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

CGMM::CGMM() : CDistribution(), m_components(),	m_coefficients()
{
	register_params();
}

CGMM::CGMM(int32_t n, ECovType cov_type) : CDistribution(), m_components(), m_coefficients()
{
	m_coefficients.vector=new float64_t[n];
	m_coefficients.vlen=n;
	m_components.vector=new CGaussian*[n];
	m_components.vlen=n;

	for (int i=0; i<n; i++)
	{
		m_components.vector[i]=new CGaussian();
		SG_REF(m_components.vector[i]);
		m_components.vector[i]->set_cov_type(cov_type);
	}

	register_params();
}

CGMM::CGMM(SGVector<CGaussian*> components, SGVector<float64_t> coefficients) : CDistribution()
{
	ASSERT(components.vlen==coefficients.vlen);

	m_components=components;
	m_coefficients=coefficients;

	register_params();
}

CGMM::~CGMM()
{
	if (m_components.vector)
		cleanup();
}

void CGMM::cleanup()
{
	for (int i = 0; i < m_components.vlen; i++)
		SG_UNREF(m_components.vector[i]);

	m_components.free_vector();
	m_coefficients.free_vector();
}

bool CGMM::train(CFeatures* data)
{
	ASSERT(m_components.vlen != 0);

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

	CKMeans* init_k_means=new CKMeans(m_components.vlen, new CEuclidianDistance());
	init_k_means->train(dotdata);
	SGMatrix<float64_t> init_means=init_k_means->get_cluster_centers();

	SGMatrix<float64_t> alpha;

	alpha.matrix=alpha_init(init_means.matrix, init_means.num_rows, init_means.num_cols);
	alpha.num_rows=num_vectors;
	alpha.num_cols=m_components.vlen;

	SG_UNREF(init_k_means);

	max_likelihood(alpha, min_cov);

	int32_t iter=0;
	float64_t log_likelihood_prev=0;
	float64_t log_likelihood_cur=0;
	float64_t* logPxy=new float64_t[num_vectors*m_components.vlen];
	float64_t* logPx=new float64_t[num_vectors];
	float64_t* logPost=new float64_t[num_vectors*m_components.vlen];

	while (iter<max_iter)
	{
		log_likelihood_prev=log_likelihood_cur;
		log_likelihood_cur=0;

		for (int i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
			for (int j=0; j<m_components.vlen; j++)
			{
				logPxy[i*m_components.vlen+j]=m_components.vector[j]->compute_log_PDF(v)+CMath::log(m_coefficients.vector[j]);
				logPx[i]+=CMath::exp(logPxy[i*m_components.vlen+j]);
			}

			logPx[i]=CMath::log(logPx[i]);
			log_likelihood_cur+=logPx[i];
			v.free_vector();

			for (int j=0; j<m_components.vlen; j++)
			{
				logPost[i*m_components.vlen+j]=logPxy[i*m_components.vlen+j]-logPx[i];
				alpha.matrix[i*m_components.vlen+j]=CMath::exp(logPost[i*m_components.vlen+j]);
			}
		}

		if (iter>0 && log_likelihood_cur-log_likelihood_prev<min_change)
			break;

		max_likelihood(alpha, min_cov);

		iter++;
	}

	delete[] logPxy;
	delete[] logPx;
	delete[] logPost;
	alpha.free_matrix();

	return true;
}

void CGMM::max_likelihood(SGMatrix<float64_t> alpha, float64_t min_cov)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_dim=dotdata->get_dim_feature_space();

	float64_t alpha_sum;
	float64_t alpha_sum_sum=0;
	float64_t* mean_sum;
	float64_t* cov_sum;

	for (int i=0; i<alpha.num_cols; i++)
	{
		alpha_sum=0;
		mean_sum=new float64_t[num_dim];
		memset(mean_sum, 0, num_dim*sizeof(float64_t));

		for (int j=0; j<alpha.num_rows; j++)
		{
			alpha_sum+=alpha.matrix[j*alpha.num_cols+i];
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(j);
			CMath::add<float64_t>(mean_sum, alpha.matrix[j*alpha.num_cols+i], v.vector, 1, mean_sum, v.vlen);
			v.free_vector();
		}

		for (int j=0; j<num_dim; j++)
			mean_sum[j]/=alpha_sum;

		m_components.vector[i]->set_mean(SGVector<float64_t>(mean_sum, num_dim));

		ECovType cov_type=m_components.vector[i]->get_cov_type();

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
			cov_sum[0]=0;
		}

		for (int j=0; j<alpha.num_rows; j++)
		{
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(j);
			CMath::add<float64_t>(v.vector, 1, v.vector, -1, mean_sum, v.vlen);

			switch (cov_type)
			{
				case FULL:
					cblas_dger(CblasRowMajor, num_dim, num_dim, alpha.matrix[j*alpha.num_cols+i], v.vector, 1, v.vector,
								 1, (double*) cov_sum, num_dim);

					break;
				case DIAG:
					for (int k=0; k<num_dim; k++)
						cov_sum[k]+=v.vector[k]*v.vector[k]*alpha.matrix[j*alpha.num_cols+i];

					break;
				case SPHERICAL:
					float64_t temp=0;

					for (int k=0; k<num_dim; k++)
						temp+=v.vector[k]*v.vector[k];

					cov_sum[0]+=temp*alpha.matrix[j*alpha.num_cols+i];
					break;
			}
			
			v.free_vector();
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

				m_components.vector[i]->set_d(SGVector<float64_t>(d0, num_dim));
				m_components.vector[i]->set_u(SGMatrix<float64_t>(cov_sum, num_dim, num_dim));

				break;
			case DIAG:
				for (int j=0; j<num_dim; j++)
				{
					cov_sum[j]/=alpha_sum;
					cov_sum[j]=CMath::max(min_cov, cov_sum[j]);
				}

				m_components.vector[i]->set_d(SGVector<float64_t>(cov_sum, num_dim));

				break;
			case SPHERICAL:
				cov_sum[0]/=alpha_sum*num_dim;
				cov_sum[0]=CMath::max(min_cov, cov_sum[0]);

				m_components.vector[i]->set_d(SGVector<float64_t>(cov_sum, 1));

				break;
		}

		m_coefficients.vector[i]=alpha_sum;
		alpha_sum_sum+=alpha_sum;
	}

	for (int i=0; i<alpha.num_cols; i++)
		m_coefficients.vector[i]/=alpha_sum_sum;
}

int32_t CGMM::get_num_model_parameters()
{
	return 1;
}

float64_t CGMM::get_log_model_parameter(int32_t num_param)
{
	ASSERT(num_param==1);

	return CMath::log(m_components.vlen);
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

	float64_t* alpha=new float64_t[num_vectors*m_components.vlen];
	memset(alpha, 0, num_vectors*m_components.vlen*sizeof(float64_t));

	for (int i=0; i<num_vectors; i++)
		alpha[i*m_components.vlen+init_labels->get_int_label(i)]=1;

	SG_UNREF(init_labels);

	return alpha;
}

SGVector<float64_t> CGMM::sample()
{
	ASSERT(m_components.vector);
	float64_t rand_num=CMath::random(float64_t(0), float64_t(1));
	float64_t cum_sum=0;
	for (int i=0; i<m_coefficients.vlen; i++)
	{
		cum_sum+=m_coefficients.vector[i];
		if (cum_sum>=rand_num)
			return m_components.vector[i]->sample();
	}
	return m_components.vector[m_coefficients.vlen-1]->sample();
}

SGVector<float64_t> CGMM::cluster(SGVector<float64_t> point)
{
	SGVector<float64_t> answer;
	answer.vector=new float64_t[m_components.vlen+1];
	answer.vlen=m_components.vlen+1;
	answer.vector[m_components.vlen]=0;

	for (int i=0; i<m_components.vlen; i++)
	{
		answer.vector[i]=m_components.vector[i]->compute_log_PDF(point)+CMath::log(m_coefficients.vector[i]);
		answer.vector[m_components.vlen]+=CMath::exp(answer.vector[i]);
	}

	return answer;
}

void CGMM::register_params()
{
	m_parameters->add((SGVector<CSGObject*>*) &m_components, "m_components", "Mixture components");
	m_parameters->add(&m_coefficients, "m_coefficients", "Mixture coefficients.");
}

#endif
