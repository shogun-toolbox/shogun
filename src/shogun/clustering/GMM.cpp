/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Alesis Novik
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include <shogun/lib/config.h>

#ifdef HAVE_LAPACK

#include <shogun/clustering/GMM.h>
#include <shogun/clustering/KMeans.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/features/Labels.h>
#include <shogun/multiclass/KNN.h>

using namespace shogun;

CGMM::CGMM() : CDistribution(), m_components(),	m_coefficients()
{
	register_params();
}

CGMM::CGMM(int32_t n, ECovType cov_type) : CDistribution(), m_components(), m_coefficients()
{
	m_coefficients.vector=SG_MALLOC(float64_t, n);
	m_coefficients.vlen=n;
	m_components.vector=SG_MALLOC(CGaussian*, n);
	m_components.vlen=n;

	for (int32_t i=0; i<n; i++)
	{
		m_components.vector[i]=new CGaussian();
		SG_REF(m_components.vector[i]);
		m_components.vector[i]->set_cov_type(cov_type);
	}

	register_params();
}

CGMM::CGMM(const SGVector<CGaussian*>& components, const SGVector<float64_t>& coefficients, bool copy) : CDistribution()
{
	ASSERT(components.vlen==coefficients.vlen);

	if (!copy)
	{
		m_components=components;
		m_coefficients=coefficients;
		for (int32_t i=0; i<components.vlen; i++)
		{
			SG_REF(m_components.vector[i]);
		}
	}
	else
	{
		m_coefficients.vector=SG_MALLOC(float64_t, components.vlen);
		m_coefficients.vlen=components.vlen;
		m_components.vector=SG_MALLOC(CGaussian*, components.vlen);
		m_components.vlen=components.vlen;

		for (int32_t i=0; i<components.vlen; i++)
		{
			m_components.vector[i]=new CGaussian();
			SG_REF(m_components.vector[i]);
			m_components.vector[i]->set_cov_type(components.vector[i]->get_cov_type());

			SGVector<float64_t> old_mean=components.vector[i]->get_mean();
			SGVector<float64_t> new_mean(old_mean.vlen);
			memcpy(new_mean.vector, old_mean.vector, old_mean.vlen*sizeof(float64_t));
			m_components.vector[i]->set_mean(new_mean);

			SGVector<float64_t> old_d=components.vector[i]->get_d();
			SGVector<float64_t> new_d(old_d.vlen);
			memcpy(new_d.vector, old_d.vector, old_d.vlen*sizeof(float64_t));
			m_components.vector[i]->set_d(new_d);

			if (components.vector[i]->get_cov_type()==FULL)
			{
				SGMatrix<float64_t> old_u=components.vector[i]->get_u();
				SGMatrix<float64_t> new_u(old_u.num_rows, old_u.num_cols);
				memcpy(new_u.matrix, old_u.matrix, old_u.num_rows*old_u.num_cols*sizeof(float64_t));
				m_components.vector[i]->set_u(new_u);
			}

			m_coefficients.vector[i]=coefficients.vector[i];
		}
	}

	register_params();
}

CGMM::~CGMM()
{
	if (m_components.vector)
		cleanup();
}

void CGMM::cleanup()
{
	for (int32_t i = 0; i < m_components.vlen; i++)
		SG_UNREF(m_components.vector[i]);

	m_components.unref();
	m_coefficients.unref();
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

float64_t CGMM::train_em(float64_t min_cov, int32_t max_iter, float64_t min_change)
{
	if (!features)
		SG_ERROR("No features to train on.\n");

	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	SGMatrix<float64_t> alpha;

	if (m_components.vector[0]->get_mean().vector==NULL)
	{
		CKMeans* init_k_means=new CKMeans(m_components.vlen, new CEuclidianDistance());
		init_k_means->train(dotdata);
		SGMatrix<float64_t> init_means=init_k_means->get_cluster_centers();

		alpha=alpha_init(init_means);

		SG_UNREF(init_k_means);

		max_likelihood(alpha, min_cov);
	}
	else
	{
		alpha.matrix=SG_MALLOC(float64_t, num_vectors*m_components.vlen);
		alpha.num_rows=num_vectors;
		alpha.num_cols=m_components.vlen;
	}

	int32_t iter=0;
	float64_t log_likelihood_prev=0;
	float64_t log_likelihood_cur=0;
	float64_t* logPxy=SG_MALLOC(float64_t, num_vectors*m_components.vlen);
	float64_t* logPx=SG_MALLOC(float64_t, num_vectors);
	//float64_t* logPost=SG_MALLOC(float64_t, num_vectors*m_components.vlen);

	while (iter<max_iter)
	{
		log_likelihood_prev=log_likelihood_cur;
		log_likelihood_cur=0;

		for (int32_t i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
			for (int32_t j=0; j<m_components.vlen; j++)
			{
				logPxy[i*m_components.vlen+j]=m_components.vector[j]->compute_log_PDF(v)+CMath::log(m_coefficients.vector[j]);
				logPx[i]+=CMath::exp(logPxy[i*m_components.vlen+j]);
			}

			logPx[i]=CMath::log(logPx[i]);
			log_likelihood_cur+=logPx[i];

			for (int32_t j=0; j<m_components.vlen; j++)
			{
				//logPost[i*m_components.vlen+j]=logPxy[i*m_components.vlen+j]-logPx[i];
				alpha.matrix[i*m_components.vlen+j]=CMath::exp(logPxy[i*m_components.vlen+j]-logPx[i]);
			}
		}

		if (iter>0 && log_likelihood_cur-log_likelihood_prev<min_change)
			break;

		max_likelihood(alpha, min_cov);

		iter++;
	}

	SG_FREE(logPxy);
	SG_FREE(logPx);
	//SG_FREE(logPost);
	alpha.free_matrix();

	return log_likelihood_cur;
}

float64_t CGMM::train_smem(int32_t max_iter, int32_t max_cand, float64_t min_cov, int32_t max_em_iter, float64_t min_change)
{
	if (!features)
		SG_ERROR("No features to train on.\n");

	if (m_components.vlen<3)
		SG_ERROR("Can't run SMEM with less than 3 component mixture model.\n");

	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	float64_t cur_likelihood=train_em(min_cov, max_em_iter, min_change);

	int32_t iter=0;
	float64_t* logPxy=SG_MALLOC(float64_t, num_vectors*m_components.vlen);
	float64_t* logPx=SG_MALLOC(float64_t, num_vectors);
	float64_t* logPost=SG_MALLOC(float64_t, num_vectors*m_components.vlen);
	float64_t* logPostSum=SG_MALLOC(float64_t, m_components.vlen);
	float64_t* logPostSum2=SG_MALLOC(float64_t, m_components.vlen);
	float64_t* logPostSumSum=SG_MALLOC(float64_t, m_components.vlen*(m_components.vlen-1)/2);
	float64_t* split_crit=SG_MALLOC(float64_t, m_components.vlen);
	float64_t* merge_crit=SG_MALLOC(float64_t, m_components.vlen*(m_components.vlen-1)/2);
	int32_t* split_ind=SG_MALLOC(int32_t, m_components.vlen);
	int32_t* merge_ind=SG_MALLOC(int32_t, m_components.vlen*(m_components.vlen-1)/2);

	while (iter<max_iter)
	{
		memset(logPostSum, 0, m_components.vlen*sizeof(float64_t));
		memset(logPostSum2, 0, m_components.vlen*sizeof(float64_t));
		memset(logPostSumSum, 0, (m_components.vlen*(m_components.vlen-1)/2)*sizeof(float64_t));
		for (int32_t i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
			for (int32_t j=0; j<m_components.vlen; j++)
			{
				logPxy[i*m_components.vlen+j]=m_components.vector[j]->compute_log_PDF(v)+CMath::log(m_coefficients.vector[j]);
				logPx[i]+=CMath::exp(logPxy[i*m_components.vlen+j]);
			}

			logPx[i]=CMath::log(logPx[i]);

			for (int32_t j=0; j<m_components.vlen; j++)
			{
				logPost[i*m_components.vlen+j]=logPxy[i*m_components.vlen+j]-logPx[i];
				logPostSum[j]+=CMath::exp(logPost[i*m_components.vlen+j]);
				logPostSum2[j]+=CMath::exp(2*logPost[i*m_components.vlen+j]);
			}

			int32_t counter=0;
			for (int32_t j=0; j<m_components.vlen; j++)
			{
				for (int32_t k=j+1; k<m_components.vlen; k++)
				{
					logPostSumSum[counter]+=CMath::exp(logPost[i*m_components.vlen+j]+logPost[i*m_components.vlen+k]);
					counter++;
				}
			}
		}

		int32_t counter=0;
		for (int32_t i=0; i<m_components.vlen; i++)
		{
			logPostSum[i]=CMath::log(logPostSum[i]);
			split_crit[i]=0;
			split_ind[i]=i;
			for (int32_t j=0; j<num_vectors; j++)
			{
				split_crit[i]+=(logPost[j*m_components.vlen+i]-logPostSum[i]-logPxy[j*m_components.vlen+i]+CMath::log(m_coefficients.vector[i]))*
								(CMath::exp(logPost[j*m_components.vlen+i])/CMath::exp(logPostSum[i]));
			}
			for (int32_t j=i+1; j<m_components.vlen; j++)
			{
				merge_crit[counter]=CMath::log(logPostSumSum[counter])-(0.5*CMath::log(logPostSum2[i]))-(0.5*CMath::log(logPostSum2[j]));
				merge_ind[counter]=i*m_components.vlen+j;
				counter++;
			}
		}
		CMath::qsort_backward_index(split_crit, split_ind, m_components.vlen);
		CMath::qsort_backward_index(merge_crit, merge_ind, m_components.vlen*(m_components.vlen-1)/2);

		bool better_found=false;
		int32_t candidates_checked=0;
		for (int32_t i=0; i<m_components.vlen; i++)
		{
			for (int32_t j=0; j<m_components.vlen*(m_components.vlen-1)/2; j++)
			{
				if (merge_ind[j]/m_components.vlen != split_ind[i] && merge_ind[j]%m_components.vlen != split_ind[i])
				{
					candidates_checked++;
					CGMM* candidate=new CGMM(m_components, m_coefficients, true);
					candidate->train(features);
					candidate->partial_em(split_ind[i], merge_ind[j]/m_components.vlen, merge_ind[j]%m_components.vlen, min_cov, max_em_iter, min_change);
					float64_t cand_likelihood=candidate->train_em(min_cov, max_em_iter, min_change);

					if (cand_likelihood>cur_likelihood)
					{
						cur_likelihood=cand_likelihood;
						set_comp(candidate->get_comp());
						set_coef(candidate->get_coef());

						for (int32_t k=0; k<candidate->get_comp().vlen; k++)
						{
							SG_UNREF(candidate->get_comp().vector[k]);
						}

						better_found=true;
						break;
					}
					else
						delete candidate;

					if (candidates_checked>=max_cand)
						break;
				}
			}
			if (better_found || candidates_checked>=max_cand)
				break;
		}
		if (!better_found)
			break;
		iter++;
	}

	SG_FREE(logPxy);
	SG_FREE(logPx);
	SG_FREE(logPost);
	SG_FREE(split_crit);
	SG_FREE(merge_crit);
	SG_FREE(logPostSum);
	SG_FREE(logPostSum2);
	SG_FREE(logPostSumSum);
	SG_FREE(split_ind);
	SG_FREE(merge_ind);

	return cur_likelihood;
}

void CGMM::partial_em(int32_t comp1, int32_t comp2, int32_t comp3, float64_t min_cov, int32_t max_em_iter, float64_t min_change)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	float64_t* init_logPxy=SG_MALLOC(float64_t, num_vectors*m_components.vlen);
	float64_t* init_logPx=SG_MALLOC(float64_t, num_vectors);
	float64_t* init_logPx_fix=SG_MALLOC(float64_t, num_vectors);
	float64_t* post_add=SG_MALLOC(float64_t, num_vectors);

	for (int32_t i=0; i<num_vectors; i++)
	{
		init_logPx[i]=0;
		init_logPx_fix[i]=0;

		SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
		for (int32_t j=0; j<m_components.vlen; j++)
		{
			init_logPxy[i*m_components.vlen+j]=m_components.vector[j]->compute_log_PDF(v)+CMath::log(m_coefficients.vector[j]);
			init_logPx[i]+=CMath::exp(init_logPxy[i*m_components.vlen+j]);
			if (j!=comp1 && j!=comp2 && j!=comp3)
			{
				init_logPx_fix[i]+=CMath::exp(init_logPxy[i*m_components.vlen+j]);
			}
		}

		init_logPx[i]=CMath::log(init_logPx[i]);
		post_add[i]=CMath::log(CMath::exp(init_logPxy[i*m_components.vlen+comp1]-init_logPx[i])+
					CMath::exp(init_logPxy[i*m_components.vlen+comp2]-init_logPx[i])+
					CMath::exp(init_logPxy[i*m_components.vlen+comp3]-init_logPx[i]));
	}

	SGVector<CGaussian*> components(3);
	SGVector<float64_t> coefficients(3);
	components.vector[0]=m_components.vector[comp1];
	components.vector[1]=m_components.vector[comp2];
	components.vector[2]=m_components.vector[comp3];
	coefficients.vector[0]=m_coefficients.vector[comp1];
	coefficients.vector[1]=m_coefficients.vector[comp2];
	coefficients.vector[2]=m_coefficients.vector[comp3];
	float64_t coef_sum=coefficients.vector[0]+coefficients.vector[1]+coefficients.vector[2];

	int32_t dim_n=components.vector[0]->get_mean().vlen;

	float64_t alpha1=coefficients.vector[1]/(coefficients.vector[1]+coefficients.vector[2]);
	float64_t alpha2=coefficients.vector[2]/(coefficients.vector[1]+coefficients.vector[2]);

	float64_t noise_mag=CMath::twonorm(components.vector[0]->get_mean().vector, dim_n)*0.1/
						CMath::sqrt((float64_t)dim_n);

	CMath::add(components.vector[1]->get_mean().vector, alpha1, components.vector[1]->get_mean().vector, alpha2,
				components.vector[2]->get_mean().vector, dim_n);

	for (int32_t i=0; i<dim_n; i++)
	{
		components.vector[2]->get_mean().vector[i]=components.vector[0]->get_mean().vector[i]+CMath::randn_double()*noise_mag;
		components.vector[0]->get_mean().vector[i]=components.vector[0]->get_mean().vector[i]+CMath::randn_double()*noise_mag;
	}

	coefficients.vector[1]=coefficients.vector[1]+coefficients.vector[2];
	coefficients.vector[2]=coefficients.vector[0]*0.5;
	coefficients.vector[0]=coefficients.vector[0]*0.5;

	if (components.vector[0]->get_cov_type()==FULL)
	{
		SGMatrix<float64_t> c1=components.vector[1]->get_cov();
		SGMatrix<float64_t> c2=components.vector[2]->get_cov();
		CMath::add(c1.matrix, alpha1, c1.matrix, alpha2, c2.matrix, dim_n*dim_n);

		components.vector[1]->set_d(SGVector<float64_t>(CMath::compute_eigenvectors(c1.matrix, dim_n, dim_n), dim_n));
		components.vector[1]->set_u(c1);

		c2.destroy_matrix();

		float64_t new_d=0;
		for (int32_t i=0; i<dim_n; i++)
		{
			new_d+=CMath::log(components.vector[0]->get_d().vector[i]);
			for (int32_t j=0; j<dim_n; j++)
			{
				if (i==j)
				{
					components.vector[0]->get_u().matrix[i*dim_n+j]=1;
					components.vector[2]->get_u().matrix[i*dim_n+j]=1;
				}
				else
				{
					components.vector[0]->get_u().matrix[i*dim_n+j]=0;
					components.vector[2]->get_u().matrix[i*dim_n+j]=0;
				}
			}
		}
		new_d=CMath::exp(new_d*(1./dim_n));
		for (int32_t i=0; i<dim_n; i++)
		{
			components.vector[0]->get_d().vector[i]=new_d;
			components.vector[2]->get_d().vector[i]=new_d;
		}
	}
	else if(components.vector[0]->get_cov_type()==DIAG)
	{
		CMath::add(components.vector[1]->get_d().vector, alpha1, components.vector[1]->get_d().vector,
					alpha2, components.vector[2]->get_d().vector, dim_n);

		float64_t new_d=0;
		for (int32_t i=0; i<dim_n; i++)
		{
			new_d+=CMath::log(components.vector[0]->get_d().vector[i]);
		}
		new_d=CMath::exp(new_d*(1./dim_n));
		for (int32_t i=0; i<dim_n; i++)
		{
			components.vector[0]->get_d().vector[i]=new_d;
			components.vector[2]->get_d().vector[i]=new_d;
		}
	}
	else if(components.vector[0]->get_cov_type()==SPHERICAL)
	{
		components.vector[1]->get_d().vector[0]=alpha1*components.vector[1]->get_d().vector[0]+
												alpha2*components.vector[2]->get_d().vector[0];

		components.vector[2]->get_d().vector[0]=components.vector[0]->get_d().vector[0];
	}

	CGMM* partial_candidate=new CGMM(components, coefficients);
	partial_candidate->train(features);

	float64_t log_likelihood_prev=0;
	float64_t log_likelihood_cur=0;
	int32_t iter=0;
	SGMatrix<float64_t> alpha(num_vectors, 3);
	float64_t* logPxy=SG_MALLOC(float64_t, num_vectors*3);
	float64_t* logPx=SG_MALLOC(float64_t, num_vectors);
	//float64_t* logPost=SG_MALLOC(float64_t, num_vectors*m_components.vlen);

	while (iter<max_em_iter)
	{
		log_likelihood_prev=log_likelihood_cur;
		log_likelihood_cur=0;

		for (int32_t i=0; i<num_vectors; i++)
		{
			logPx[i]=0;
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(i);
			for (int32_t j=0; j<3; j++)
			{
				logPxy[i*3+j]=components.vector[j]->compute_log_PDF(v)+CMath::log(coefficients.vector[j]);
				logPx[i]+=CMath::exp(logPxy[i*3+j]);
			}

			logPx[i]=CMath::log(logPx[i]+init_logPx_fix[i]);
			log_likelihood_cur+=logPx[i];

			for (int32_t j=0; j<3; j++)
			{
				//logPost[i*m_components.vlen+j]=logPxy[i*m_components.vlen+j]-logPx[i];
				alpha.matrix[i*3+j]=CMath::exp(logPxy[i*3+j]-logPx[i]+post_add[i]);
			}
		}

		if (iter>0 && log_likelihood_cur-log_likelihood_prev<min_change)
			break;

		partial_candidate->max_likelihood(alpha, min_cov);
		partial_candidate->get_coef().vector[0]=partial_candidate->get_coef().vector[0]*coef_sum;
		partial_candidate->get_coef().vector[1]=partial_candidate->get_coef().vector[1]*coef_sum;
		partial_candidate->get_coef().vector[2]=partial_candidate->get_coef().vector[2]*coef_sum;
		iter++;
	}

	m_coefficients.vector[comp1]=coefficients.vector[0];
	m_coefficients.vector[comp2]=coefficients.vector[1];
	m_coefficients.vector[comp3]=coefficients.vector[2];

	delete partial_candidate;
	alpha.destroy_matrix();
	SG_FREE(logPxy);
	SG_FREE(logPx);
	SG_FREE(init_logPxy);
	SG_FREE(init_logPx);
	SG_FREE(init_logPx_fix);
	SG_FREE(post_add);
}

void CGMM::max_likelihood(SGMatrix<float64_t> alpha, float64_t min_cov)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_dim=dotdata->get_dim_feature_space();

	float64_t alpha_sum;
	float64_t alpha_sum_sum=0;
	float64_t* mean_sum;
	float64_t* cov_sum=NULL;

	for (int32_t i=0; i<alpha.num_cols; i++)
	{
		alpha_sum=0;
		mean_sum=SG_MALLOC(float64_t, num_dim);
		memset(mean_sum, 0, num_dim*sizeof(float64_t));

		for (int32_t j=0; j<alpha.num_rows; j++)
		{
			alpha_sum+=alpha.matrix[j*alpha.num_cols+i];
			SGVector<float64_t> v=dotdata->get_computed_dot_feature_vector(j);
			CMath::add<float64_t>(mean_sum, alpha.matrix[j*alpha.num_cols+i], v.vector, 1, mean_sum, v.vlen);
		}

		for (int32_t j=0; j<num_dim; j++)
			mean_sum[j]/=alpha_sum;

		m_components.vector[i]->set_mean(SGVector<float64_t>(mean_sum, num_dim));

		ECovType cov_type=m_components.vector[i]->get_cov_type();

		if (cov_type==FULL)
		{
			cov_sum=SG_MALLOC(float64_t, num_dim*num_dim);
			memset(cov_sum, 0, num_dim*num_dim*sizeof(float64_t));
		}
		else if(cov_type==DIAG)
		{
			cov_sum=SG_MALLOC(float64_t, num_dim);
			memset(cov_sum, 0, num_dim*sizeof(float64_t));
		}
		else if(cov_type==SPHERICAL)
		{
			cov_sum=SG_MALLOC(float64_t, 1);
			cov_sum[0]=0;
		}

		for (int32_t j=0; j<alpha.num_rows; j++)
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
					for (int32_t k=0; k<num_dim; k++)
						cov_sum[k]+=v.vector[k]*v.vector[k]*alpha.matrix[j*alpha.num_cols+i];

					break;
				case SPHERICAL:
					float64_t temp=0;

					for (int32_t k=0; k<num_dim; k++)
						temp+=v.vector[k]*v.vector[k];

					cov_sum[0]+=temp*alpha.matrix[j*alpha.num_cols+i];
					break;
			}
		}

		switch (cov_type)
		{
			case FULL:
				for (int32_t j=0; j<num_dim*num_dim; j++)
					cov_sum[j]/=alpha_sum;

				float64_t* d0;
				d0=CMath::compute_eigenvectors(cov_sum, num_dim, num_dim);
				for (int32_t j=0; j<num_dim; j++)
					d0[j]=CMath::max(min_cov, d0[j]);

				m_components.vector[i]->set_d(SGVector<float64_t>(d0, num_dim));
				m_components.vector[i]->set_u(SGMatrix<float64_t>(cov_sum, num_dim, num_dim));

				break;
			case DIAG:
				for (int32_t j=0; j<num_dim; j++)
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

	for (int32_t i=0; i<alpha.num_cols; i++)
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

float64_t CGMM::get_likelihood_example(int32_t num_example)
{
	return CMath::exp(get_log_likelihood_example(num_example));
}

SGVector<float64_t> CGMM::get_nth_mean(int32_t num)
{
	ASSERT(num<m_components.vlen);
	return m_components.vector[num]->get_mean();
}

void CGMM::set_nth_mean(const SGVector<float64_t>& mean, int32_t num)
{
	ASSERT(num<m_components.vlen);
	m_components.vector[num]->set_mean(mean);
}

SGMatrix<float64_t> CGMM::get_nth_cov(int32_t num)
{
	ASSERT(num<m_components.vlen);
	return m_components.vector[num]->get_cov();
}

void CGMM::set_nth_cov(SGMatrix<float64_t> cov, int32_t num)
{
	ASSERT(num<m_components.vlen);
	m_components.vector[num]->set_cov(cov);
}

SGVector<float64_t> CGMM::get_coef()
{
	return m_coefficients;
}

void CGMM::set_coef(const SGVector<float64_t> coefficients)
{
	m_coefficients=coefficients;
}

SGVector<CGaussian*> CGMM::get_comp()
{
	return m_components;
}

void CGMM::set_comp(const SGVector<CGaussian*>& components)
{
	for (int32_t i=0; i<m_components.vlen; i++)
	{
		SG_UNREF(m_components.vector[i]);
	}

	m_components=components;

	for (int32_t i=0; i<m_components.vlen; i++)
	{
		SG_REF(m_components.vector[i]);
	}
}

SGMatrix<float64_t> CGMM::alpha_init(SGMatrix<float64_t> init_means)
{
	CDotFeatures* dotdata=(CDotFeatures *) features;
	int32_t num_vectors=dotdata->get_num_vectors();

	SGVector<float64_t> label_num(init_means.num_cols);

	for (int32_t i=0; i<init_means.num_cols; i++)
		label_num.vector[i]=i;

	CKNN* knn=new CKNN(1, new CEuclidianDistance(), new CLabels(label_num));
	knn->train(new CSimpleFeatures<float64_t>(init_means));
	CLabels* init_labels=knn->apply(features);

	SGMatrix<float64_t> alpha(num_vectors, m_components.vlen);
	memset(alpha.matrix, 0, num_vectors*m_components.vlen*sizeof(float64_t));

	for (int32_t i=0; i<num_vectors; i++)
		alpha.matrix[i*m_components.vlen+init_labels->get_int_label(i)]=1;

	SG_UNREF(init_labels);

	return alpha;
}

SGVector<float64_t> CGMM::sample()
{
	ASSERT(m_components.vector);
	float64_t rand_num=CMath::random(float64_t(0), float64_t(1));
	float64_t cum_sum=0;
	for (int32_t i=0; i<m_coefficients.vlen; i++)
	{
		cum_sum+=m_coefficients.vector[i];
		if (cum_sum>=rand_num)
			return m_components.vector[i]->sample();
	}
	return m_components.vector[m_coefficients.vlen-1]->sample();
}

SGVector<float64_t> CGMM::cluster(const SGVector<float64_t>& point)
{
	SGVector<float64_t> answer(m_components.vlen+1);
	answer.vector[m_components.vlen]=0;

	for (int32_t i=0; i<m_components.vlen; i++)
	{
		answer.vector[i]=m_components.vector[i]->compute_log_PDF(point)+CMath::log(m_coefficients.vector[i]);
		answer.vector[m_components.vlen]+=CMath::exp(answer.vector[i]);
	}
	answer.vector[m_components.vlen]=CMath::log(answer.vector[m_components.vlen]);

	return answer;
}

void CGMM::register_params()
{
	m_parameters->add((SGVector<CSGObject*>*) &m_components, "m_components", "Mixture components");
	m_parameters->add(&m_coefficients, "m_coefficients", "Mixture coefficients.");
}

#endif
