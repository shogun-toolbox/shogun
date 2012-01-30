/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/LocalTangentSpaceAlignment.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct LTSA_THREAD_PARAM
{
	/// starting index of loop
	int32_t idx_start;
	/// step of loop
	int32_t idx_step;
	/// stop index of loop
	int32_t idx_stop;
	/// number of neighbors
	int32_t m_k;
	/// target dimension
	int32_t target_dim;
	/// matrix containing indexes of ith vector's neighbors in ith column
	SGMatrix<int32_t> neighborhood_matrix;
	/// G matrix
	float64_t* G_matrix;
	/// mean vector
	float64_t* mean_vector;
	/// local feature matrix containing neighbors of vector
	float64_t* local_feature_matrix;
	/// feature matrix of given features instance
	SGMatrix<float64_t> feature_matrix;
	/// used to store singular values
	float64_t* s_values_vector;
	/// q matrix
	float64_t* q_matrix;
	/// weight matrix
	float64_t* W_matrix;
#ifdef HAVE_PTHREAD
	/// lock used on modifying to weight matrix
	PTHREAD_LOCK_T* W_matrix_lock;
#endif
};
#endif

CLocalTangentSpaceAlignment::CLocalTangentSpaceAlignment() :
		CLocallyLinearEmbedding()
{
}

CLocalTangentSpaceAlignment::~CLocalTangentSpaceAlignment()
{
}

const char* CLocalTangentSpaceAlignment::get_name() const
{ 
	return "LocalTangentSpaceAlignment"; 
};

SGMatrix<float64_t> CLocalTangentSpaceAlignment::construct_weight_matrix(CSimpleFeatures<float64_t>* simple_features, float64_t* W_matrix,
                                                                         SGMatrix<int32_t> neighborhood_matrix)
{
	int32_t N = simple_features->get_num_vectors();
	int32_t dim = simple_features->get_num_features();
	int32_t t;
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads and params
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	LTSA_THREAD_PARAM* parameters = SG_MALLOC(LTSA_THREAD_PARAM, num_threads);
#else
	int32_t num_threads = 1;
#endif

	// init matrices and norm factor to be used
	float64_t* local_feature_matrix = SG_MALLOC(float64_t, m_k*dim*num_threads);
	float64_t* mean_vector = SG_MALLOC(float64_t, dim*num_threads);
	float64_t* q_matrix = SG_MALLOC(float64_t, m_k*m_k*num_threads);
	float64_t* s_values_vector = SG_MALLOC(float64_t, dim*num_threads);
	float64_t* G_matrix = SG_MALLOC(float64_t, m_k*(1+m_target_dim)*num_threads);
	
	// get feature matrix
	SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();

#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T W_matrix_lock;
	pthread_attr_t attr;
	PTHREAD_LOCK_INIT(&W_matrix_lock);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (t=0; t<num_threads; t++)
	{
		parameters[t].idx_start = t;
		parameters[t].idx_step = num_threads;
		parameters[t].idx_stop = N;
		parameters[t].m_k = m_k;
		parameters[t].target_dim = m_target_dim;
		parameters[t].neighborhood_matrix = neighborhood_matrix;
		parameters[t].G_matrix = G_matrix + (m_k*(1+m_target_dim))*t;
		parameters[t].mean_vector = mean_vector + dim*t;
		parameters[t].local_feature_matrix = local_feature_matrix + (m_k*dim)*t;
		parameters[t].feature_matrix = feature_matrix;
		parameters[t].s_values_vector = s_values_vector + dim*t;
		parameters[t].q_matrix = q_matrix + (m_k*m_k)*t;
		parameters[t].W_matrix = W_matrix;
		parameters[t].W_matrix_lock = &W_matrix_lock;
		pthread_create(&threads[t], &attr, run_ltsa_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	PTHREAD_LOCK_DESTROY(&W_matrix_lock);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	LTSA_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_step = 1;
	single_thread_param.idx_stop = N;
	single_thread_param.m_k = m_k;
	single_thread_param.target_dim = m_target_dim;
	single_thread_param.neighborhood_matrix = neighborhood_matrix;
	single_thread_param.G_matrix = G_matrix;
	single_thread_param.mean_vector = mean_vector;
	single_thread_param.local_feature_matrix = local_feature_matrix;
	single_thread_param.feature_matrix = feature_matrix;
	single_thread_param.s_values_vector = s_values_vector;
	single_thread_param.q_matrix = q_matrix;
	single_thread_param.W_matrix = W_matrix;
	run_ltsa_thread((void*)&single_thread_param);
#endif

	// clean
	SG_FREE(G_matrix);
	SG_FREE(s_values_vector);
	SG_FREE(mean_vector);
	SG_FREE(local_feature_matrix);
	SG_FREE(q_matrix);

	int32_t actual_k = neighborhood_matrix.num_rows;
	for (int32_t i=0; i<N; i++)
	{
		for (int32_t j=0; j<m_k; j++)
			W_matrix[N*neighborhood_matrix[i*actual_k+j]+neighborhood_matrix[i*actual_k+j]] += 1.0;
	}

	return SGMatrix<float64_t>(W_matrix,N,N);
}

void* CLocalTangentSpaceAlignment::run_ltsa_thread(void* p)
{
	LTSA_THREAD_PARAM* parameters = (LTSA_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t m_k = parameters->m_k;
	int32_t target_dim = parameters->target_dim;
	SGMatrix<int32_t> neighborhood_matrix = parameters->neighborhood_matrix;
	float64_t* G_matrix = parameters->G_matrix;
	float64_t* mean_vector = parameters->mean_vector;
	float64_t* local_feature_matrix = parameters->local_feature_matrix;
	SGMatrix<float64_t> feature_matrix = parameters->feature_matrix;
	float64_t* s_values_vector = parameters->s_values_vector;
	float64_t* q_matrix = parameters->q_matrix;
	float64_t* W_matrix = parameters->W_matrix;
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T* W_matrix_lock = parameters->W_matrix_lock;
#endif

	int32_t i,j,k;
	int32_t N = feature_matrix.num_cols;
	int32_t dim = feature_matrix.num_rows;
	int32_t actual_k = neighborhood_matrix.num_rows;

	for (j=0; j<m_k; j++)
		G_matrix[j] = 1.0/CMath::sqrt((float64_t)m_k);

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		// fill mean vector with zeros
		memset(mean_vector,0,sizeof(float64_t)*dim);

		// compute local feature matrix containing neighbors of i-th vector
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
				local_feature_matrix[j*dim+k] = feature_matrix[neighborhood_matrix[i*actual_k+j]*dim+k];

			cblas_daxpy(dim,1.0,local_feature_matrix+j*dim,1,mean_vector,1);
		}

		// compute mean
		cblas_dscal(dim,1.0/m_k,mean_vector,1);

		// center feature vectors by mean
		for (j=0; j<m_k; j++)
			cblas_daxpy(dim,-1.0,mean_vector,1,local_feature_matrix+j*dim,1);

		int32_t info = 0;
		// find right eigenvectors of local_feature_matrix
		wrap_dgesvd('N','O',dim,m_k,local_feature_matrix,dim,
		            s_values_vector,NULL,1, NULL,1,&info);
		ASSERT(info==0);
		
		for (j=0; j<target_dim; j++)
		{
			for (k=0; k<m_k; k++)
				G_matrix[(j+1)*m_k+k] = local_feature_matrix[k*dim+j];
		}

		// compute GG'
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
		            m_k,m_k,1+target_dim,
		            1.0,G_matrix,m_k,
		                G_matrix,m_k,
		            0.0,q_matrix,m_k);
		
		// W[neighbors of i, neighbors of i] = I - GG'
#ifdef HAVE_PTHREAD
		PTHREAD_LOCK(W_matrix_lock);
#endif
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
				W_matrix[N*neighborhood_matrix[i*actual_k+k]+neighborhood_matrix[i*actual_k+j]] -= q_matrix[j*m_k+k];
		}
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(W_matrix_lock);
#endif
	}
	return NULL;
}

#endif /* HAVE_LAPACK */
