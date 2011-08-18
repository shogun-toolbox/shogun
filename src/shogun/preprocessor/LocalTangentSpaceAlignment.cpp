/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/LocalTangentSpaceAlignment.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/lib/Signal.h>

#ifndef WIN32
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct D_THREAD_PARAM
{
	int32_t idx_start;
	int32_t idx_step;
	int32_t idx_stop;
	int32_t m_k;
	int32_t m_target_dim;
	int32_t dim;
	int32_t N;
	const int32_t* neighborhood_matrix;
	float64_t* G_matrix;
	float64_t* mean_vector;
	float64_t* local_feature_matrix;
	const float64_t* feature_matrix;
	float64_t* s_values_vector;
	float64_t* q_matrix;
	float64_t* W_matrix;
#ifndef WIN32
	pthread_mutex_t* W_matrix_mutex;
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

bool CLocalTangentSpaceAlignment::init(CFeatures* features)
{
	return true;
}

void CLocalTangentSpaceAlignment::cleanup()
{
}

SGMatrix<float64_t> CLocalTangentSpaceAlignment::apply_to_feature_matrix(CFeatures* features)
{
	// shorthand for simplefeatures
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;
	SG_REF(features);
	ASSERT(simple_features);

	// get dimensionality and number of vectors of data
	int32_t dim = simple_features->get_num_features();
	ASSERT(m_target_dim<=dim);
	int32_t N = simple_features->get_num_vectors();
	ASSERT(m_k<N);

	// loop variables
	int32_t t;

	// compute distance matrix
	CDistance* distance = new CEuclidianDistance(simple_features,simple_features);
	SGMatrix<int32_t> neighborhood_matrix = get_neighborhood_matrix(distance);

	// init W (weight) matrix
	float64_t* W_matrix = SG_CALLOC(float64_t, N*N);

#ifndef WIN32
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads and params
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	D_THREAD_PARAM* parameters = SG_MALLOC(D_THREAD_PARAM, num_threads);
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

#ifndef WIN32
	pthread_mutex_t W_matrix_mutex;
	pthread_attr_t attr;
	pthread_mutex_init(&W_matrix_mutex, NULL);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (t=0; t<num_threads; t++)
	{
		parameters[t].idx_start = t;
		parameters[t].idx_step = num_threads;
		parameters[t].idx_stop = N;
		parameters[t].m_k = m_k;
		parameters[t].m_target_dim = m_target_dim;
		parameters[t].dim = dim;
		parameters[t].N = N;
		parameters[t].neighborhood_matrix = neighborhood_matrix.matrix;
		parameters[t].G_matrix = G_matrix + (m_k*(1+m_target_dim))*t;
		parameters[t].mean_vector = mean_vector + dim*t;
		parameters[t].local_feature_matrix = local_feature_matrix + (m_k*dim)*t;
		parameters[t].feature_matrix = feature_matrix.matrix;
		parameters[t].s_values_vector = s_values_vector + dim*t;
		parameters[t].q_matrix = q_matrix + (m_k*m_k)*t;
		parameters[t].W_matrix = W_matrix;
		parameters[t].W_matrix_mutex = &W_matrix_mutex;
		pthread_create(&threads[t], &attr, run_ltsa_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_mutex_destroy(&W_matrix_mutex);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	D_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_step = 1;
	single_thread_param.idx_stop = N;
	single_thread_param.m_k = m_k;
	single_thread_param.m_target_dim = m_target_dim;
	single_thread_param.dim = dim;
	single_thread_param.N = N;
	single_thread_param.neighborhood_matrix = neighborhood_matrix.matrix;
	single_thread_param.G_matrix = G_matrix;
	single_thread_param.mean_vector = mean_vector;
	single_thread_param.local_feature_matrix = local_feature_matrix;
	single_thread_param.feature_matrix = feature_matrix.matrix;
	single_thread_param.s_values_vector = s_values_vector;
	single_thread_param.q_matrix = q_matrix;
	single_thread_param.W_matrix = W_matrix;
	run_ltsa_thread((void*)&single_thread_param);
#endif

	// clean
	SG_FREE(G_matrix);
	SG_FREE(s_values_vector);
	SG_FREE(mean_vector);
	neighborhood_matrix.destroy_matrix();
	SG_FREE(local_feature_matrix);
	SG_FREE(q_matrix);

	// finally construct embedding
	SGMatrix<float64_t> W_sgmatrix(W_matrix,N,N);
	simple_features->set_feature_matrix(find_null_space(W_sgmatrix,m_target_dim,false));
	W_sgmatrix.destroy_matrix();

	SG_UNREF(features);
	return simple_features->get_feature_matrix();
}

SGVector<float64_t> CLocalTangentSpaceAlignment::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

void* CLocalTangentSpaceAlignment::run_ltsa_thread(void* p)
{
	D_THREAD_PARAM* parameters = (D_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t m_k = parameters->m_k;
	int32_t m_target_dim = parameters->m_target_dim;
	int32_t dim = parameters->dim;
	int32_t N = parameters->N;
	const int32_t* neighborhood_matrix = parameters->neighborhood_matrix;
	float64_t* G_matrix = parameters->G_matrix;
	float64_t* mean_vector = parameters->mean_vector;
	float64_t* local_feature_matrix = parameters->local_feature_matrix;
	const float64_t* feature_matrix = parameters->feature_matrix;
	float64_t* s_values_vector = parameters->s_values_vector;
	float64_t* q_matrix = parameters->q_matrix;
	float64_t* W_matrix = parameters->W_matrix;
#ifndef WIN32
	pthread_mutex_t* W_matrix_mutex = parameters->W_matrix_mutex;
#endif

	int i,j,k;

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		for (j=0; j<m_k; j++)
			G_matrix[j] = 1.0/CMath::sqrt((float64_t)m_k);

		// fill mean vector with zeros
		for (j=0; j<dim; j++)
			mean_vector[j] = 0.0;

		// compute local feature matrix containing neighbors of i-th vector
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
			{
				local_feature_matrix[j*dim+k] = feature_matrix[neighborhood_matrix[j*N+i]*dim+k];
				mean_vector[k] += local_feature_matrix[j*dim+k];
			}
		}

		// compute mean
		for (j=0; j<dim; j++)
			mean_vector[j] /= m_k;

		// center feature vectors by mean
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
				local_feature_matrix[j*dim+k] -= mean_vector[k];
		}

		int32_t info = 0;
		// find right eigenvectors of local_feature_matrix
		wrap_dgesvd('N','O', dim,m_k,local_feature_matrix,dim,
		                     s_values_vector,
		                     NULL,1, NULL,1, &info);
		ASSERT(info==0);
		
		for (j=0; j<m_target_dim; j++)
		{
			for (k=0; k<m_k; k++)
				G_matrix[(j+1)*m_k+k] = local_feature_matrix[k*dim+j];
		}
	
		// compute GG'
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
		            m_k,m_k,1+m_target_dim,
		            1.0,G_matrix,m_k,
		                G_matrix,m_k,
		            0.0,q_matrix,m_k);
		
		// W[neighbors of i, neighbors of i] = I - GG'
#ifndef WIN32
		pthread_mutex_lock(W_matrix_mutex);
#endif
		for (j=0; j<m_k; j++)
		{
			W_matrix[N*neighborhood_matrix[j*N+i]+neighborhood_matrix[j*N+i]] += 1.0;
			for (k=0; k<m_k; k++)
				W_matrix[N*neighborhood_matrix[k*N+i]+neighborhood_matrix[j*N+i]] -= q_matrix[j*m_k+k];
		}
#ifndef WIN32
		pthread_mutex_unlock(W_matrix_mutex);
#endif
	}
	return NULL;
}

#endif /* HAVE_LAPACK */
