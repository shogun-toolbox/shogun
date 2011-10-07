/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/KernelLocalTangentSpaceAlignment.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parallel.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct KLTSA_THREAD_PARAM
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
	/// number of objects
	int32_t N;
	/// matrix containing indexes of ith vector's neighbors in ith column
	const int32_t* neighborhood_matrix;
	/// kernel matrix
	const float64_t* kernel_matrix;
	/// local gram matrix
	float64_t* local_gram_matrix;
	/// eigenvalues 
	float64_t* ev_vector;
	/// G matrix
	float64_t* G_matrix;
	/// weight matrix
	float64_t* W_matrix;
#ifdef HAVE_PTHREAD
	/// lock used on modifying to weight matrix
	PTHREAD_LOCK_T* W_matrix_lock;
#endif
};
#endif

CKernelLocalTangentSpaceAlignment::CKernelLocalTangentSpaceAlignment() :
		CKernelLocallyLinearEmbedding()
{
}

CKernelLocalTangentSpaceAlignment::~CKernelLocalTangentSpaceAlignment()
{
}

bool CKernelLocalTangentSpaceAlignment::init(CFeatures* features)
{
	return true;
}

void CKernelLocalTangentSpaceAlignment::cleanup()
{
}

const char* CKernelLocalTangentSpaceAlignment::get_name() const
{ 
	return "KernelLocalTangentSpaceAlignment"; 
};

EPreprocessorType CKernelLocalTangentSpaceAlignment::get_type() const
{
	return P_KERNELLOCALTANGENTSPACEALIGNMENT;
};

SGMatrix<float64_t> CKernelLocalTangentSpaceAlignment::construct_weight_matrix(SGMatrix<float64_t> kernel_matrix,
                                                                               SGMatrix<int32_t> neighborhood_matrix,
                                                                               int32_t target_dim)
{
	int32_t N = kernel_matrix.num_cols;
	int32_t t;
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads and params
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	KLTSA_THREAD_PARAM* parameters = SG_MALLOC(KLTSA_THREAD_PARAM, num_threads);
#else
	int32_t num_threads = 1;
#endif

	// init matrices and norm factor to be used
	float64_t* local_gram_matrix = SG_MALLOC(float64_t, m_k*m_k*num_threads);
	float64_t* G_matrix = SG_MALLOC(float64_t, m_k*(1+target_dim)*num_threads);
	float64_t* W_matrix = SG_CALLOC(float64_t, N*N);
	float64_t* ev_vector = SG_MALLOC(float64_t, m_k*num_threads);

#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T W_matrix_lock;
	pthread_attr_t attr;
	PTHREAD_LOCK_INIT(&W_matrix_lock);
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (t=0; t<num_threads; t++)
	{
		KLTSA_THREAD_PARAM params = {t,num_threads,N,m_k,target_dim,N,neighborhood_matrix.matrix,
		                            kernel_matrix.matrix,local_gram_matrix+(m_k*m_k)*t,ev_vector+m_k*t,
		                            G_matrix+(m_k*(1+target_dim))*t,W_matrix,&W_matrix_lock};
		parameters[t] = params;
		pthread_create(&threads[t], &attr, run_kltsa_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	PTHREAD_LOCK_DESTROY(&W_matrix_lock);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	KLTSA_THREAD_PARAM single_thread_param = {0,1,N,m_k,target_dim,neighborhood_matrix.matrix,
	                                          kernel_matrix.matrix,local_gram_matrix,ev_vector,
	                                          G_matrix,W_matrix};
	run_kltsa_thread((void*)&single_thread_param);
#endif

	// clean
	SG_FREE(local_gram_matrix);
	SG_FREE(ev_vector);
	SG_FREE(G_matrix);
	kernel_matrix.destroy_matrix();

	for (int32_t i=0; i<N; i++)
	{
		for (int32_t j=0; j<m_k; j++)
			W_matrix[N*neighborhood_matrix[j*N+i]+neighborhood_matrix[j*N+i]] += 1.0;
	}

	return SGMatrix<float64_t>(W_matrix,N,N);
}

void* CKernelLocalTangentSpaceAlignment::run_kltsa_thread(void* p)
{
	KLTSA_THREAD_PARAM* parameters = (KLTSA_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t m_k = parameters->m_k;
	int32_t target_dim = parameters->target_dim;
	int32_t N = parameters->N;
	const int32_t* neighborhood_matrix = parameters->neighborhood_matrix;
	const float64_t* kernel_matrix = parameters->kernel_matrix;
	float64_t* ev_vector = parameters->ev_vector;
	float64_t* G_matrix = parameters->G_matrix;
	float64_t* local_gram_matrix = parameters->local_gram_matrix;
	float64_t* W_matrix = parameters->W_matrix;
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T* W_matrix_lock = parameters->W_matrix_lock;
#endif

	int32_t i,j,k;

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		for (j=0; j<m_k; j++)
			G_matrix[j] = 1.0/CMath::sqrt((float64_t)m_k);

		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
				local_gram_matrix[j*m_k+k] = kernel_matrix[neighborhood_matrix[j*N+i]*N+neighborhood_matrix[k*N+i]];
		}

		CMath::center_matrix(local_gram_matrix,m_k,m_k);

		int32_t info = 0; 
		wrap_dsyevr('V','U',m_k,local_gram_matrix,m_k,m_k-target_dim+1,m_k,ev_vector,G_matrix+m_k,&info);
		ASSERT(info==0);

		for (j=0; j<target_dim/2; j++)
		{
			cblas_dswap(m_k,G_matrix+(j+1)*m_k,1,G_matrix+(target_dim-j)*m_k,1);
		}

		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
		            m_k,m_k,1+target_dim,
		            1.0,G_matrix,m_k,
		                G_matrix,m_k,
		            0.0,local_gram_matrix,m_k);

#ifdef HAVE_PTHREAD
		PTHREAD_LOCK(W_matrix_lock);
#endif
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
				W_matrix[N*neighborhood_matrix[k*N+i]+neighborhood_matrix[j*N+i]] -= local_gram_matrix[j*m_k+k];
		}
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(W_matrix_lock);
#endif
	}
	return NULL;
}

#endif /* HAVE_LAPACK */
