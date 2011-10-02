/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/HessianLocallyLinearEmbedding.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
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
struct HESSIANESTIMATION_THREAD_PARAM
{
	/// starting index of loop
	int32_t idx_start;
	/// step of loop
	int32_t idx_step;
	/// end index of loop
	int32_t idx_stop;
	/// number of neighbors
	int32_t m_k;
	/// current dimensionality
	int32_t dim;
	///
	int32_t N;
	/// dp
	int32_t dp;
	/// target dimensionality
	int32_t target_dim;
	/// matrix containing indexes of neighbors of ith vector in ith column
	const int32_t* neighborhood_matrix;
	/// feature matrix 
	const float64_t* feature_matrix;
	/// local feature matrix contating features of neighbors
	float64_t* local_feature_matrix;
	/// Yi matrix
	float64_t* Yi_matrix;
	/// mean vector
	float64_t* mean_vector;
	/// singular values vector
	float64_t* s_values_vector;
	/// QR factorization reflectors
	float64_t* tau;
	/// length of reflectors vector
	int32_t tau_len;
	/// w sum vector
	float64_t* w_sum_vector;
	/// q matrix
	float64_t* q_matrix;
	/// weight matrix
	float64_t* W_matrix;
#ifdef HAVE_PTHREAD
	/// lock used on modifying of weight matrix
	PTHREAD_LOCK_T* W_matrix_lock;
#endif
};
#endif

CHessianLocallyLinearEmbedding::CHessianLocallyLinearEmbedding() :
		CLocallyLinearEmbedding()
{
}

CHessianLocallyLinearEmbedding::~CHessianLocallyLinearEmbedding()
{
}

bool CHessianLocallyLinearEmbedding::init(CFeatures* features)
{
	return true;
}

void CHessianLocallyLinearEmbedding::cleanup()
{
}

const char* CHessianLocallyLinearEmbedding::get_name() const 
{ 
	return "HessianLocallyLinearEmbedding";
};

EPreprocessorType CHessianLocallyLinearEmbedding::get_type() const
{
	return P_HESSIANLOCALLYLINEAREMBEDDING;
};


SGMatrix<float64_t> CHessianLocallyLinearEmbedding::construct_weight_matrix(CSimpleFeatures<float64_t>* simple_features,float64_t* W_matrix, 
                                                                            SGMatrix<int32_t> neighborhood_matrix)
{
	int32_t N = simple_features->get_num_vectors();
	int32_t dim = simple_features->get_num_features();
	int32_t target_dim = calculate_effective_target_dim(dim);
	int32_t dp = target_dim*(target_dim+1)/2;
	ASSERT(m_k>=(1+target_dim+dp));
	int32_t t;
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads and params
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	HESSIANESTIMATION_THREAD_PARAM* parameters = SG_MALLOC(HESSIANESTIMATION_THREAD_PARAM, num_threads);

#else
	int32_t num_threads = 1;
#endif

	// init matrices to be used
	float64_t* local_feature_matrix = SG_MALLOC(float64_t, m_k*dim*num_threads);
	float64_t* s_values_vector = SG_MALLOC(float64_t, dim*num_threads);
	int32_t tau_len = CMath::min((1+target_dim+dp), m_k);
	float64_t* tau = SG_MALLOC(float64_t, tau_len*num_threads);
	float64_t* mean_vector = SG_MALLOC(float64_t, dim*num_threads);
	float64_t* q_matrix = SG_MALLOC(float64_t, m_k*m_k*num_threads);
	float64_t* w_sum_vector = SG_MALLOC(float64_t, dp*num_threads);
	float64_t* Yi_matrix = SG_MALLOC(float64_t, m_k*(1+target_dim+dp)*num_threads);
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
		parameters[t].dim = dim;
		parameters[t].target_dim = target_dim;
		parameters[t].N = N;
		parameters[t].dp = dp;
		parameters[t].neighborhood_matrix = neighborhood_matrix.matrix;
		parameters[t].feature_matrix = feature_matrix.matrix;
		parameters[t].local_feature_matrix = local_feature_matrix + (m_k*dim)*t;
		parameters[t].Yi_matrix = Yi_matrix + (m_k*(1+target_dim+dp))*t;
		parameters[t].mean_vector = mean_vector + dim*t;
		parameters[t].s_values_vector = s_values_vector + dim*t;
		parameters[t].tau = tau+tau_len*t;
		parameters[t].tau_len = tau_len;
		parameters[t].w_sum_vector = w_sum_vector + dp*t;
		parameters[t].q_matrix = q_matrix + (m_k*m_k)*t;
		parameters[t].W_matrix = W_matrix;
		parameters[t].W_matrix_lock = &W_matrix_lock;
		pthread_create(&threads[t], &attr, run_hessianestimation_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	PTHREAD_LOCK_DESTROY(&W_matrix_lock);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	HESSIANESTIMATION_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = t;
	single_thread_param.idx_step = num_threads;
	single_thread_param.idx_stop = N;
	single_thread_param.m_k = m_k;
	single_thread_param.dim = dim;
	single_thread_param.target_dim = target_dim;
	single_thread_param.N = N;
	single_thread_param.dp = dp;
	single_thread_param.neighborhood_matrix = neighborhood_matrix.matrix;
	single_thread_param.feature_matrix = feature_matrix.matrix;
	single_thread_param.local_feature_matrix = local_feature_matrix;
	single_thread_param.Yi_matrix = Yi_matrix;
	single_thread_param.mean_vector = mean_vector;
	single_thread_param.s_values_vector = s_values_vector;
	single_thread_param.tau = tau;
	single_thread_param.tau_len = tau_len;
	single_thread_param.w_sum_vector = w_sum_vector;
	single_thread_param.q_matrix = q_matrix;
	single_thread_param.W_matrix = W_matrix;
	run_hessianestimation_thread((void*)&single_thread_param);
#endif

	// clean
	SG_FREE(Yi_matrix);
	SG_FREE(s_values_vector);
	SG_FREE(mean_vector);
	SG_FREE(tau);
	SG_FREE(w_sum_vector);
	SG_FREE(local_feature_matrix);
	SG_FREE(q_matrix);

	return SGMatrix<float64_t>(W_matrix,N,N);
}

void* CHessianLocallyLinearEmbedding::run_hessianestimation_thread(void* p)
{
	HESSIANESTIMATION_THREAD_PARAM* parameters = (HESSIANESTIMATION_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t m_k = parameters->m_k;
	int32_t dim = parameters->dim;
	int32_t N = parameters->N;
	int32_t dp = parameters->dp;
	int32_t target_dim = parameters->target_dim;
	const int32_t* neighborhood_matrix = parameters->neighborhood_matrix;
	const float64_t* feature_matrix = parameters->feature_matrix;
	float64_t* local_feature_matrix = parameters->local_feature_matrix;
	float64_t* Yi_matrix = parameters->Yi_matrix;
	float64_t* mean_vector = parameters->mean_vector;
	float64_t* s_values_vector = parameters->s_values_vector;
	float64_t* tau = parameters->tau;
	int32_t tau_len = parameters->tau_len;
	float64_t* w_sum_vector = parameters->w_sum_vector;
	float64_t* q_matrix = parameters->q_matrix;
	float64_t* W_matrix = parameters->W_matrix;
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_T* W_matrix_lock = parameters->W_matrix_lock;
#endif

	int i,j,k,l;

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		// Yi(:,0) = 1
		for (j=0; j<m_k; j++)
			Yi_matrix[j] = 1.0;

		// fill mean vector with zeros
		memset(mean_vector,0,sizeof(float64_t)*dim);

		// compute local feature matrix containing neighbors of i-th vector
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
				local_feature_matrix[j*dim+k] = feature_matrix[neighborhood_matrix[j*N+i]*dim+k];

			cblas_daxpy(dim,1.0,local_feature_matrix+j*dim,1,mean_vector,1);
		}

		// compute mean
		cblas_dscal(dim,1.0/m_k,mean_vector,1);

		// center feature vectors by mean
		for (j=0; j<m_k; j++)
			cblas_daxpy(dim,-1.0,mean_vector,1,local_feature_matrix+j*dim,1);

		int32_t info = 0;
		// find right eigenvectors of local_feature_matrix
		wrap_dgesvd('N','O', dim,m_k,local_feature_matrix,dim,
		                     s_values_vector,
		                     NULL,1, NULL,1, &info);
		ASSERT(info==0);

		// Yi(0:m_k,1:1+target_dim) = Vh(0:m_k, 0:target_dim)
		for (j=0; j<target_dim; j++)
		{
			for (k=0; k<m_k; k++)
				Yi_matrix[(j+1)*m_k+k] = local_feature_matrix[k*dim+j];
		}

		int32_t ct = 0;
		
		// construct 2nd order hessian approx
		for (j=0; j<target_dim; j++)
		{
			for (k=0; k<target_dim-j; k++)
			{
				for (l=0; l<m_k; l++)
				{
					Yi_matrix[(ct+k+1+target_dim)*m_k+l] = Yi_matrix[(j+1)*m_k+l]*Yi_matrix[(j+k+1)*m_k+l];
				}
			}
			ct += ct + target_dim - j;
		}
	
		// perform QR factorization
		wrap_dgeqrf(m_k,(1+target_dim+dp),Yi_matrix,m_k,tau,&info);
		ASSERT(info==0);
		wrap_dorgqr(m_k,(1+target_dim+dp),tau_len,Yi_matrix,m_k,tau,&info);
		ASSERT(info==0);
		
		float64_t* Pii = (Yi_matrix+m_k*(1+target_dim));

		for (j=0; j<dp; j++)
		{
			w_sum_vector[j] = 0.0;
			for (k=0; k<m_k; k++)
			{
				w_sum_vector[j] += Pii[j*m_k+k];
			}
			if (w_sum_vector[j]<0.001) 
				w_sum_vector[j] = 1.0;
			for (k=0; k<m_k; k++)
				Pii[j*m_k+k] /= w_sum_vector[j];
		}
		
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,
		            m_k,m_k,dp,
		            1.0,Pii,m_k,
		                Pii,m_k,
		            0.0,q_matrix,m_k);
#ifdef HAVE_PTHREAD
		PTHREAD_LOCK(W_matrix_lock);
#endif
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
				W_matrix[N*neighborhood_matrix[k*N+i]+neighborhood_matrix[j*N+i]] += q_matrix[j*m_k+k];
		}
#ifdef HAVE_PTHREAD
		PTHREAD_UNLOCK(W_matrix_lock);
#endif
	}
	return NULL;
}
#endif /* HAVE_LAPACK */
