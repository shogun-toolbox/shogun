/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/KernelLocallyLinearEmbedding.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/lib/common.h>
#include <shogun/base/DynArray.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct LK_RECONSTRUCTION_THREAD_PARAM
{
	/// starting index of loop
	int32_t idx_start;
	/// step of loop
	int32_t idx_step;
		/// end loop index
	int32_t idx_stop;
	/// number of neighbors
	int32_t m_k;
	/// number of vectors
	int32_t N;
	/// matrix containing indexes of ith neighbors of jth vector in ith column
	const int32_t* neighborhood_matrix;
	/// local gram matrix 
	float64_t* local_gram_matrix;
	/// gram matrix
	const float64_t* kernel_matrix;
	/// vector used for solving equation 
	float64_t* id_vector;
	/// weight matrix
	float64_t* W_matrix;
};

struct K_NEIGHBORHOOD_THREAD_PARAM
{
	/// starting index of loop
	int32_t idx_start;
	/// step of loop
	int32_t idx_step;
	/// end index of loop
	int32_t idx_stop;
	/// number of vectors
	int32_t N;
	/// number of neighbors
	int32_t m_k;
	/// fibonacci heaps
	CFibonacciHeap* heap;
	/// kernel matrix
	const float64_t* kernel_matrix;
	/// matrix containing neighbors indexes
	int32_t* neighborhood_matrix;
};

struct SPARSEDOT_THREAD_PARAM
{
	/// starting index of loop
	int32_t idx_start;
	/// step of loop
	int32_t idx_step;
	/// end index of loop
	int32_t idx_stop;
	/// number of vectors
	int32_t N;
	/// weight matrix
	const float64_t* W_matrix;
	/// result matrix
	float64_t* M_matrix;
	/// non zero indexes dynamic array
	DynArray<int32_t>** nz_idxs;
};
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CKernelLocallyLinearEmbedding::CKernelLocallyLinearEmbedding() :
		CLocallyLinearEmbedding()
{
	init();
}

CKernelLocallyLinearEmbedding::CKernelLocallyLinearEmbedding(CKernel* kernel)
{
	init();
	
	set_kernel(kernel);
}

const char* CKernelLocallyLinearEmbedding::get_name() const
{
	return "KernelLocallyLinearEmbedding";
};

EPreprocessorType CKernelLocallyLinearEmbedding::get_type() const
{
	return P_KERNELLOCALLYLINEAREMBEDDING;
};

void CKernelLocallyLinearEmbedding::init()
{
}

CKernelLocallyLinearEmbedding::~CKernelLocallyLinearEmbedding()
{
}

bool CKernelLocallyLinearEmbedding::init(CFeatures* features)
{
	return true;
}

void CKernelLocallyLinearEmbedding::cleanup()
{
}

SGMatrix<float64_t> CKernelLocallyLinearEmbedding::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(features);
	SG_REF(features);

	// get dimensionality and number of vectors of data
	bool is_simple = ((features->get_feature_class()==C_SIMPLE) && (features->get_feature_type()==F_DREAL));
	int32_t N = features->get_num_vectors();
	int32_t target_dim = 0;
	if (is_simple)
		target_dim = calculate_effective_target_dim(((CSimpleFeatures<float64_t>*)features)->get_num_features());
	else
	{
		if (m_target_dim<=0)
			SG_ERROR("Cannot decrease dimensionality of given features by %d.\n", -m_target_dim);
	}
	if (target_dim<=0)
		SG_ERROR("Trying to decrease dimensionality to non-positive value, not possible.\n");
	if (m_k>=N)
		SG_ERROR("Number of neighbors (%d) should be less than number of objects (%d).\n",
		         m_k, N);

	// compute kernel matrix
	ASSERT(m_kernel);
	m_kernel->init(features,features);
	SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();
	SGMatrix<int32_t> neighborhood_matrix = get_neighborhood_matrix(kernel_matrix);
	m_kernel->cleanup();

	// init W (weight) matrix
	SGMatrix<float64_t> M_matrix = construct_weight_matrix(kernel_matrix,neighborhood_matrix,target_dim);
	neighborhood_matrix.destroy_matrix();

	SGMatrix<float64_t> nullspace = find_null_space(M_matrix,target_dim);
	M_matrix.destroy_matrix();

	if (is_simple)
	{
		((CSimpleFeatures<float64_t>*)features)->set_feature_matrix(nullspace);
		SG_UNREF(features);
		return ((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();
	}
	else
	{
		SG_UNREF(features);
		SG_WARNING("Can't set feature matrix, returning feature matrix.\n");
		return nullspace;
	}
}

SGMatrix<float64_t> CKernelLocallyLinearEmbedding::construct_weight_matrix(SGMatrix<float64_t> kernel_matrix, 
                                                                           SGMatrix<int32_t> neighborhood_matrix,
                                                                           int32_t target_dim)
{
	int32_t N = kernel_matrix.num_cols;
	// loop variables
	int32_t i,j,t;
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	LK_RECONSTRUCTION_THREAD_PARAM* parameters = SG_MALLOC(LK_RECONSTRUCTION_THREAD_PARAM, num_threads);
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
#else
	int32_t num_threads = 1;
#endif 
	float64_t* W_matrix = SG_CALLOC(float64_t, N*N);
	// init matrices and norm factor to be used
	float64_t* local_gram_matrix = SG_MALLOC(float64_t, m_k*m_k*num_threads);
	float64_t* id_vector = SG_MALLOC(float64_t, m_k*num_threads);

#ifdef HAVE_PTHREAD
	for (t=0; t<num_threads; t++)
	{
		parameters[t] = (LK_RECONSTRUCTION_THREAD_PARAM){t,num_threads,N,m_k,N,neighborhood_matrix.matrix,
		                                                 local_gram_matrix+(m_k*m_k)*t,kernel_matrix.matrix,
		                                                 id_vector+m_k*t,W_matrix};
		pthread_create(&threads[t], &attr, run_linearreconstruction_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	LK_RECONSTRUCTION_THREAD_PARAM single_thread_param = {0,1,N,m_k,N,neighborhood_matrix.matrix,local_gram_matrix,
	                                                      kernel_matrix.matrix,id_vector,W_matrix};
	run_linearreconstruction_thread((void*)single_thread_param);
#endif

	// clean
	SG_FREE(id_vector);
	SG_FREE(local_gram_matrix);

	// W=I-W
	// W=I-W
	for (i=0; i<N*N; i++)
	{
		W_matrix[i] *= -1.0;
	}
	for (i=0; i<N; i++)
	{
		W_matrix[i*N+i] = 1.0;
	}

	// compute M=(W-I)'*(W-I)
	DynArray<int32_t>** nz_idxs = SG_MALLOC(DynArray<int32_t>*,N);
	for (i=0; i<N; i++)
	{
		nz_idxs[i] = new DynArray<int32_t>(m_k,false);
		for (j=0; j<N; j++)
		{
			if (W_matrix[i*N+j]!=0.0)
				nz_idxs[i]->push_back(j);
		}
	}
	SGMatrix<float64_t> M_matrix(kernel_matrix.matrix,N,N);
#ifdef HAVE_PTHREAD
	// allocate threads
	threads = SG_MALLOC(pthread_t, num_threads);
	SPARSEDOT_THREAD_PARAM* parameters_ = SG_MALLOC(SPARSEDOT_THREAD_PARAM, num_threads);
	pthread_attr_t attr_;
	pthread_attr_init(&attr_);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for (t=0; t<num_threads; t++)
	{
		parameters_[t] = (SPARSEDOT_THREAD_PARAM){t,num_threads,N,N,W_matrix,M_matrix.matrix,nz_idxs};
		pthread_create(&threads[t], &attr_, run_sparsedot_thread, (void*)&parameters_[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr_);
	SG_FREE(parameters_);
	SG_FREE(threads);
#else
	SPARSEDOT_THREAD_PARAM single_thread_param = {0,1,N,N,W_matrix,M_matrix.matrix,nz_idxs};
	run_sparsedot_thread((void*)single_thread_param);
#endif
	for (i=0; i<N; i++)
	{
		delete nz_idxs[i];
		for (j=0; j<i; j++)
		{
			M_matrix[i*N+j] = M_matrix[j*N+i];
		}
	}
	SG_FREE(nz_idxs);
	SG_FREE(W_matrix);
	return M_matrix;
}

SGVector<float64_t> CKernelLocallyLinearEmbedding::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

void* CKernelLocallyLinearEmbedding::run_linearreconstruction_thread(void* p)
{
	LK_RECONSTRUCTION_THREAD_PARAM* parameters = (LK_RECONSTRUCTION_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t m_k = parameters->m_k;
	int32_t N = parameters->N;
	const int32_t* neighborhood_matrix = parameters->neighborhood_matrix;
	float64_t* local_gram_matrix = parameters->local_gram_matrix;
	const float64_t* kernel_matrix = parameters->kernel_matrix;
	float64_t* id_vector = parameters->id_vector;
	float64_t* W_matrix = parameters->W_matrix;

	int32_t i,j,k;
	float64_t norming,trace;

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
				local_gram_matrix[j*m_k+k] = 
					kernel_matrix[i*N+i] -
					kernel_matrix[i*N+neighborhood_matrix[j*N+i]] -
					kernel_matrix[i*N+neighborhood_matrix[k*N+i]] +
					kernel_matrix[neighborhood_matrix[j*N+i]*N+neighborhood_matrix[k*N+i]];
		}

		for (j=0; j<m_k; j++)
			id_vector[j] = 1.0;

		// compute tr(C)
		trace = 0.0;
		for (j=0; j<m_k; j++)
			trace += local_gram_matrix[j*m_k+j];
		
		// regularize gram matrix
		for (j=0; j<m_k; j++)
			local_gram_matrix[j*m_k+j] += 1e-3*trace/m_k;

		clapack_dposv(CblasColMajor,CblasLower,m_k,1,local_gram_matrix,m_k,id_vector,m_k);

		// normalize weights
		norming=0.0;
		for (j=0; j<m_k; j++)
			norming += id_vector[j];

		for (j=0; j<m_k; j++)
			id_vector[j]/=norming;

		// put weights into W matrix
		for (j=0; j<m_k; j++)
			W_matrix[N*neighborhood_matrix[j*N+i]+i]=id_vector[j];
	}
	return NULL;
}

SGMatrix<int32_t> CKernelLocallyLinearEmbedding::get_neighborhood_matrix(SGMatrix<float64_t> kernel_matrix)
{
	int32_t t;
	int32_t N = kernel_matrix.num_cols;
	// init matrix and heap to be used
	int32_t* neighborhood_matrix = SG_MALLOC(int32_t, N*m_k);
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	K_NEIGHBORHOOD_THREAD_PARAM* parameters = SG_MALLOC(K_NEIGHBORHOOD_THREAD_PARAM, num_threads);
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
#else
	int32_t num_threads = 1;
#endif
	CFibonacciHeap** heaps = SG_MALLOC(CFibonacciHeap*, num_threads);
	for (t=0; t<num_threads; t++)
		heaps[t] = new CFibonacciHeap(N);

#ifdef HAVE_PTHREAD
	for (t=0; t<num_threads; t++)
	{
		parameters[t] = (K_NEIGHBORHOOD_THREAD_PARAM){t,num_threads,N,m_k,N,heaps[t],kernel_matrix.matrix,neighborhood_matrix};
		pthread_create(&threads[t], &attr, run_neighborhood_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	SG_FREE(threads);
	SG_FREE(parameters);
#else
	K_NEIGHBORHOOD_THREAD_PARAM single_thread_param = {0,1,N,m_k,N,heaps[0],kernel_matrix.matrix,neighborhood_matrix};
	run_neighborhood_thread((void*)&single_thread_param);
#endif

	for (t=0; t<num_threads; t++)
		delete heaps[t];
	SG_FREE(heaps);

	return SGMatrix<int32_t>(neighborhood_matrix,m_k,N);
}

void* CKernelLocallyLinearEmbedding::run_neighborhood_thread(void* p)
{
	K_NEIGHBORHOOD_THREAD_PARAM* parameters = (K_NEIGHBORHOOD_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t N = parameters->N;
	int32_t m_k = parameters->m_k;
	CFibonacciHeap* heap = parameters->heap;
	const float64_t* kernel_matrix = parameters->kernel_matrix;
	int32_t* neighborhood_matrix = parameters->neighborhood_matrix;

	int32_t i,j;
	float64_t tmp;
	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		for (j=0; j<N; j++)
		{
			heap->insert(j,kernel_matrix[i*N+i]-2*kernel_matrix[i*N+j]+kernel_matrix[j*N+j]);
		}

		heap->extract_min(tmp);

		for (j=0; j<m_k; j++)
			neighborhood_matrix[j*N+i] = heap->extract_min(tmp);

		heap->clear();
	}

	return NULL;
}
#endif /* HAVE_LAPACK */
