/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/LocallyLinearEmbedding.h>
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/lib/Signal.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct D_THREAD_PARAM
{
	int32_t idx_start;
	int32_t idx_stop;
	int32_t idx_step;
	int32_t m_k;
	int32_t dim;
	int32_t N;
	const int32_t* neighborhood_matrix;
	const float64_t* feature_matrix;
	float64_t* z_matrix;
	float64_t* covariance_matrix;
	float64_t* id_vector;
	float64_t* W_matrix;
};

struct D_NEIGHBORHOOD_THREAD_PARAM
{
	int32_t idx_start;
	int32_t idx_step;
	int32_t idx_stop;
	int32_t N;
	int32_t m_k;
	CFibonacciHeap* heap;
	const float64_t* distance_matrix;
	int32_t* neighborhood_matrix;
};
#endif

CLocallyLinearEmbedding::CLocallyLinearEmbedding() :
		CDimensionReductionPreprocessor(), m_k(3), m_posdef(true)
{
}

void CLocallyLinearEmbedding::init()
{
	m_k = 3;
	m_posdef = true;

	m_parameters->add(&m_k, "k", "number of neighbors");
	m_parameters->add(&m_posdef, "posdef", 
	                  "indicates if matrix should be considered as positive definite");
}

CLocallyLinearEmbedding::~CLocallyLinearEmbedding()
{
}

bool CLocallyLinearEmbedding::init(CFeatures* features)
{
	return true;
}

void CLocallyLinearEmbedding::cleanup()
{
}

SGMatrix<float64_t> CLocallyLinearEmbedding::apply_to_feature_matrix(CFeatures* features)
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
	int32_t i,j,t;

	// compute distance matrix
	CDistance* distance = new CEuclidianDistance(simple_features,simple_features);
	SG_UNREF(distance->parallel);
	distance->parallel = this->parallel;
	SGMatrix<int32_t> neighborhood_matrix = get_neighborhood_matrix(distance);
	delete distance;

	// init W (weight) matrix
	float64_t* W_matrix = SG_CALLOC(float64_t, N*N);

#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	D_THREAD_PARAM* parameters = SG_MALLOC(D_THREAD_PARAM, num_threads);
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
#else
	int32_t num_threads = 1;
#endif 
	// init matrices and norm factor to be used
	float64_t* z_matrix = SG_MALLOC(float64_t, m_k*dim*num_threads);
	float64_t* covariance_matrix = SG_MALLOC(float64_t, m_k*m_k*num_threads);
	float64_t* id_vector = SG_MALLOC(float64_t, m_k*num_threads);

	// get feature matrix
	SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();

#ifdef HAVE_PTHREAD
	for (t=0; t<num_threads; t++)
	{
		parameters[t].idx_start = t;
		parameters[t].idx_step = num_threads;
		parameters[t].idx_stop = N;
		parameters[t].m_k = m_k;
		parameters[t].dim = dim;
		parameters[t].N = N;
		parameters[t].neighborhood_matrix = neighborhood_matrix.matrix;
		parameters[t].z_matrix = z_matrix+(m_k*dim)*t;
		parameters[t].feature_matrix = feature_matrix.matrix;
		parameters[t].covariance_matrix = covariance_matrix+(m_k*m_k)*t;
		parameters[t].id_vector = id_vector+m_k*t;
		parameters[t].W_matrix = W_matrix;
		pthread_create(&threads[t], &attr, run_linearreconstruction_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	D_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_step = 1;
	single_thread_param.idx_stop = N;
	single_thread_param.m_k = m_k;
	single_thread_param.dim = dim;
	single_thread_param.N = N;
	single_thread_param.neighborhood_matrix = neighborhood_matrix.matrix;
	single_thread_param.z_matrix = z_matrix;
	single_thread_param.feature_matrix = feature_matrix.matrix;
	single_thread_param.covariance_matrix = covariance_matrix;
	single_thread_param.id_vector = id_vector;
	single_thread_param.W_matrix = W_matrix;
	run_linearreconstruction_thread((void*)single_thread_param);
#endif

	// clean
	SG_FREE(id_vector);
	neighborhood_matrix.destroy_matrix();
	SG_FREE(z_matrix);
	SG_FREE(covariance_matrix);

	// W=I-W
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
			W_matrix[j*N+i] = (i==j) ? 1.0-W_matrix[j*N+i] : -W_matrix[j*N+i];
	}

	// compute M=(W-I)'*(W-I)
	SGMatrix<float64_t> M_matrix(N,N);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans,
	            N,N,N,
	            1.0,W_matrix,N,
	            W_matrix,N,
	            0.0,M_matrix.matrix,N);

	SG_FREE(W_matrix);

	simple_features->set_feature_matrix(find_null_space(M_matrix,m_target_dim,false));
	M_matrix.destroy_matrix();

	SG_UNREF(features);
	return simple_features->get_feature_matrix();
}

SGVector<float64_t> CLocallyLinearEmbedding::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

SGMatrix<float64_t> CLocallyLinearEmbedding::find_null_space(SGMatrix<float64_t> matrix, int dimension, bool force_lapack)
{
	int i,j;
	ASSERT(matrix.num_cols==matrix.num_rows);
	int N = matrix.num_cols;
	// get eigenvectors with ARPACK or LAPACK
	int eigenproblem_status = 0;

	bool arpack = false;

#ifdef HAVE_ARPACK
	arpack = true;
#endif

	if (force_lapack) arpack = false;

	float64_t* eigenvalues_vector;
	float64_t* eigenvectors;
	if (arpack)
	{
		// using ARPACK (faster)
		eigenvalues_vector = SG_MALLOC(float64_t, dimension+1);
		#ifdef HAVE_ARPACK
		arpack_dsaupd(matrix.matrix,NULL,N,dimension+1,"LA",3,m_posdef,-1e-6,eigenvalues_vector,matrix.matrix,eigenproblem_status);
		#endif
	}
	else
	{
		// using LAPACK (slower)
		eigenvalues_vector = SG_MALLOC(float64_t, N);
		eigenvectors = SG_MALLOC(float64_t,(dimension+1)*N);
		wrap_dsyevr('V','U',N,matrix.matrix,N,2,dimension+2,eigenvalues_vector,eigenvectors,&eigenproblem_status);
	}

	// check if failed
	if (eigenproblem_status)
		SG_ERROR("Eigenproblem failed with code: %d", eigenproblem_status);
	
	// allocate null space feature matrix
	float64_t* null_space_features = SG_MALLOC(float64_t, N*dimension);

	// construct embedding w.r.t to used solver (prefer ARPACK if available)
	if (arpack) 
	{
		// ARPACKed eigenvectors
		for (i=0; i<dimension; i++)
		{
			for (j=0; j<N; j++)
				null_space_features[j*dimension+i] = matrix.matrix[j*(dimension+1)+i+1];
		}
	}
	else
	{
		// LAPACKed eigenvectors
		for (i=0; i<dimension; i++)
		{
			for (j=0; j<N; j++)
				null_space_features[j*dimension+i] = eigenvectors[i*N+j];
		}
		SG_FREE(eigenvectors);
	}
	SG_FREE(eigenvalues_vector);

	return SGMatrix<float64_t>(null_space_features,dimension,N);
}

void* CLocallyLinearEmbedding::run_linearreconstruction_thread(void* p)
{
	D_THREAD_PARAM* parameters = (D_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t m_k = parameters->m_k;
	int32_t dim = parameters->dim;
	int32_t N = parameters->N;
	const int32_t* neighborhood_matrix = parameters->neighborhood_matrix;
	float64_t* z_matrix = parameters->z_matrix;
	const float64_t* feature_matrix = parameters->feature_matrix;
	float64_t* covariance_matrix = parameters->covariance_matrix;
	float64_t* id_vector = parameters->id_vector;
	float64_t* W_matrix = parameters->W_matrix;

	int32_t i,j,k;
	float64_t norming,trace;

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		// compute local feature matrix containing neighbors of i-th vector
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
				z_matrix[j*dim+k] = feature_matrix[neighborhood_matrix[j*N+i]*dim+k];
		}

		// center features by subtracting i-th feature column
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
				z_matrix[j*dim+k] -= feature_matrix[i*dim+k];
		}

		// compute local covariance matrix
		cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,
		            m_k,m_k,dim,
		            1.0,z_matrix,dim,
		            z_matrix,dim,
		            0.0,covariance_matrix,m_k);

		for (j=0; j<m_k; j++)
			id_vector[j] = 1.0;

		// regularize in case of ill-posed system
		if (m_k>dim)
		{
			// compute tr(C)
			trace = 0.0;
			for (j=0; j<m_k; j++)
				trace += covariance_matrix[j*m_k+j];

			for (j=0; j<m_k; j++)
				covariance_matrix[j*m_k+j] += 1e-3*trace;
		}

		// solve system of linear equations: covariance_matrix * X = 1
		// covariance_matrix is a pos-def matrix
		clapack_dposv(CblasColMajor,CblasLower,m_k,1,covariance_matrix,m_k,id_vector,m_k);

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

SGMatrix<int32_t> CLocallyLinearEmbedding::get_neighborhood_matrix(CDistance* distance)
{
	int32_t t;
	SGMatrix<float64_t> distance_matrix = distance->get_distance_matrix();
	int32_t N = distance->get_num_vec_lhs();
	// init matrix and heap to be used
	int32_t* neighborhood_matrix = SG_MALLOC(int32_t, N*m_k);
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	D_NEIGHBORHOOD_THREAD_PARAM* parameters = SG_MALLOC(D_NEIGHBORHOOD_THREAD_PARAM, num_threads);
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
		parameters[t].idx_start = t;
		parameters[t].idx_step = num_threads;
		parameters[t].idx_stop = N;
		parameters[t].m_k = m_k;
		parameters[t].N = N;
		parameters[t].heap = heaps[t];
		parameters[t].neighborhood_matrix = neighborhood_matrix;
		parameters[t].distance_matrix = distance_matrix.matrix;
		pthread_create(&threads[t], &attr, run_neighborhood_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	SG_FREE(threads);
	SG_FREE(parameters);
#else
	D_NEIGHBORHOOD_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_step = 1;
	single_thread_param.idx_stop = N;
	single_thread_param.m_k = m_k;
	single_thread_param.N = N;
	single_thread_param.heap = heaps[0]
	single_thread_param.neighborhood_matrix = neighborhood_matrix;
	single_thread_param.distance_matrix = distance_matrix;
	run_neighborhood_thread((void*)&single_thread_param);
#endif

	for (t=0; t<num_threads; t++)
		delete heaps[t];
	SG_FREE(heaps);
	distance_matrix.destroy_matrix();

	return SGMatrix<int32_t>(neighborhood_matrix,m_k,N);
}


void* CLocallyLinearEmbedding::run_neighborhood_thread(void* p)
{
	D_NEIGHBORHOOD_THREAD_PARAM* parameters = (D_NEIGHBORHOOD_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t N = parameters->N;
	int32_t m_k = parameters->m_k;
	CFibonacciHeap* heap = parameters->heap;
	const float64_t* distance_matrix = parameters->distance_matrix;
	int32_t* neighborhood_matrix = parameters->neighborhood_matrix;

	int32_t i,j;
	float64_t tmp;
	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		for (j=0; j<N; j++)
		{
			heap->insert(j,distance_matrix[i*N+j]);
		}

		heap->extract_min(tmp);

		for (j=0; j<m_k; j++)
			neighborhood_matrix[j*N+i] = heap->extract_min(tmp);

		heap->clear();
	}

	return NULL;
}
#endif /* HAVE_LAPACK */
