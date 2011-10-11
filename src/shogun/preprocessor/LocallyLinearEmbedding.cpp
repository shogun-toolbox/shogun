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
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/base/DynArray.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct LINRECONSTRUCTION_THREAD_PARAM
{
	/// starting index of loop
	int32_t idx_start;
	/// end loop index
	int32_t idx_stop;
	/// step of loop
	int32_t idx_step;
	/// number of neighbors
	int32_t m_k;
	/// current dimension
	int32_t dim;
	/// number of vectors
	int32_t N;
	/// matrix containing indexes of neighbors of ith object in ith column
	const int32_t* neighborhood_matrix;
	/// old feature matrix
	const float64_t* feature_matrix;
	/// Z matrix containing features of neighbors
	float64_t* z_matrix;
	/// covariance matrix, ZZ'
	float64_t* covariance_matrix;
	/// vector used for solving equation 
	float64_t* id_vector;
	/// weight matrix
	float64_t* W_matrix;
	/// reconstruction regularization shift
	float64_t m_reconstruction_shift;
};

struct NEIGHBORHOOD_THREAD_PARAM
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
	/// heap used to get nearest vector's indexes
	CFibonacciHeap* heap;
	/// distance matrix
	const float64_t* distance_matrix;
	/// matrix containing indexes of neighbors of ith object in ith column
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

CLocallyLinearEmbedding::CLocallyLinearEmbedding() :
		CDimensionReductionPreprocessor<float64_t>()
{
	m_k = 10;
	m_max_k = 60;
	m_auto_k = false;
	m_nullspace_shift = -1e-9;
	m_reconstruction_shift = 1e-3;
#ifdef HAVE_ARPACK
	m_use_arpack = true;
#else
	m_use_arpack = false;
#endif
	init();
}

void CLocallyLinearEmbedding::init()
{
	m_parameters->add(&m_auto_k, "auto_k", 
	                  "whether k should be determined automatically in range");
	m_parameters->add(&m_k, "k", "number of neighbors");
	m_parameters->add(&m_max_k, "max_k", 
	                  "maximum number of neighbors used to compute optimal one");
	m_parameters->add(&m_nullspace_shift, "nullspace_shift",
	                  "nullspace finding regularization shift");
	m_parameters->add(&m_reconstruction_shift, "reconstruction_shift", 
	                  "shift used to regularize reconstruction step");
	m_parameters->add(&m_use_arpack, "use_arpack",
	                  "whether arpack is being used or not");
}


CLocallyLinearEmbedding::~CLocallyLinearEmbedding()
{
}

void CLocallyLinearEmbedding::set_k(int32_t k)
{
	ASSERT(k>0);
	m_k = k;
}

int32_t CLocallyLinearEmbedding::get_k() const
{
	return m_k;
}

void CLocallyLinearEmbedding::set_max_k(int32_t max_k)
{
	ASSERT(max_k>=m_k);
	m_max_k = max_k;
}

int32_t CLocallyLinearEmbedding::get_max_k() const
{
	return m_max_k;
}

void CLocallyLinearEmbedding::set_auto_k(bool auto_k)
{
	m_auto_k = auto_k;
}

bool CLocallyLinearEmbedding::get_auto_k() const
{
	return m_auto_k;
}

void CLocallyLinearEmbedding::set_nullspace_shift(float64_t nullspace_shift)
{
	m_nullspace_shift = nullspace_shift;
}

float64_t CLocallyLinearEmbedding::get_nullspace_shift() const
{
	return m_nullspace_shift;
}

void CLocallyLinearEmbedding::set_reconstruction_shift(float64_t reconstruction_shift)
{
	m_reconstruction_shift = reconstruction_shift;
}

float64_t CLocallyLinearEmbedding::get_reconstruction_shift() const
{
	return m_reconstruction_shift;
}

void CLocallyLinearEmbedding::set_use_arpack(bool use_arpack)
{
	m_use_arpack = use_arpack;
}

bool CLocallyLinearEmbedding::get_use_arpack() const
{
	return m_use_arpack;
}

const char* CLocallyLinearEmbedding::get_name() const
{
	return "LocallyLinearEmbedding";
}

EPreprocessorType CLocallyLinearEmbedding::get_type() const
{
	return P_LOCALLYLINEAREMBEDDING;
}

SGMatrix<float64_t> CLocallyLinearEmbedding::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(features);
	// check features
	if (!(features->get_feature_class()==C_SIMPLE &&
	      features->get_feature_type()==F_DREAL))
	{
		SG_ERROR("Given features are not of SimpleRealFeatures type.\n");
	}
	// shorthand for simplefeatures
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;
	SG_REF(features);

	// get and check number of vectors
	int32_t N = simple_features->get_num_vectors();
	if (m_k>=N)
		SG_ERROR("Number of neighbors (%d) should be less than number of objects (%d).\n",
		         m_k, N);

	// compute distance matrix
	SG_DEBUG("Computing distance matrix\n");
	ASSERT(m_distance);
	m_distance->init(simple_features,simple_features);
	SGMatrix<float64_t> distance_matrix = m_distance->get_distance_matrix();
	m_distance->remove_lhs_and_rhs();
	SG_DEBUG("Calculating neighborhood matrix\n");
	SGMatrix<int32_t> neighborhood_matrix;

	if (m_auto_k) 
	{
		neighborhood_matrix = get_neighborhood_matrix(distance_matrix,m_max_k);
		m_k = estimate_k(simple_features,neighborhood_matrix);
		SG_DEBUG("Estimated k with value of %d\n",m_k);
	} 
	else
		neighborhood_matrix = get_neighborhood_matrix(distance_matrix,m_k);

	// init W (weight) matrix
	float64_t* W_matrix = distance_matrix.matrix;
	memset(W_matrix,0,sizeof(float64_t)*N*N);

	// construct weight matrix
	SG_DEBUG("Constructing weight matrix\n");
	SGMatrix<float64_t> weight_matrix = construct_weight_matrix(simple_features,W_matrix,neighborhood_matrix);
	neighborhood_matrix.destroy_matrix();

	// find null space of weight matrix
	SG_DEBUG("Finding nullspace\n");
	simple_features->set_feature_matrix(find_null_space(weight_matrix,m_target_dim));
	weight_matrix.destroy_matrix();

	SG_UNREF(features);
	return simple_features->get_feature_matrix();
}

int32_t CLocallyLinearEmbedding::estimate_k(CSimpleFeatures<float64_t>* simple_features, SGMatrix<int32_t> neighborhood_matrix)
{
	int32_t right = m_max_k;
	int32_t left = m_k;
	int32_t left_third;
	int32_t right_third;
	ASSERT(right>=left);
	if (right==left) return left;
	int32_t dim;
	int32_t N;
	float64_t* feature_matrix= simple_features->get_feature_matrix(dim,N);
	float64_t* z_matrix = SG_MALLOC(float64_t,right*dim);
	float64_t* covariance_matrix = SG_MALLOC(float64_t,right*right);
	float64_t* resid_vector = SG_MALLOC(float64_t, right);
	float64_t* id_vector = SG_MALLOC(float64_t, right);
	while (right-left>2)
	{
		left_third = (left*2+right)/3;
		right_third = (right*2+left)/3;
		float64_t left_val = compute_reconstruction_error(left_third,dim,N,feature_matrix,z_matrix,
		                                                  covariance_matrix,resid_vector,
		                                                  id_vector,neighborhood_matrix);
		float64_t right_val = compute_reconstruction_error(right_third,dim,N,feature_matrix,z_matrix,
		                                                   covariance_matrix,resid_vector,
		                                                   id_vector,neighborhood_matrix);
		if (left_val<right_val)
			right = right_third;
		else 
			left = left_third;
	}
	SG_FREE(z_matrix);
	SG_FREE(covariance_matrix);
	SG_FREE(resid_vector);
	SG_FREE(id_vector);
	return right;
}

float64_t CLocallyLinearEmbedding::compute_reconstruction_error(int32_t k, int dim, int N, float64_t* feature_matrix,
                                                                float64_t* z_matrix, float64_t* covariance_matrix,
                                                                float64_t* resid_vector, float64_t* id_vector,
                                                                SGMatrix<int32_t> neighborhood_matrix)
{
	// todo parse params
	int32_t i,j;
	float64_t total_residual_norm = 0.0;
	for (i=0; i<N; i+=N/20)
	{
		for (j=0; j<k; j++)
		{
			cblas_dcopy(dim,feature_matrix+neighborhood_matrix[j*N+i]*dim,1,z_matrix+j*dim,1);   
			cblas_daxpy(dim,-1.0,feature_matrix+i*dim,1,z_matrix+j*dim,1);
		}
		cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,
			    k,k,dim,
			    1.0,z_matrix,dim,
			    z_matrix,dim,
			    0.0,covariance_matrix,k);
		for (j=0; j<k; j++)
		{
			resid_vector[j] = 1.0;
			id_vector[j] = 1.0;
		}
		if (k>dim)
		{
			float64_t trace = 0.0;
			for (j=0; j<k; j++)
				trace += covariance_matrix[j*k+j];
			for (j=0; j<m_k; j++)
				covariance_matrix[j*k+j] += m_reconstruction_shift*trace;
		}
		clapack_dposv(CblasColMajor,CblasLower,k,1,covariance_matrix,k,id_vector,k);
		float64_t norming=0.0;
		for (j=0; j<k; j++)
			norming += id_vector[j];
		cblas_dscal(k,-1.0/norming,id_vector,1);
		cblas_dsymv(CblasColMajor,CblasLower,k,-1.0,covariance_matrix,k,id_vector,1,1.0,resid_vector,1);
		total_residual_norm += cblas_dnrm2(k,resid_vector,1);
	}
	return total_residual_norm/k;
}

SGVector<float64_t> CLocallyLinearEmbedding::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

SGMatrix<float64_t> CLocallyLinearEmbedding::construct_weight_matrix(CSimpleFeatures<float64_t>* simple_features,
                                                                     float64_t* W_matrix, SGMatrix<int32_t> neighborhood_matrix)
{
	int32_t N = simple_features->get_num_vectors();
	int32_t dim = simple_features->get_num_features();
	int32_t i,j,t;
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	LINRECONSTRUCTION_THREAD_PARAM* parameters = SG_MALLOC(LINRECONSTRUCTION_THREAD_PARAM, num_threads);
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
#else
	int32_t num_threads = 1;
#endif 
	// init storages to be used
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
		parameters[t].m_reconstruction_shift = m_reconstruction_shift;
		pthread_create(&threads[t], &attr, run_linearreconstruction_thread, (void*)&parameters[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	LINRECONSTRUCTION_THREAD_PARAM single_thread_param;
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
	single_thread_param.m_reconstruction_shift = m_reconstruction_shift;
	run_linearreconstruction_thread((void*)single_thread_param);
#endif

	// clean
	SG_FREE(id_vector);
	SG_FREE(z_matrix);
	SG_FREE(covariance_matrix);

	// diag(W)=1
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
	SGMatrix<float64_t> M_matrix(N,N);
#ifdef HAVE_PTHREAD
	// allocate threads
	threads = SG_MALLOC(pthread_t, num_threads);
	SPARSEDOT_THREAD_PARAM* parameters_ = SG_MALLOC(SPARSEDOT_THREAD_PARAM, num_threads);
	pthread_attr_t attr_;
	pthread_attr_init(&attr_);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	
	for (t=0; t<num_threads; t++)
	{
		parameters_[t].idx_start = t;
		parameters_[t].idx_step = num_threads;
		parameters_[t].idx_stop = N;
		parameters_[t].N = N;
		parameters_[t].W_matrix = W_matrix;
		parameters_[t].M_matrix = M_matrix.matrix;
		parameters_[t].nz_idxs = nz_idxs;
		pthread_create(&threads[t], &attr_, run_sparsedot_thread, (void*)&parameters_[t]);
	}
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr_);
	SG_FREE(parameters_);
	SG_FREE(threads);
#else
	SPARSEDOT_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_step = 1;
	single_thread_param.idx_stop = N;
	single_thread_param.N = N;
	single_thread_param.W_matrix = W_matrix;
	single_thread_param.M_matrix = M_matrix.matrix;
	single_thread_param.nz_idxs = nz_idxs;
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

SGMatrix<float64_t> CLocallyLinearEmbedding::find_null_space(SGMatrix<float64_t> matrix, int dimension)
{
	int i,j;
	ASSERT(matrix.num_cols==matrix.num_rows);
	int N = matrix.num_cols;
	// get eigenvectors with ARPACK or LAPACK
	int eigenproblem_status = 0;

	float64_t* eigenvalues_vector = NULL;
	float64_t* eigenvectors = NULL;
	float64_t* nullspace_features = NULL;
	if (m_use_arpack)
	{
		#ifndef HAVE_ARPACK
		SG_ERROR("ARPACK is not supported in this configuration.\n");
		#endif
		// using ARPACK (faster)
		eigenvalues_vector = SG_MALLOC(float64_t, dimension+1);
		#ifdef HAVE_ARPACK
		arpack_xsxupd<float64_t>(matrix.matrix,NULL,N,dimension+1,"LA",3,true,m_nullspace_shift,0.0,
		                         eigenvalues_vector,matrix.matrix,eigenproblem_status);
		#endif
		if (eigenproblem_status)
			SG_ERROR("ARPACK failed with code: %d", eigenproblem_status);
		nullspace_features = SG_MALLOC(float64_t, N*dimension);
		for (i=0; i<dimension; i++)
		{
			for (j=0; j<N; j++)
				nullspace_features[j*dimension+i] = matrix[j*(dimension+1)+i+1];
		}
		SG_FREE(eigenvalues_vector);
	}
	else
	{
		// using LAPACK (slower)
		eigenvalues_vector = SG_MALLOC(float64_t, N);
		eigenvectors = SG_MALLOC(float64_t,(dimension+1)*N);
		wrap_dsyevr('V','U',N,matrix.matrix,N,2,dimension+2,eigenvalues_vector,eigenvectors,&eigenproblem_status);
		if (eigenproblem_status)
			SG_ERROR("LAPACK failed with code: %d", eigenproblem_status);
		nullspace_features = SG_MALLOC(float64_t, N*dimension);
		// LAPACKed eigenvectors
		for (i=0; i<dimension; i++)
		{
			for (j=0; j<N; j++)
				nullspace_features[j*dimension+i] = eigenvectors[i*N+j];
		}
		SG_FREE(eigenvectors);
		SG_FREE(eigenvalues_vector);
	}
	return SGMatrix<float64_t>(nullspace_features,dimension,N);
}

void* CLocallyLinearEmbedding::run_linearreconstruction_thread(void* p)
{
	LINRECONSTRUCTION_THREAD_PARAM* parameters = (LINRECONSTRUCTION_THREAD_PARAM*)p;
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
	float64_t m_reconstruction_shift = parameters->m_reconstruction_shift;

	int32_t i,j;
	float64_t norming,trace;

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		// compute local feature matrix containing neighbors of i-th vector
		// center features by subtracting i-th feature column
		for (j=0; j<m_k; j++)
		{
			cblas_dcopy(dim,feature_matrix+neighborhood_matrix[j*N+i]*dim,1,z_matrix+j*dim,1);
			cblas_daxpy(dim,-1.0,feature_matrix+i*dim,1,z_matrix+j*dim,1);
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
				covariance_matrix[j*m_k+j] += m_reconstruction_shift*trace;
		}

		// solve system of linear equations: covariance_matrix * X = 1
		// covariance_matrix is a pos-def matrix
		clapack_dposv(CblasColMajor,CblasLower,m_k,1,covariance_matrix,m_k,id_vector,m_k);

		// normalize weights
		norming=0.0;
		for (j=0; j<m_k; j++)
			norming += id_vector[j];

		cblas_dscal(m_k,-1.0/norming,id_vector,1);

		// put weights into W matrix
		for (j=0; j<m_k; j++)
			W_matrix[N*neighborhood_matrix[j*N+i]+i]=id_vector[j];
	}
	return NULL;
}

SGMatrix<int32_t> CLocallyLinearEmbedding::get_neighborhood_matrix(SGMatrix<float64_t> distance_matrix, int32_t k)
{
	int32_t t;
	int32_t N = distance_matrix.num_rows;
	// init matrix and heap to be used
	int32_t* neighborhood_matrix = SG_MALLOC(int32_t, N*k);
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	NEIGHBORHOOD_THREAD_PARAM* parameters = SG_MALLOC(NEIGHBORHOOD_THREAD_PARAM, num_threads);
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
		parameters[t].m_k = k;
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
	NEIGHBORHOOD_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_step = 1;
	single_thread_param.idx_stop = N;
	single_thread_param.m_k = k;
	single_thread_param.N = N;
	single_thread_param.heap = heaps[0]
	single_thread_param.neighborhood_matrix = neighborhood_matrix;
	single_thread_param.distance_matrix = distance_matrix.matrix;
	run_neighborhood_thread((void*)&single_thread_param);
#endif

	for (t=0; t<num_threads; t++)
		delete heaps[t];
	SG_FREE(heaps);

	return SGMatrix<int32_t>(neighborhood_matrix,k,N);
}

void* CLocallyLinearEmbedding::run_sparsedot_thread(void* p)
{
	SPARSEDOT_THREAD_PARAM* parameters = (SPARSEDOT_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	int32_t N = parameters->N;
	const float64_t* W_matrix = parameters->W_matrix;
	float64_t* M_matrix = parameters->M_matrix;
	DynArray<int32_t>** nz_idxs = parameters->nz_idxs;

	int i,j,k;
	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		for (j=i; j<N; j++)
		{
			M_matrix[i*N+j] = 0.0;
			for (k=0; k<nz_idxs[i]->get_num_elements(); k++)
			{
				M_matrix[i*N+j] += W_matrix[i*N+(*nz_idxs[i])[k]]*W_matrix[j*N+(*nz_idxs[i])[k]];
			}

		}
	}

	return NULL;
}

void* CLocallyLinearEmbedding::run_neighborhood_thread(void* p)
{
	NEIGHBORHOOD_THREAD_PARAM* parameters = (NEIGHBORHOOD_THREAD_PARAM*)p;
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
