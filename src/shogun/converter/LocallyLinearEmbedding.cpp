/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/lib/CoverTree.h>
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
	/// matrix containing indexes of neighbors of ith object in ith column
	SGMatrix<int32_t> neighborhood_matrix;
	/// old feature matrix
	SGMatrix<float64_t> feature_matrix;
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

class LLE_COVERTREE_POINT
{
public:

	LLE_COVERTREE_POINT(int32_t index, const SGMatrix<float64_t>& dmatrix)
	{
		point_index = index;
		distance_matrix = dmatrix;
	}

	inline double distance(const LLE_COVERTREE_POINT& p) const
	{
		return distance_matrix[point_index*distance_matrix.num_rows+p.point_index];
	}

	inline bool operator==(const LLE_COVERTREE_POINT& p) const
	{
		return (p.point_index==point_index);
	}

	int32_t point_index;
	SGMatrix<float64_t> distance_matrix;
};

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

CLocallyLinearEmbedding::CLocallyLinearEmbedding() :
		CEmbeddingConverter()
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

CFeatures* CLocallyLinearEmbedding::apply(CFeatures* features)
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
	CTime* time = new CTime();
	time->start();
	m_distance->init(simple_features,simple_features);
	SGMatrix<float64_t> distance_matrix = m_distance->get_distance_matrix();
	m_distance->remove_lhs_and_rhs();
	SG_DEBUG("Distance matrix computation took %fs\n",time->cur_time_diff());
	SG_DEBUG("Calculating neighborhood matrix\n");
	SGMatrix<int32_t> neighborhood_matrix;

	time->start();
	if (m_auto_k)
	{
		neighborhood_matrix = get_neighborhood_matrix(distance_matrix,m_max_k);
		m_k = estimate_k(simple_features,neighborhood_matrix);
		SG_DEBUG("Estimated k with value of %d\n",m_k);
	}
	else
		neighborhood_matrix = get_neighborhood_matrix(distance_matrix,m_k);

	SG_DEBUG("Neighbors finding took %fs\n",time->cur_time_diff());

	// init W (weight) matrix
	float64_t* W_matrix = distance_matrix.matrix;
	memset(W_matrix,0,sizeof(float64_t)*N*N);

	// construct weight matrix
	SG_DEBUG("Constructing weight matrix\n");
	time->start();
	SGMatrix<float64_t> weight_matrix = construct_weight_matrix(simple_features,W_matrix,neighborhood_matrix);
	SG_DEBUG("Weight matrix construction took %.5fs\n", time->cur_time_diff());
	neighborhood_matrix.destroy_matrix();

	// find null space of weight matrix
	SG_DEBUG("Finding nullspace\n");
	time->start();
	SGMatrix<float64_t> new_feature_matrix = construct_embedding(weight_matrix,m_target_dim);
	SG_DEBUG("Eigenproblem solving took %.5fs\n", time->cur_time_diff());
	delete time;
	weight_matrix.destroy_matrix();

	SG_UNREF(features);
	return (CFeatures*)(new CSimpleFeatures<float64_t>(new_feature_matrix));
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
	for (i=CMath::random(0,20); i<N; i+=N/20)
	{
		for (j=0; j<k; j++)
		{
			cblas_dcopy(dim,feature_matrix+neighborhood_matrix[i*m_k+j]*dim,1,z_matrix+j*dim,1);
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

SGMatrix<float64_t> CLocallyLinearEmbedding::construct_weight_matrix(CSimpleFeatures<float64_t>* simple_features,
                                                                     float64_t* W_matrix, SGMatrix<int32_t> neighborhood_matrix)
{
	int32_t N = simple_features->get_num_vectors();
	int32_t dim = simple_features->get_num_features();
#ifdef HAVE_PTHREAD
	int32_t t;
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
		parameters[t].neighborhood_matrix = neighborhood_matrix;
		parameters[t].z_matrix = z_matrix+(m_k*dim)*t;
		parameters[t].feature_matrix = feature_matrix;
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
	single_thread_param.neighborhood_matrix = neighborhood_matrix;
	single_thread_param.z_matrix = z_matrix;
	single_thread_param.feature_matrix = feature_matrix;
	single_thread_param.covariance_matrix = covariance_matrix;
	single_thread_param.id_vector = id_vector;
	single_thread_param.W_matrix = W_matrix;
	single_thread_param.m_reconstruction_shift = m_reconstruction_shift;
	run_linearreconstruction_thread((void*)&single_thread_param);
#endif

	// clean
	SG_FREE(id_vector);
	SG_FREE(z_matrix);
	SG_FREE(covariance_matrix);

	return SGMatrix<float64_t>(W_matrix,N,N);
}

SGMatrix<float64_t> CLocallyLinearEmbedding::construct_embedding(SGMatrix<float64_t> matrix,int dimension)
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
		arpack_dsxupd(matrix.matrix,NULL,false,N,dimension+1,"LA",true,3,true,false,m_nullspace_shift,0.0,
		              eigenvalues_vector,matrix.matrix,eigenproblem_status);
		matrix.num_rows = dimension+1;
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
	SGMatrix<int32_t> neighborhood_matrix = parameters->neighborhood_matrix;
	float64_t* z_matrix = parameters->z_matrix;
	SGMatrix<float64_t> feature_matrix = parameters->feature_matrix;
	float64_t* covariance_matrix = parameters->covariance_matrix;
	float64_t* id_vector = parameters->id_vector;
	float64_t* W_matrix = parameters->W_matrix;
	float64_t m_reconstruction_shift = parameters->m_reconstruction_shift;

	int32_t i,j,k;
	int32_t dim = feature_matrix.num_rows;
	int32_t N = feature_matrix.num_cols;
	int32_t actual_k = neighborhood_matrix.num_rows;
	float64_t norming,trace;

	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		// compute local feature matrix containing neighbors of i-th vector
		// center features by subtracting i-th feature column
		for (j=0; j<m_k; j++)
		{
			cblas_dcopy(dim,feature_matrix.matrix+neighborhood_matrix[i*actual_k+j]*dim,1,z_matrix+j*dim,1);
			cblas_daxpy(dim,-1.0,feature_matrix.matrix+i*dim,1,z_matrix+j*dim,1);
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

		cblas_dscal(m_k,1.0/norming,id_vector,1);

		memset(covariance_matrix,0,sizeof(float64_t)*m_k*m_k);
		cblas_dger(CblasColMajor,m_k,m_k,1.0,id_vector,1,id_vector,1,covariance_matrix,m_k);

		// put weights into W matrix
		W_matrix[N*i+i] += 1.0;
		for (j=0; j<m_k; j++)
		{
			W_matrix[N*i+neighborhood_matrix[i*actual_k+j]] -= id_vector[j];
			W_matrix[N*neighborhood_matrix[i*actual_k+j]+i] -= id_vector[j];
		}
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
				W_matrix[N*neighborhood_matrix[i*actual_k+j]+neighborhood_matrix[i*actual_k+k]]+=
				  covariance_matrix[j*m_k+k];
		}
	}
	return NULL;
}

SGMatrix<int32_t> CLocallyLinearEmbedding::get_neighborhood_matrix(SGMatrix<float64_t> distance_matrix, int32_t k)
{
	int32_t i;
	int32_t N = distance_matrix.num_rows;

	int32_t* neighborhood_matrix = SG_MALLOC(int32_t, N*k);

	float64_t max_dist = CMath::max(distance_matrix.matrix,N*N);

	CoverTree<LLE_COVERTREE_POINT>* coverTree = new CoverTree<LLE_COVERTREE_POINT>(max_dist);

	for (i=0; i<N; i++)
		coverTree->insert(LLE_COVERTREE_POINT(i,distance_matrix));

	for (i=0; i<N; i++)
	{
		std::vector<LLE_COVERTREE_POINT> neighbors =
		   coverTree->kNearestNeighbors(LLE_COVERTREE_POINT(i,distance_matrix),k+1);
		for (std::size_t m=1; m<unsigned(k+1); m++)
			neighborhood_matrix[i*k+m-1] = neighbors[m].point_index;
	}

	delete coverTree;

	return SGMatrix<int32_t>(neighborhood_matrix,k,N);
}
#endif /* HAVE_LAPACK */
