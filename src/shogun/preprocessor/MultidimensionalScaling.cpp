/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/MultidimensionalScaling.h>
#ifdef HAVE_LAPACK
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclidianDistance.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct TRIANGULATION_THREAD_PARAM
{
	/// idx of loop start
	int32_t idx_start;
	/// idx of loop stop
	int32_t idx_stop;
	/// idx step of loop
	int32_t idx_step;
	/// number of landmarks
	int32_t lmk_N;
	/// total number of examples
	int32_t total_N;
	/// target dimensionality
	int32_t m_target_dim;
	/// distances from current object to landmarks
	float64_t* current_dist_to_lmks;
	/// feature matrix of landmarks
	float64_t* lmk_feature_matrix;
	/// new feature matrix to write
	float64_t* new_feature_matrix;
	/// const distance matrix
	const float64_t* distance_matrix;
	/// const mean squared distances
	const float64_t* mean_sq_dist_vector;
	/// idxs of landmark examples
	const int32_t* lmk_idxs;
	/// indicates which examples to triangulate
	const bool* to_process;
};
#endif

CMultidimensionalScaling::CMultidimensionalScaling() : CDimensionReductionPreprocessor()
{
	m_eigenvalues = SGVector<float64_t>(NULL,0,false);
	m_landmark_number = 3;
	m_landmark = false;
	
	init();
}

void CMultidimensionalScaling::init()
{
	m_parameters->add(&m_eigenvalues, "eigenvalues", "eigenvalues of last embedding");
	m_parameters->add(&m_landmark, "landmark", "indicates if landmark approximation should be used");
	m_parameters->add(&m_landmark_number, "landmark number", "the number of landmarks for approximation");
}

bool CMultidimensionalScaling::init(CFeatures* features)
{
	return true;
}

void CMultidimensionalScaling::cleanup()
{
}

CMultidimensionalScaling::~CMultidimensionalScaling()
{
	m_eigenvalues.destroy_vector();
}

SGVector<float64_t> CMultidimensionalScaling::get_eigenvalues() const
{
	return m_eigenvalues;
}

void CMultidimensionalScaling::set_landmark_number(int32_t num)
{
	if (num<3)
		SG_ERROR("Number of landmarks should be greater than 3 to make triangulation possible while %d given.",
		         num);
	m_landmark_number = num;
}

int32_t CMultidimensionalScaling::get_landmark_number() const
{
	return m_landmark_number;
}

void CMultidimensionalScaling::set_landmark(bool landmark)
{
	m_landmark = landmark;
}

bool CMultidimensionalScaling::get_landmark() const
{
	return m_landmark;
}

CSimpleFeatures<float64_t>* CMultidimensionalScaling::apply_to_distance(CDistance* distance)
{
	ASSERT(distance);
	// reference distance for not being delete while applying
	SG_REF(distance);

	// compute feature_matrix by landmark or classic embedding of distance matrix
	SGMatrix<float64_t> distance_matrix = distance->get_distance_matrix();
	SGMatrix<float64_t> feature_matrix;
	if (m_landmark)
		feature_matrix = landmark_embedding(distance_matrix);
	else
		feature_matrix = classic_embedding(distance_matrix);
	
	distance_matrix.destroy_matrix();
	CSimpleFeatures<float64_t>* features =
			new CSimpleFeatures<float64_t>(feature_matrix);

	// unreference distance after embedding
	SG_UNREF(distance);
	return features;
}

SGMatrix<float64_t> CMultidimensionalScaling::apply_to_feature_matrix(CFeatures* features)
{
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;
	// reference features for not being deleted while applying
	SG_REF(features);
	
	// compute embedding according to m_landmark value
	SGMatrix<float64_t> new_feature_matrix;
	ASSERT(m_distance);
	m_distance->init(simple_features,simple_features);
	SGMatrix<float64_t> distance_matrix = m_distance->get_distance_matrix();
	if (m_landmark)
		new_feature_matrix = landmark_embedding(distance_matrix);
	else
		new_feature_matrix = classic_embedding(distance_matrix);

	simple_features->set_feature_matrix(new_feature_matrix);

	// delete used distance matrix
	distance_matrix.destroy_matrix();

	// unreference features
	SG_UNREF(features);
	return simple_features->get_feature_matrix();
}

SGVector<float64_t> CMultidimensionalScaling::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

SGMatrix<float64_t> CMultidimensionalScaling::classic_embedding(SGMatrix<float64_t> distance_matrix)
{
	ASSERT(distance_matrix.num_cols==distance_matrix.num_rows);
	int32_t N = distance_matrix.num_cols;

	// loop variables
	int32_t i,j;
	
	// double center distance_matrix
	float64_t dsq;
	for (i=0; i<N; i++)
	{
		for (j=i; j<N; j++)
		{
			dsq = CMath::sq(distance_matrix[i*N+j]);
			distance_matrix[i*N+j] = dsq;
			distance_matrix[j*N+i] = dsq;
		}
	}
	CMath::center_matrix(distance_matrix.matrix,N,N);
	for (i=0; i<N; i++)
	{
		distance_matrix[i*N+i] *= -0.5;
		for (j=i+1; j<N; j++)
		{
			distance_matrix[i*N+j] *= -0.5;
			distance_matrix[j*N+i] *= -0.5;
		}
	}

	// feature matrix representing given distance
	float64_t* replace_feature_matrix = SG_MALLOC(float64_t, N*m_target_dim);
 
	// status of eigenproblem to be solved
	int eigenproblem_status = 0;
#ifdef HAVE_ARPACK
	// using ARPACK
	float64_t* eigenvalues_vector = SG_MALLOC(float64_t, m_target_dim);
	// solve eigenproblem with ARPACK (faster)
	arpack_dsaeupd_wrap(distance_matrix.matrix, NULL, N, m_target_dim, "LM", 1, false, 0.0, 0.0,
	                    eigenvalues_vector, replace_feature_matrix,
	                    eigenproblem_status);
	// check for failure
	ASSERT(eigenproblem_status == 0);
	// reverse eigenvectors order
	float64_t tmp;
	for (j=0; j<N; j++)
	{
		for (i=0; i<m_target_dim/2; i++)
		{
			tmp = replace_feature_matrix[j*m_target_dim+i];
			replace_feature_matrix[j*m_target_dim+i] = 
				replace_feature_matrix[j*m_target_dim+(m_target_dim-i-1)];
			replace_feature_matrix[j*m_target_dim+(m_target_dim-i-1)] = tmp;
		}
	}
	// reverse eigenvalues order
	for (i=0; i<m_target_dim/2; i++)
	{
		tmp = eigenvalues_vector[i];
		eigenvalues_vector[i] = eigenvalues_vector[m_target_dim-i-1];
		eigenvalues_vector[m_target_dim-i-1] = tmp;
	}

	// finally construct embedding
	for (i=0; i<m_target_dim; i++)
	{
		for (j=0; j<N; j++)
			replace_feature_matrix[j*m_target_dim+i] *=
				CMath::sqrt(eigenvalues_vector[i]);
	}
		
	// set eigenvalues vector
	m_eigenvalues.destroy_vector();
	m_eigenvalues = SGVector<float64_t>(eigenvalues_vector,m_target_dim,true);
#else /* not HAVE_ARPACK */
	// using LAPACK
	float64_t* eigenvalues_vector = SG_MALLOC(float64_t, N);
	float64_t* eigenvectors = SG_MALLOC(float64_t, m_target_dim*N);
	// solve eigenproblem with LAPACK
	wrap_dsyevr('V','U',N,distance_matrix.matrix,N,N-m_target_dim+1,N,eigenvalues_vector,eigenvectors,&eigenproblem_status);
	// check for failure
	ASSERT(eigenproblem_status==0);
	
	// set eigenvalues vector
	m_eigenvalues.destroy_vector();
	m_eigenvalues = SGVector<float64_t>(m_target_dim);
	m_eigenvalues.do_free = false;

	// fill eigenvalues vector in backwards order
	for (i=0; i<m_target_dim; i++)
		m_eigenvalues.vector[i] = eigenvalues_vector[m_target_dim-i-1];

	SG_FREE(eigenvalues_vector);

	// construct embedding
	for (i=0; i<m_target_dim; i++)
	{
		for (j=0; j<N; j++)
		{
			replace_feature_matrix[j*m_target_dim+i] = 
			      eigenvectors[(m_target_dim-i-1)*N+j] * CMath::sqrt(m_eigenvalues.vector[i]);
		}
	}
	SG_FREE(eigenvectors);
#endif /* HAVE_ARPACK else */
	
	// warn user if there are negative or zero eigenvalues
	for (i=0; i<m_eigenvalues.vlen; i++)
	{
		if (m_eigenvalues.vector[i]<=0.0)
		{
			SG_WARNING("Embedding is not consistent (got neg eigenvalues): features %d-%d are wrong",
			           i, m_eigenvalues.vlen-1);
			break;
		}
	}
	
	return SGMatrix<float64_t>(replace_feature_matrix,m_target_dim,N);
}

SGMatrix<float64_t> CMultidimensionalScaling::landmark_embedding(SGMatrix<float64_t> distance_matrix)
{
	ASSERT(distance_matrix.num_cols==distance_matrix.num_rows);
	int32_t lmk_N = m_landmark_number;
	int32_t i,j,t;
	int32_t total_N = distance_matrix.num_cols;
	if (lmk_N<3)
	{
		SG_ERROR("Number of landmarks (%d) should be greater than 3 for proper triangulation.\n", 
		         lmk_N);
	}
	if (lmk_N>total_N)
	{
		SG_ERROR("Number of landmarks (%d) should be less than total number of vectors (%d).\n",
		         lmk_N, total_N);
	}
	
	// get landmark indexes with random permutation
	SGVector<int32_t> lmk_idxs = shuffle(lmk_N,total_N);
	// compute distances between landmarks
	float64_t* lmk_dist_matrix = SG_MALLOC(float64_t, lmk_N*lmk_N);
	for (i=0; i<lmk_N; i++)
	{
		for (j=0; j<lmk_N; j++)
			lmk_dist_matrix[i*lmk_N+j] =
				distance_matrix[lmk_idxs.vector[i]*total_N+lmk_idxs.vector[j]];
	}

	// get landmarks embedding
	SGMatrix<float64_t> lmk_dist_sgmatrix(lmk_dist_matrix,lmk_N,lmk_N);
	// compute mean vector of squared distances
	float64_t* mean_sq_dist_vector = SG_CALLOC(float64_t, lmk_N);
	for (i=0; i<lmk_N; i++)
	{
		for (j=0; j<lmk_N; j++)
			mean_sq_dist_vector[i] += CMath::sq(lmk_dist_matrix[i*lmk_N+j]);

		mean_sq_dist_vector[i] /= lmk_N;
	}	
	SGMatrix<float64_t> lmk_feature_matrix = classic_embedding(lmk_dist_sgmatrix);

	lmk_dist_sgmatrix.destroy_matrix();

	// construct new feature matrix
	float64_t* new_feature_matrix = SG_CALLOC(float64_t, m_target_dim*total_N);

	// fill new feature matrix with embedded landmarks
	for (i=0; i<lmk_N; i++)
	{
		for (j=0; j<m_target_dim; j++)
			new_feature_matrix[lmk_idxs.vector[i]*m_target_dim+j] =
				lmk_feature_matrix[i*m_target_dim+j];
	}

	// get exactly defined pseudoinverse of landmarks feature matrix
	ASSERT(m_eigenvalues.vector && m_eigenvalues.vlen == m_target_dim);
	for (i=0; i<lmk_N; i++)
	{
		for (j=0; j<m_target_dim; j++)
			lmk_feature_matrix[i*m_target_dim+j] /= m_eigenvalues.vector[j];
	}


	// set to_process els true if should be processed
	bool* to_process = SG_MALLOC(bool, total_N);
	for (j=0; j<total_N; j++)
		to_process[j] = true;
	for (j=0; j<lmk_N; j++)
		to_process[lmk_idxs.vector[j]] = false;

	// get embedding for non-landmark vectors
#ifdef HAVE_PTHREAD
	int32_t num_threads = parallel->get_num_threads();
	ASSERT(num_threads>0);
	// allocate threads and it's parameters
	pthread_t* threads = SG_MALLOC(pthread_t, num_threads);
	TRIANGULATION_THREAD_PARAM* parameters = SG_MALLOC(TRIANGULATION_THREAD_PARAM, num_threads);
	pthread_attr_t attr;
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	float64_t* current_dist_to_lmks = SG_MALLOC(float64_t, lmk_N*num_threads);
	// run threads
	for (t=0; t<num_threads; t++)
	{
		parameters[t].idx_start = t;
		parameters[t].idx_stop = total_N;
		parameters[t].idx_step = num_threads;
		parameters[t].lmk_N = lmk_N;
		parameters[t].total_N = total_N;
		parameters[t].m_target_dim = m_target_dim;
		parameters[t].current_dist_to_lmks = current_dist_to_lmks+t*lmk_N;
		parameters[t].distance_matrix = distance_matrix.matrix;
		parameters[t].mean_sq_dist_vector = mean_sq_dist_vector;
		parameters[t].lmk_idxs = lmk_idxs.vector;
		parameters[t].lmk_feature_matrix = lmk_feature_matrix.matrix;
		parameters[t].new_feature_matrix = new_feature_matrix;
		parameters[t].to_process = to_process;
		pthread_create(&threads[t], &attr, run_triangulation_thread, (void*)&parameters[t]);
	}
	// join threads
	for (t=0; t<num_threads; t++)
		pthread_join(threads[t], NULL);
	pthread_attr_destroy(&attr);
	SG_FREE(parameters);
	SG_FREE(threads);
#else
	// run single 'thread'
	float64_t* current_dist_to_lmks = SG_MALLOC(float64_t, lmk_N);
	TRIANGULATION_THREAD_PARAM single_thread_param;
	single_thread_param.idx_start = 0;
	single_thread_param.idx_stop = total_N;
	single_thread_param.idx_step = 1;
	single_thread_param.lmk_N = lmk_N;
	single_thread_param.total_N = total_N;
	single_thread_param.m_target_dim = m_target_dim;
	single_thread_param.current_dist_to_lmks = current_dist_to_lmks;
	single_thread_param.distance_matrix = distance_matrix.matrix;
	single_thread_param.mean_sq_dist_vector = mean_sq_dist_vector;
	single_thread_param.lmk_idxs = lmk_idxs.vector;
	single_thread_param.lmk_feature_matrix = lmk_feature_matrix.matrix;
	single_thread_param.new_feature_matrix = new_feature_matrix;
	single_thread_param.to_process = to_process;
	run_triangulation_thread((void*)&single_thread_param);
#endif
	// cleanup
	lmk_feature_matrix.destroy_matrix();
	SG_FREE(current_dist_to_lmks);
	lmk_idxs.destroy_vector();
	SG_FREE(mean_sq_dist_vector);
	SG_FREE(to_process);
	lmk_idxs.destroy_vector();

	return SGMatrix<float64_t>(new_feature_matrix,m_target_dim,total_N);
}

void* CMultidimensionalScaling::run_triangulation_thread(void* p)
{
	TRIANGULATION_THREAD_PARAM* parameters = (TRIANGULATION_THREAD_PARAM*)p;
	int32_t idx_start = parameters->idx_start;
	int32_t idx_step = parameters->idx_step;
	int32_t idx_stop = parameters->idx_stop;
	const int32_t* lmk_idxs = parameters->lmk_idxs;
	const float64_t* distance_matrix = parameters->distance_matrix;
	const float64_t* mean_sq_dist_vector = parameters->mean_sq_dist_vector;
	float64_t* current_dist_to_lmks = parameters->current_dist_to_lmks;
	int32_t m_target_dim = parameters->m_target_dim;
	int32_t lmk_N = parameters->lmk_N;
	int32_t total_N = parameters->total_N;
	const bool* to_process = parameters->to_process;
	float64_t* lmk_feature_matrix = parameters->lmk_feature_matrix;
	float64_t* new_feature_matrix = parameters->new_feature_matrix;

	int32_t i,k;
	for (i=idx_start; i<idx_stop; i+=idx_step)
	{
		// skip if landmark
		if (!to_process[i])
			continue;

		// compute difference from mean landmark distance vector
		for (k=0; k<lmk_N; k++)
		{
			current_dist_to_lmks[k] =
				CMath::sq(distance_matrix[i*total_N+lmk_idxs[k]]) -
				mean_sq_dist_vector[k];
		}
		// compute embedding
		cblas_dgemv(CblasColMajor,CblasNoTrans,
		            m_target_dim,lmk_N,
		            -0.5,lmk_feature_matrix,m_target_dim,
		            current_dist_to_lmks,1,
		            0.0,(new_feature_matrix+i*m_target_dim),1);
	}
	return NULL;
}


SGVector<int32_t> CMultidimensionalScaling::shuffle(int32_t count, int32_t total_count)
{
	int32_t* idxs = SG_MALLOC(int32_t, total_count);
	int32_t i,rnd;
	int32_t* permuted_idxs = SG_MALLOC(int32_t, count);

	// reservoir sampling
	for (i=0; i<total_count; i++)
		idxs[i] = i;
	for (i=0; i<count; i++)
		permuted_idxs[i] = idxs[i];
	for (i=count; i<total_count; i++)
	{
		rnd = CMath::random(1,i);
		if (rnd<count)
			permuted_idxs[rnd] = idxs[i];
	}
	SG_FREE(idxs);

	CMath::qsort(permuted_idxs,count);
	return SGVector<int32_t>(permuted_idxs, count);
}

#endif /* HAVE_LAPACK */
