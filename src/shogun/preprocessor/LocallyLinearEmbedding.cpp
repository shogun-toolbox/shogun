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
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CLocallyLinearEmbedding::CLocallyLinearEmbedding() :
		CDimensionReductionPreprocessor(), m_k(3)
{
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
	ASSERT(simple_features);

	// get dimensionality and number of vectors of data
	int32_t dim = simple_features->get_num_features();
	ASSERT(m_target_dim<=dim);
	int32_t N = simple_features->get_num_vectors();
	ASSERT(m_k<N);

	// loop variables
	int32_t i,j,k;

	// compute distance matrix
	CDistance* distance = new CEuclidianDistance(simple_features,simple_features);
	SGMatrix<float64_t> distance_matrix = distance->get_distance_matrix();
	delete distance;

	// init matrices to be used
	int32_t* neighborhood_matrix = new int32_t[N*m_k];
	int32_t* local_neighbors_idxs = new int32_t[N];

	// construct neighborhood matrix (contains idxs of neighbors for
	// i-th object in i-th column)
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			local_neighbors_idxs[j] = j;
		}

		CMath::qsort_index(distance_matrix.matrix+(i*N),local_neighbors_idxs,N);

		for (j=0; j<m_k; j++)
			neighborhood_matrix[j*N+i] = local_neighbors_idxs[j+1];
	}

	delete[] distance_matrix.matrix;
	delete[] local_neighbors_idxs;

	// init W (weight) matrix
	float64_t* W_matrix = new float64_t[N*N];
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			W_matrix[i*N+j]=0.0;

	// init matrices and norm factor to be used
	float64_t* z_matrix = new float64_t[m_k*dim];
	float64_t* covariance_matrix = new float64_t[m_k*m_k];
	float64_t* id_vector = new float64_t[m_k];
	float64_t norming;

	// get feature matrix
	SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();

	for (i=0; i<N && !(CSignal::cancel_computations()); i++)
	{
		// compute local feature matrix containing neighbors of i-th vector
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
				z_matrix[j*dim+k] = feature_matrix.matrix[neighborhood_matrix[j*N+i]*dim+k];
		}

		// center features by subtracting i-th feature column
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
				z_matrix[j*dim+k] -= feature_matrix.matrix[i*dim+k];
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
			float64_t trace = 0.0;
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

	// clean
	delete[] id_vector;
	delete[] neighborhood_matrix;
	delete[] z_matrix;
	delete[] covariance_matrix;

	// W=I-W
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
			W_matrix[j*N+i] = (i==j) ? 1.0-W_matrix[j*N+i] : -W_matrix[j*N+i];
	}

	// compute M=(W-I)'*(W-I)
	SGMatrix<float64_t> M_matrix = SGMatrix<float64_t>(new float64_t[N*N],N,N,true);
	cblas_dgemm(CblasColMajor,CblasTrans, CblasNoTrans,
	            N,N,N,
	            1.0,W_matrix,N,
	            W_matrix,N,
	            0.0,M_matrix.matrix,N);

	delete[] W_matrix;

	simple_features->set_feature_matrix(find_null_space(M_matrix,m_target_dim,true));
	M_matrix.free_matrix();

	return simple_features->get_feature_matrix();
}

SGVector<float64_t> CLocallyLinearEmbedding::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

SGMatrix<float64_t> CLocallyLinearEmbedding::find_null_space(SGMatrix<float64_t> matrix, int dimension, bool force_lapack = false)
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

	if (arpack)
	{
		// using ARPACK (faster)
		eigenvalues_vector = new float64_t[dimension+1];
		arpack_dsaupd(matrix.matrix,N,dimension+1,"LA",3,-1e-3,false,eigenvalues_vector,matrix.matrix,eigenproblem_status);
	}
	else
	{
		// using LAPACK (slower)
		eigenvalues_vector = new float64_t[N];
		wrap_dsyev('V','U',N,matrix.matrix,N,eigenvalues_vector,&eigenproblem_status);
	}

	// check if failed
	if (eigenproblem_status)
		SG_ERROR("Eigenproblem failed with code: %d", eigenproblem_status);
	
	// allocate null space feature matrix
	float64_t* null_space_features = new float64_t[N*dimension];

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
				null_space_features[j*dimension+i] = matrix.matrix[(i+1)*N+j];
		}
	}
	delete[] eigenvalues_vector;

	return SGMatrix<float64_t>(null_space_features,dimension,N);
}

#endif /* HAVE_LAPACK */
