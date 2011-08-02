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

using namespace shogun;

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
	SGMatrix<int32_t> neighborhood_matrix = get_neighborhood_matrix(distance);

	// init W (weight) matrix
	float64_t* W_matrix = SG_CALLOC(float64_t, N*N);

	// init matrices and norm factor to be used
	float64_t* local_feature_matrix = SG_MALLOC(float64_t, m_k*dim);
	float64_t* mean_vector = SG_MALLOC(float64_t, dim);
	float64_t* q_matrix = SG_MALLOC(float64_t, m_k*m_k);
	float64_t* s_values_vector = SG_MALLOC(float64_t, dim);

	// G
	float64_t* G_matrix = SG_MALLOC(float64_t, m_k*(1+m_target_dim));
	// get feature matrix
	SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();

	for (i=0; i<N && !(CSignal::cancel_computations()); i++)
	{
		// Yi(:,0) = 1
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
				local_feature_matrix[j*dim+k] = feature_matrix.matrix[neighborhood_matrix.matrix[j*N+i]*dim+k];
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
		for (j=0; j<m_k; j++)
		{
			W_matrix[N*neighborhood_matrix.matrix[j*N+i]+neighborhood_matrix.matrix[j*N+i]] += 1.0;
			for (k=0; k<m_k; k++)
				W_matrix[N*neighborhood_matrix.matrix[k*N+i]+neighborhood_matrix.matrix[j*N+i]] -= q_matrix[j*m_k+k];
		}
	}

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

	return simple_features->get_feature_matrix();
}

SGVector<float64_t> CLocalTangentSpaceAlignment::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

#endif /* HAVE_LAPACK */
