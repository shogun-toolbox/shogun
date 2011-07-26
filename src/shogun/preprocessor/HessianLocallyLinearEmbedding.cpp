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
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/EuclidianDistance.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

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

SGMatrix<float64_t> CHessianLocallyLinearEmbedding::apply_to_feature_matrix(CFeatures* features)
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
	int32_t i,j,k,l;

	int32_t dp = m_target_dim*(m_target_dim+1)/2;
	ASSERT(m_k>=(1+m_target_dim+dp));

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

	SG_FREE(distance_matrix.matrix);
	SG_FREE(local_neighbors_idxs);

	// init W (weight) matrix
	float64_t* W_matrix = new float64_t[N*N];
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			W_matrix[i*N+j]=0.0;

	// init matrices and norm factor to be used
	float64_t* local_feature_matrix = new float64_t[m_k*dim];
	float64_t* s_values_vector = new float64_t[dim];
	float64_t* tau = new float64_t[CMath::min((1+m_target_dim+dp),m_k)];
	float64_t* mean_vector = new float64_t[dim];
	float64_t* q_matrix = new float64_t[m_k*m_k];
	float64_t* w_sum_vector = new float64_t[dp];

	// Yi
	float64_t* Yi_matrix = new float64_t[m_k*(1+m_target_dim+dp)];
	// get feature matrix
	SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();

	for (i=0; i<N && !(CSignal::cancel_computations()); i++)
	{
		// Yi(:,0) = 1
		for (j=0; j<m_k; j++)
			Yi_matrix[j] = 1.0;

		// fill mean vector with zeros
		for (j=0; j<dim; j++)
			mean_vector[j] = 0.0;

		// compute local feature matrix containing neighbors of i-th vector
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<dim; k++)
			{
				local_feature_matrix[j*dim+k] = feature_matrix.matrix[neighborhood_matrix[j*N+i]*dim+k];
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

		// Yi(0:m_k,1:1+m_target_dim) = Vh(0:m_k, 0:target_dim)
		for (j=0; j<m_target_dim; j++)
		{
			for (k=0; k<m_k; k++)
				Yi_matrix[(j+1)*m_k+k] = local_feature_matrix[k*dim+j];
		}

		int32_t ct = 0;
		
		// construct 2nd order hessian approx
		for (j=0; j<m_target_dim; j++)
		{
			for (k=0; k<m_target_dim-j; k++)
			{
				for (l=0; l<m_k; l++)
				{
					Yi_matrix[(ct+k+1+m_target_dim)*m_k+l] = Yi_matrix[(j+1)*m_k+l]*Yi_matrix[(j+k+1)*m_k+l];
				}
			}
			ct += ct + m_target_dim - j;
		}
	
		// perform QR factorization
		wrap_dgeqrf(m_k,(1+m_target_dim+dp),Yi_matrix,m_k,tau,&info);
		ASSERT(info==0);
		wrap_dorgqr(m_k,(1+m_target_dim+dp),CMath::min((1+m_target_dim+dp),m_k),Yi_matrix,m_k,tau,&info);
		ASSERT(info==0);
		
		float64_t* Pii = (Yi_matrix+m_k*(1+m_target_dim));

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
		
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
				W_matrix[N*neighborhood_matrix[k*N+i]+neighborhood_matrix[j*N+i]] += q_matrix[j*m_k+k];
		}
	}

	// clean
	SG_FREE(Yi_matrix);
	SG_FREE(s_values_vector);
	SG_FREE(mean_vector);
	SG_FREE(neighborhood_matrix);
	SG_FREE(local_feature_matrix);
	SG_FREE(q_matrix);

	// finally construct embedding
	SGMatrix<float64_t> W_sgmatrix = SGMatrix<float64_t>(W_matrix,N,N,true);
	simple_features->set_feature_matrix(find_null_space(W_sgmatrix,m_target_dim,false));
	W_sgmatrix.free_matrix();

	return simple_features->get_feature_matrix();
}

SGVector<float64_t> CHessianLocallyLinearEmbedding::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

#endif /* HAVE_LAPACK */
