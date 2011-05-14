/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "preproc/LLE.h"
#include "lib/lapack.h"
#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "distance/EuclidianDistance.h"
#include "lib/Signal.h"

using namespace shogun;

CLLE::CLLE() : CSimplePreProc<float64_t>(), m_k(2)
{
	// temporary hack. which one will make sense?
	m_distance = new CEuclidianDistance();
}

CLLE::~CLLE()
{
	delete m_distance;
}

bool CLLE::init(CFeatures* data)
{
	CSimpleFeatures<float64_t>* pdata = (CSimpleFeatures<float64_t>*) data;

	ASSERT(pdata);
	int32_t dim = pdata->get_num_features();
	int32_t N = pdata->get_num_vectors();
	ASSERT(m_k<N);

	// loop variables
	int32_t i,j,k;

	// compute distance matrix
	ASSERT(m_distance);
	m_distance->init(data,data);
	float64_t* distance_matrix = new float64_t[N*N];
	float64_t pairwise_distance;
	for (i=0; i<N; i++)
		for (j=0; j<=i; j++)
		{
			pairwise_distance = m_distance->distance(i,j);
			distance_matrix[i*N+j] = pairwise_distance;
			distance_matrix[j*N+i] = pairwise_distance;
		}

	// init matrices to be used
	int32_t* neighborhood_matrix = new int32_t[N*m_k];
	float64_t* local_distances = new float64_t[N];
	int32_t* local_neighbors = new int32_t[N];

	// construct neighborhood matrix (contains idxs of neighbors for
	// i-th object in i-th row)
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			local_neighbors[j] = j;
			local_distances[j] = distance_matrix[i*N+j];
		}

		CMath::qsort_index(local_distances,local_neighbors,N);

		for (j=0; j<m_k; j++)
			neighborhood_matrix[i*m_k+j] = local_neighbors[j+1];
	}

	delete[] distance_matrix;
	delete[] local_neighbors;
	delete[] local_distances;

	float64_t* W_matrix = new float64_t[N*N];
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			W_matrix[i*N+j]=0.0;

	float64_t* z_matrix = new float64_t[N*m_k];
	float64_t* covariance_matrix = new float64_t[N*N];
	float64_t* feature_vector = new float64_t[dim];
	float64_t* id_vector = new float64_t[m_k];
	int32_t* ipiv = new int32_t[m_k];

	for (i=0; i<N; i++)
	{
		// compute local feature matrix containing neighbors of i-th vector
		for (j=0; j<m_k; j++)
		{
			pdata->get_feature_vector(&feature_vector, &dim, neighborhood_matrix[i*m_k+j]);
			for (k=0; k<dim; k++)
			{
				z_matrix[k*m_k+j] = feature_vector[k];
			}
		}

		// get i-th feature vector
		pdata->get_feature_vector(&feature_vector, &dim, i);

		// center features by subtracting
		for (j=0; j<m_k; j++)
			for (k=0; k<dim; k++)
			{
				z_matrix[k*m_k+j] -= feature_vector[k];
			}

		cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans,
		            m_k,m_k,dim,1.0,
		            z_matrix,m_k,
		            z_matrix,m_k,
		            0.0,
		            covariance_matrix,m_k);

		for (j=0; j<m_k; j++)
		{
			id_vector[j] = 1.0;
		}

		// regularize
		if (m_k>dim)
		{
			// compute tr(C)
			float64_t trace = 0.0;
			for (j=0; j<m_k; j++)
			{
				trace += covariance_matrix[j*m_k+j];
			}

			for (j=0; j<m_k; j++)
			{
				covariance_matrix[j*m_k+j] += 1e-3*trace;
			}
		}

		clapack_dgesv(CblasRowMajor,
		              m_k,1,
		              covariance_matrix,m_k,
		              ipiv,//memset(W_matrix,0,N*N);
		              id_vector,m_k);

		// normalize
		float64_t normalizer=0.0;
		for (j=0; j<m_k; j++)
			normalizer += CMath::abs(id_vector[j]);

		for (j=0; j<m_k; j++)
			id_vector[j]/=normalizer;

		// fill W matrix
		for (j=0; j<m_k; j++)
			W_matrix[i*N+neighborhood_matrix[i*m_k+j]]=id_vector[j];

	}

	delete[] ipiv;
	delete[] id_vector;
	delete[] neighborhood_matrix;
	delete[] feature_vector;
	delete[] z_matrix;
	delete[] covariance_matrix;

	// W=I-W
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			W_matrix[i*N+j] = i==j ? -W_matrix[i*N+j] : 1.0-W_matrix[i*N+j];

	// compute W'*W
	float64_t* M_matrix = new float64_t[N*N];
	cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans,
	            N,N,N,1.0,
	            W_matrix,N,
	            W_matrix,N,
	            0.0,
	            M_matrix,N);

	delete[] W_matrix;

	// compute eigenvectors
	float64_t* eigs = new float64_t[N];
	int32_t status = 0;
	wrap_dsyev('V','U',N,M_matrix,N,eigs,&status);

	for (i=0; i<N; i++)
	{
		SG_PRINT("%f :", eigs[i]);

		for (j=0; j<N; j++)
		{
			SG_PRINT("%5.3f ",M_matrix[i*N+j]);
		}
		SG_PRINT("\n");
	}

	delete[] M_matrix;

	return true;
}

void CLLE::cleanup()
{

}

float64_t* CLLE::apply_to_feature_matrix(CFeatures* f)
{
	return 0;
}

float64_t* CLLE::apply_to_feature_vector(float64_t* f, int32_t &len)
{
	return 0;
}
