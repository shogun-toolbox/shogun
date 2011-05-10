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

CLLE::CLLE() : CSimplePreProc<float64_t>(), m_k(5)
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

	int32_t dim = pdata->get_num_features();
	int32_t N = pdata->get_num_vectors();
	ASSERT(m_k<N);

	// output
	SG_PRINT("DIM = %d\n", dim);
	SG_PRINT("N = %d\n", N);

	// loop variables
	int32_t i,j,k;

	// compute distance matrix
	ASSERT(m_distance);
	m_distance->init(data,data);
	float64_t* distance_matrix = new float64_t[N*N];
	float64_t d;
	for (i=0; i<N; i++)
		for (j=0; j<=i; j++)
		{
			d = m_distance->distance(i,j);
			distance_matrix[i*N+j] = d;
			distance_matrix[j*N+i] = d;
		}

	// output matrix
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			SG_PRINT("%5.3f ",distance_matrix[i*N+j]);
		}
		SG_PRINT("\n");
	}

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

	// output matrix
	for (i=0; i<N; i++)
	{
		SG_PRINT("%dth \n",i);
		for (j=0; j<m_k; j++)
			SG_PRINT("%d ", neighborhood_matrix[i*m_k+j]);
		SG_PRINT("\n");
	}

	float64_t* z_matrix = new float64_t[N*m_k];
	float64_t* covariance_matrix = new float64_t[N*N];
	float64_t* feature_vector = new float64_t[dim];

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

		// output Z matrix
		SG_PRINT("%dth Z matrix\n", i);
		for (j=0; j<dim; j++)
		{
			for (k=0; k<m_k; k++)
			{
				SG_PRINT("[%d] %5.3f ", j*m_k+k, z_matrix[j*m_k+k]);
			}
			SG_PRINT("\n");
		}

		cblas_dgemm(CblasRowMajor,CblasTrans, CblasNoTrans,
					m_k,m_k,dim,1.0,
					z_matrix,m_k,
					z_matrix,m_k,
					0.0,
					covariance_matrix,m_k);

		SG_PRINT("%dth covariance matrix\n", i);
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
			{
				SG_PRINT("[%d] %5.3f ", j*m_k+k, covariance_matrix[j*m_k+k]);
			}
			SG_PRINT("\n");
		}

		float64_t* id_vector = new float64_t[m_k];
		int32_t* ipiv = new int32_t[m_k];

		for (j=0; j<m_k; j++)
		{
			id_vector[j] = 1.0;
		}

		// regularize
		for (j=0; j<m_k; j++)
		{
			covariance_matrix[j*m_k+j] += 0.5;
		}

		clapack_dgesv(CblasRowMajor,m_k,1,covariance_matrix,m_k,ipiv,id_vector,m_k);

		SG_PRINT("%dth covariance matrix\n", i);
		for (j=0; j<m_k; j++)
		{
			for (k=0; k<m_k; k++)
			{
				SG_PRINT("[%d] %5.3f ", j*m_k+k, covariance_matrix[j*m_k+k]);
			}
			SG_PRINT("\n");
		}


		SG_PRINT("weights\n");

		for (j=0; j<m_k; j++)
		{
			SG_PRINT("[%d] %5.3f ", j, id_vector[j]);
		}
		SG_PRINT("\n");

		// TODO eigenproblem
	}

	delete[] neighborhood_matrix;
	delete[] distance_matrix;
	delete[] local_distances;
	delete[] local_neighbors;
	delete[] feature_vector;
	delete[] covariance_matrix;
	delete[] z_matrix;

	SG_PRINT("cleared\n");
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
