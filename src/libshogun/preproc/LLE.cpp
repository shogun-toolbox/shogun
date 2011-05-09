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

CLLE::CLLE() : CSimplePreProc<float64_t>(), k(3)
{

}

CLLE::~CLLE()
{

}

bool CLLE::init(CFeatures* data)
{
	CSimpleFeatures<float64_t>* pdata = (CSimpleFeatures<float64_t>*) data;

	int32_t dim = pdata->get_num_features();
	SG_PRINT("DIM = %d\n", dim);
	int32_t N = pdata->get_num_vectors();
	ASSERT(k<N);
	SG_PRINT("N = %d\n", N);
	int32_t i,j;

	// oh it is so dirty
	CDistance* dist = new CEuclidianDistance();
	dist->init(pdata,pdata);
	ASSERT(dist);

	// get distances
	float64_t* distances = new float64_t[N*N];
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			distances[i*N+j] = dist->distance(i,j);
		}
	}

	// output matrix
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			SG_PRINT("%5.3f ",distances[i*N+j]);
		}
		SG_PRINT("\n");
	}

	int32_t* neighborhood = new int32_t[N*k];
	float64_t* dists = new float64_t[N];
	int32_t* neighs = new int32_t[N];

	// find neighbors
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			neighs[j] = j;
			dists[j] = distances[i*N+j];
		}

		CMath::qsort_index(dists,neighs,N);

		for (j=0; j<k; j++)
			neighborhood[i*k+j] = neighs[j+1];
	}

	// print
	for (i=0; i<N; i++)
	{
		SG_PRINT("%dth \n",i);
		for (j=0; j<k; j++)
			SG_PRINT("%d ", neighborhood[i*k+j]);
		SG_PRINT("\n");
	}

	float64_t* z = new float64_t[N*k];
	float64_t* z_transposed = new float64_t[N*k];
	float64_t* covariance_matrix = new float64_t[N*N];
	float64_t* feature_vector = new float64_t[dim];
	for (i=0; i<N; i++)
	{
		for (j=0; j<k; j++)
		{
			pdata->get_feature_vector(&feature_vector, &dim, neighborhood[i*k+j]);
			for (int d=0; d<dim; d++)
			{
				z[d*k+j] = feature_vector[d];
			}
		}

		pdata->get_feature_vector(&feature_vector, &dim, i);

		for (j=0; j<k; j++)
			for (int d=0; d<dim; d++)
			{
				z[d*k+j] -= feature_vector[d];
			}

		SG_PRINT("%dth Z matrix\n", i);
		for (j=0; j<dim; j++)
		{
			for (int d=0; d<k; d++)
			{
				SG_PRINT("[%d] %5.3f ", j*k+d, z[j*k+d]);
			}
			SG_PRINT("\n");
		}

		cblas_dgemm(CblasColMajor,CblasNoTrans, CblasTrans,k,k,k,1.0,z,k,z,k,0.0,covariance_matrix,k);

		SG_PRINT("%dth covariane matrix\n", i);
		for (j=0; j<k; j++)
		{
			for (int d=0; d<k; d++)
			{
				SG_PRINT("[%d] %5.3f ", j*k+d, covariance_matrix[j*k+d]);
			}
			SG_PRINT("\n");
		}
	}

	// TODO calc weights, eigenproblem

	delete[] dists;
	delete[] neighs;
	delete[] distances;
	delete[] neighborhood;
	delete dist;

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
