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
#include <stdio.h>
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
	SG_PRINT("N = %d\n", N);
	int32_t i,j;

	// oh it is so dirty
	CDistance* dist = new CEuclidianDistance();
	dist->init(pdata,pdata);

	// get distances
	float64_t* distances = new float64_t[N*N];
	for (i=0; i<N; i++)
	{
		for (j=0; j<=i; j++)
		{
			distances[i*N+j] = dist->distance(i,j);
		}
		for (j=i; j<N; j++)
		{
			distances[i*N+j] = distances[j*N+i];
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
			neighborhood[i*k+j] = neighs[j];
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
