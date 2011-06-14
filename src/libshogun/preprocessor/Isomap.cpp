/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "preprocessor/Isomap.h"
#ifdef HAVE_LAPACK
#include "lib/lapack.h"
#include "preprocessor/ClassicMDS.h"
#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "distance/EuclidianDistance.h"
#include "distance/CustomDistance.h"
#include "lib/Signal.h"

using namespace shogun;

CIsomap::CIsomap() : CClassicMDS()
{
}

CIsomap::~CIsomap()
{
}

bool CIsomap::init(CFeatures* data)
{
	return true;
}

void CIsomap::cleanup()
{
}

float64_t* CIsomap::apply_to_feature_matrix(CFeatures* data)
{
	CSimpleFeatures<float64_t>* pdata = (CSimpleFeatures<float64_t>*) data;
	CDistance* distance = new CEuclidianDistance(pdata,pdata);

	// loop variables
	int32_t i,j,k;

	// get distance matrix
	int32_t N;
	float64_t* D_matrix;
	distance->get_distance_matrix(&D_matrix,&N,&N);
	delete distance;

	// floyd-warshall
	for (k=0; k<N; k++)
	{
		for (i=0; i<N; i++)
		{
			for (j=0; j<N; j++)
				D_matrix[i*N+j] = CMath::min(D_matrix[i*N+j], D_matrix[i*N+k] + D_matrix[k*N+j]);
		}
	}

	SGMatrix<float64_t> features;
	apply_to_distance(new CCustomDistance(D_matrix,N,N),features);

	pdata->set_feature_matrix(features);
	return features.matrix;
}

float64_t* CIsomap::apply_to_feature_vector(float64_t* f, int32_t &len)
{
	SG_NOTIMPLEMENTED;
}
#endif /* HAVE_LAPACK */
