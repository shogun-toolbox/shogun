/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "preprocessor/LandmarkIsomap.h"
#ifdef HAVE_LAPACK
#include "preprocessor/DimensionReductionPreprocessor.h"
#include "lib/lapack.h"
#include "preprocessor/LandmarkMDS.h"
#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "distance/EuclidianDistance.h"
#include "distance/CustomDistance.h"
#include "lib/Signal.h"

using namespace shogun;

CLandmarkIsomap::CLandmarkIsomap() : CLandmarkMDS(), m_k(3)
{
}

CLandmarkIsomap::~CLandmarkIsomap()
{
}

bool CLandmarkIsomap::init(CFeatures* features)
{
	return true;
}

void CLandmarkIsomap::cleanup()
{
}

CCustomDistance* CLandmarkIsomap::approx_geodesic_distance(CDistance* distance)
{
	int32_t N,k,i,j;
	float64_t* D_matrix;
	distance->get_distance_matrix(&D_matrix,&N,&N);
	ASSERT(m_k<=N);

	float64_t* row = new float64_t[N];
	int32_t* row_idx = new int32_t[N];

	// cut-off by k
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			row[j] = D_matrix[i*N+j];
			row_idx[j] = j;
		}

		CMath::qsort_index(row,row_idx,N);

		for (j=m_k+1; j<N; j++)
		{
			if (i!=row_idx[j])
				D_matrix[i*N+row_idx[j]] = CMath::ALMOST_INFTY;
		}
	}

	delete[] row;
	delete[] row_idx;

	// floyd-warshall
	for (k=0; k<N; k++)
	{
		for (i=0; i<N; i++)
		{
			for (j=0; j<N; j++)
				D_matrix[i*N+j] = CMath::min(D_matrix[i*N+j], D_matrix[i*N+k] + D_matrix[k*N+j]);
		}
	}

	CCustomDistance* geodesic_distance = new CCustomDistance(D_matrix,N,N);
	delete[] D_matrix;

	return geodesic_distance;
}

CSimpleFeatures<float64_t>* CLandmarkIsomap::apply_to_distance(CDistance* distance)
{
	ASSERT(distance);

	CCustomDistance* geodesic_distance = approx_geodesic_distance(distance);

	SGMatrix<float64_t> new_feature_matrix =
			CLandmarkMDS::embed_by_distance(geodesic_distance);

	delete geodesic_distance;

	CSimpleFeatures<float64_t>* new_features =
			new CSimpleFeatures<float64_t>(new_feature_matrix);

	return new_features;
}


SGMatrix<float64_t> CLandmarkIsomap::apply_to_feature_matrix(CFeatures* features)
{
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;

	CDistance* distance = new CEuclidianDistance(simple_features,simple_features);

	CCustomDistance* geodesic_distance = approx_geodesic_distance(distance);

	SGMatrix<float64_t> new_feature_matrix =
			CLandmarkMDS::embed_by_distance(geodesic_distance);

	delete geodesic_distance;

	simple_features->set_feature_matrix(new_feature_matrix);

	return new_feature_matrix;
}

SGVector<float64_t> CLandmarkIsomap::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

#endif /* HAVE_LAPACK */
