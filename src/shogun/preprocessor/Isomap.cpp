/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/preprocessor/Isomap.h>
#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CCustomDistance* CIsomap::isomap_distance(CDistance* distance)
{
	int32_t N,k,i,j;
	SGMatrix<float64_t> D_matrix=distance->get_distance_matrix();
	N=D_matrix.num_cols;

	if (m_type==EISOMAP)
	{
		// just replace distances >e with infty
		for (i=0; i<N*N; i++)
		{
			if (D_matrix.matrix[i]>m_epsilon)
				D_matrix.matrix[i] = CMath::ALMOST_INFTY;
		}
	}
	if (m_type==KISOMAP)
	{
		// cut by k-nearest neighbors

		float64_t* col = SG_MALLOC(float64_t, N);
		int32_t* col_idx = SG_MALLOC(int32_t, N);
			
		// -> INFTY edges connecting NOT neighbors
		for (i=0; i<N; i++)
		{
			for (j=0; j<N; j++)
			{
				col[j] = D_matrix.matrix[j*N+i];
				col_idx[j] = j;
			}

			CMath::qsort_index(col,col_idx,N);

			for (j=m_k+1; j<N; j++)
			{
				D_matrix.matrix[col_idx[j]*N+i] = CMath::ALMOST_INFTY;
			}
		}

		SG_FREE(col);
		SG_FREE(col_idx);
	}

	// Floyd-Warshall on distance matrix
	// TODO replace by dijkstra
	for (k=0; k<N; k++)
	{
		for (i=0; i<N; i++)
		{
			for (j=0; j<N; j++)
			{
				D_matrix.matrix[i*N+j] =
						CMath::min(D_matrix.matrix[i*N+j],
								   D_matrix.matrix[i*N+k] + D_matrix.matrix[k*N+j]);
			}
		}
	}

	CCustomDistance* geodesic_distance = new CCustomDistance(D_matrix.matrix,N,N);

	// should be removed if custom distance doesn't copy the matrix
	SG_FREE(D_matrix.matrix);

	return geodesic_distance;
}

#endif /* HAVE_LAPACK */
