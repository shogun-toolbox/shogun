/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "preprocessor/LandmarkMDS.h"
#ifdef HAVE_LAPACK
#include "lib/lapack.h"
#include "preprocessor/ClassicMDS.h"
#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "distance/EuclidianDistance.h"
#include "lib/Signal.h"

using namespace shogun;

CLandmarkMDS::CLandmarkMDS() : CClassicMDS(), m_landmark_number(3)
{
}

CLandmarkMDS::~CLandmarkMDS()
{
}

bool CLandmarkMDS::init(CFeatures* data)
{
	return true;
}

void CLandmarkMDS::cleanup()
{
}

SGVector<int32_t> CLandmarkMDS::get_landmark_idxs(int32_t count, int32_t total_count)
{
	int32_t* idxs = new int32_t[total_count];
	int32_t i,rnd;
	int32_t* permuted_idxs = new int32_t[count];

	// reservoir sampling
	for (i=0; i<total_count; i++)
		idxs[i] = i;

	for (i=0; i<count; i++)
		permuted_idxs[i] = idxs[i];

	for (i=count; i<total_count; i++)
	{
		 rnd = CMath::random(1,i);
		 if (rnd<count)
			 permuted_idxs[rnd] = idxs[i];
	}

	delete[] idxs;

	CMath::qsort(permuted_idxs,count);

	return SGVector<int32_t>(permuted_idxs, count);
}

SGMatrix<float64_t> CLandmarkMDS::embed_by_distance(CDistance* distance)
{
	int32_t i,j,k;
	int32_t lmk_N = m_landmark_number;
	int32_t total_N = distance->get_num_vec_lhs();
	SGMatrix<float64_t> dist_matrix = distance->get_distance_matrix();
	delete distance;

	SGVector<int32_t> lmk_idxs = get_landmark_idxs(lmk_N,total_N);

	// compute distance between landmarks
	float64_t* lmk_dist_matrix = new float64_t[lmk_N*lmk_N];
	for (i=0; i<lmk_N; i++)
	{
		for (j=0; j<lmk_N; j++)
			lmk_dist_matrix[i*lmk_N+j] =
				dist_matrix.matrix[lmk_idxs.vector[i]*total_N+lmk_idxs.vector[j]];
	}

	// distance between landmarks
	CCustomDistance* lmk_distance =
		new CCustomDistance(lmk_dist_matrix, lmk_N, lmk_N);
	
	// get landmarks embedding
	SGMatrix<float64_t> lmk_feature_matrix = CClassicMDS::embed_by_distance(lmk_distance);
	
	// construct new feature matrix
	float64_t* new_feature_matrix = new float64_t[m_target_dim*total_N];
	for (i=0; i<m_target_dim*total_N; i++)
		new_feature_matrix[i] = 0.0;	

	// fill new feature matrix with embedded landmarks
	for (i=0; i<lmk_N; i++)
	{
		for (j=0; j<m_target_dim; j++)
			new_feature_matrix[lmk_idxs.vector[i]*m_target_dim+j] =
				lmk_feature_matrix.matrix[i*m_target_dim+j];
	}

	// get exactly defined pseudoinverse of landmarks feature matrix
	ASSERT(m_eigenvalues.vector && m_eigenvalues.vlen == m_target_dim);
	for (i=0; i<lmk_N; i++)
	{
		for (j=0; j<m_target_dim; j++)
			lmk_feature_matrix.matrix[i*m_target_dim+j] /= m_eigenvalues.vector[j];
	}

	// compute mean vector of squared distances
	float64_t* mean_sq_dist_vector = new float64_t[lmk_N];
	for (i=0; i<lmk_N; i++)
	{
		mean_sq_dist_vector[i] = 0.0;
		for (j=0; j<lmk_N; j++)
			mean_sq_dist_vector[i] += lmk_dist_matrix[i*lmk_N+j] / lmk_N;
	}

	// get embedding for non-landmark vectors
	float64_t* current_dist_to_lmk = new float64_t[lmk_N];
	j = 0;
	for (i=0; i<total_N; i++)
	{
		// skip if lmk
		if (i==lmk_idxs.vector[j])
		{
			j++;
			continue;
		}
		// compute difference from mean landmark distance vector
		for (k=0; k<lmk_N; k++)
			current_dist_to_lmk[k] =
				CMath::sq(dist_matrix.matrix[i*total_N+lmk_idxs.vector[k]]) -
				CMath::sq(mean_sq_dist_vector[k]);
		
		// compute embedding
		cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
		            m_target_dim,1,lmk_N,
                    -0.5,lmk_feature_matrix.matrix,m_target_dim,
		            current_dist_to_lmk,lmk_N,
		            0.0,(new_feature_matrix+i*m_target_dim),m_target_dim);
	}

	// cleanup
	delete[] lmk_feature_matrix.matrix;
	delete[] current_dist_to_lmk;
	delete[] mean_sq_dist_vector;
	delete[] lmk_dist_matrix;
	delete lmk_distance;
	delete[] dist_matrix.matrix;

	return SGMatrix<float64_t>(new_feature_matrix,m_target_dim,total_N);
}

CSimpleFeatures<float64_t>* CLandmarkMDS::apply_to_distance(CDistance* distance)
{
	ASSERT(distance);

	SGMatrix<float64_t> new_feature_matrix = embed_by_distance(distance);

	CSimpleFeatures<float64_t>* new_features =
			new CSimpleFeatures<float64_t>(new_feature_matrix);

	return new_features;
}

SGMatrix<float64_t> CLandmarkMDS::apply_to_feature_matrix(CFeatures* features)
{
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;
	CDistance* distance = new CEuclidianDistance(simple_features,simple_features);

	SGMatrix<float64_t> new_feature_matrix = embed_by_distance(distance);
	simple_features->set_feature_matrix(new_feature_matrix);

	return new_feature_matrix;
}

SGVector<float64_t> CLandmarkMDS::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}

#endif /* HAVE_LAPACK */

