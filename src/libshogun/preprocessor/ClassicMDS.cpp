/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "preprocessor/ClassicMDS.h"
#ifdef HAVE_LAPACK
#include "preprocessor/DimensionReductionPreprocessor.h"
#include "lib/lapack.h"
#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "distance/EuclidianDistance.h"
#include "lib/Signal.h"

using namespace shogun;

CClassicMDS::CClassicMDS() : CDimensionReductionPreprocessor()
{
}

CClassicMDS::~CClassicMDS()
{
}

bool CClassicMDS::init(CFeatures* data)
{
	return true;
}

void CClassicMDS::cleanup()
{
}

CSimpleFeatures<float64_t>* CClassicMDS::apply_to_distance(CDistance* distance)
{
	ASSERT(distance);

	SGMatrix<float64_t> new_feature_matrix = embed_by_distance(distance);

	CSimpleFeatures<float64_t>* new_features =
			new CSimpleFeatures<float64_t>(new_feature_matrix);

	return new_features;
}

SGMatrix<float64_t> CClassicMDS::embed_by_distance(CDistance* distance)
{
	ASSERT(distance->get_num_vec_lhs()==distance->get_num_vec_rhs());
	int32_t N = distance->get_num_vec_lhs();

	// loop variables
	int32_t i,j;

	// get distance matrix
	float64_t* D_matrix;
	distance->get_distance_matrix(&D_matrix,&N,&N);

	// get D^2 matrix
	float64_t* Ds_matrix = new float64_t[N*N];
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
			Ds_matrix[i*N+j] = D_matrix[i*N+j]*D_matrix[i*N+j];

	// centering matrix
	float64_t* H_matrix = new float64_t[N*N];
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
			H_matrix[i*N+j] = (i==j) ? 1.0-1.0/N : -1.0/N;

	// compute -1/2 H D^2 H (result in Ds_matrix)
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
			N,N,N,1.0,H_matrix,N,Ds_matrix,N,0.0,D_matrix,N);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,
			N,N,N,-0.5,D_matrix,N,H_matrix,N,0.0,Ds_matrix,N);

	// cleanup
	delete[] D_matrix;
	delete[] H_matrix;

	// eigendecomposition
	float64_t* eigenvalues_vector = new float64_t[N];
	int32_t eigenproblem_status = 0;
	wrap_dsyev('V','U',N,Ds_matrix,N,eigenvalues_vector,&eigenproblem_status);
	ASSERT(eigenproblem_status==0);

	// replace feature matrix with (top) eigenvectors associated with largest
	// positive eigenvalues (ignores negative eigenvalues)
	float64_t* replace_feature_matrix = new float64_t[N*m_target_dim];
	for (i=0; i<m_target_dim; i++)
	{
		if (eigenvalues_vector[N-i-1]<0)
		{
			m_target_dim = i;
			break;
		}

		for (j=0; j<N; j++)
			replace_feature_matrix[j*m_target_dim+i] = Ds_matrix[(N-i-1)*N+j]*CMath::sqrt(eigenvalues_vector[N-i-1]);
	}

	// cleanup
	delete[] eigenvalues_vector;
	delete[] Ds_matrix;

	return SGMatrix<float64_t>(replace_feature_matrix,m_target_dim,N);
}

SGMatrix<float64_t> CClassicMDS::apply_to_feature_matrix(CFeatures* features)
{
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;
	CDistance* distance = new CEuclidianDistance(simple_features,simple_features);

	SGMatrix<float64_t> new_feature_matrix = embed_by_distance(distance);
	simple_features->set_feature_matrix(new_feature_matrix);

	return new_feature_matrix;
}

SGVector<float64_t> CClassicMDS::apply_to_feature_vector(SGVector<float64_t> vector)
{
	SG_NOTIMPLEMENTED;
	return vector;
}
#endif /* HAVE_LAPACK */
