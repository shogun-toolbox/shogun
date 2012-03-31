/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/LocalityPreservingProjections.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CLocalityPreservingProjections::CLocalityPreservingProjections() :
		CLaplacianEigenmaps()
{
}

CLocalityPreservingProjections::~CLocalityPreservingProjections()
{
}

const char* CLocalityPreservingProjections::get_name() const
{
	return "LocalityPreservingProjections";
};

CSimpleFeatures<float64_t>* CLocalityPreservingProjections::construct_embedding(CFeatures* features,
                                                                                SGMatrix<float64_t> W_matrix)
{
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*)features;
	ASSERT(simple_features);
	int i,j;
	int N = simple_features->get_num_vectors();
	int dim = simple_features->get_num_features();

	float64_t* D_diag_vector = SG_CALLOC(float64_t, N);
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
			D_diag_vector[i] += W_matrix[i*N+j];
	}

	// W = -W
	for (i=0; i<N*N; i++)
		if (W_matrix[i]>0.0)
			W_matrix[i] *= -1.0;
	// W = W + D
	for (i=0; i<N; i++)
		W_matrix[i*N+i] += D_diag_vector[i];

	SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();
	float64_t* XTM = SG_MALLOC(float64_t, dim*N);
	float64_t* lhs_M = SG_MALLOC(float64_t, dim*dim);
	float64_t* rhs_M = SG_MALLOC(float64_t, dim*dim);

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dim,N,N,1.0,feature_matrix.matrix,dim,W_matrix.matrix,N,0.0,XTM,dim);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,dim,dim,N,1.0,XTM,dim,feature_matrix.matrix,dim,0.0,lhs_M,dim);

	for (i=0; i<N; i++)
		cblas_dscal(dim,D_diag_vector[i],feature_matrix.matrix+i*dim,1);

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,dim,dim,N,1.0,feature_matrix.matrix,dim,feature_matrix.matrix,dim,0.0,rhs_M,dim);

	for (i=0; i<N; i++)
		cblas_dscal(dim,1.0/D_diag_vector[i],feature_matrix.matrix+i*dim,1);

	float64_t* evals = SG_MALLOC(float64_t, dim);
	float64_t* evectors = SG_MALLOC(float64_t, m_target_dim*dim);
	int32_t info = 0;
#ifdef HAVE_ARPACK
	arpack_dsxupd(lhs_M,rhs_M,false,dim,m_target_dim,"LA",false,3,true,false,-1e-9,0.0,
	              evals,evectors,info);
#else
	wrap_dsygvx(1,'V','U',dim,lhs_M,dim,rhs_M,dim,dim-m_target_dim+1,dim,evals,evectors,&info);
#endif
	SG_FREE(lhs_M);
	SG_FREE(rhs_M);
	SG_FREE(evals);
	if (info!=0) SG_ERROR("Failed to solve eigenproblem (%d)\n",info);

	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,N,m_target_dim,dim,1.0,feature_matrix.matrix,dim,evectors,dim,0.0,XTM,N);
	SG_FREE(evectors);

	SGMatrix<float64_t> new_features(m_target_dim,N);
	for (i=0; i<m_target_dim; i++)
	{
		for (j=0; j<N; j++)
			new_features[j*m_target_dim+i] = XTM[i*N+j];
	}
	SG_FREE(D_diag_vector);
	SG_FREE(XTM);
	return new CSimpleFeatures<float64_t>(new_features);
}

#endif /* HAVE_LAPACK */
