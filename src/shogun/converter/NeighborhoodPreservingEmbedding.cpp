/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#include <shogun/converter/NeighborhoodPreservingEmbedding.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CNeighborhoodPreservingEmbedding::CNeighborhoodPreservingEmbedding() :
		CLocallyLinearEmbedding()
{
}

CNeighborhoodPreservingEmbedding::~CNeighborhoodPreservingEmbedding()
{
}

const char* CNeighborhoodPreservingEmbedding::get_name() const
{
	return "NeighborhoodPreservingEmbedding";
}

SGMatrix<float64_t> CNeighborhoodPreservingEmbedding::construct_embedding(CFeatures* features, SGMatrix<float64_t> matrix, int dimension)
{
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*)features;
	ASSERT(simple_features);
	int i,j;
	int N = simple_features->get_num_vectors();
	int dim = simple_features->get_num_features();
	ASSERT(dimension<=dim);
	
	SGMatrix<float64_t> feature_matrix = simple_features->get_feature_matrix();
	float64_t* XTM = SG_MALLOC(float64_t, dim*N);
	float64_t* lhs_M = SG_MALLOC(float64_t, dim*dim);
	float64_t* rhs_M = SG_MALLOC(float64_t, dim*dim);

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dim,N,N,1.0,feature_matrix.matrix,dim,matrix.matrix,N,0.0,XTM,dim);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,dim,dim,N,1.0,XTM,dim,feature_matrix.matrix,dim,0.0,lhs_M,dim);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,dim,dim,N,1.0,feature_matrix.matrix,dim,feature_matrix.matrix,dim,0.0,rhs_M,dim);

	float64_t* evals = SG_MALLOC(float64_t, dim);
	float64_t* evectors = SG_MALLOC(float64_t, dimension*dim);
	int32_t info = 0;
#ifdef HAVE_ARPACK
	arpack_dsxupd(lhs_M,rhs_M,false,dim,dimension,"LA",false,3,true,false,m_nullspace_shift,0.0,
	              evals,evectors,info);
#else
	wrap_dsygvx(1,'V','U',dim,lhs_M,dim,rhs_M,dim,dim-dimension+1,dim,evals,evectors,&info);
#endif
	SG_FREE(lhs_M);
	SG_FREE(rhs_M);
	SG_FREE(evals);
	if (info!=0) SG_ERROR("Failed to solve eigenproblem (%d)\n",info);

	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,N,dimension,dim,1.0,feature_matrix.matrix,dim,evectors,dim,0.0,XTM,N);
	SG_FREE(evectors);

	float64_t* new_features = SG_MALLOC(float64_t, dimension*N);
	for (i=0; i<dimension; i++)
	{
		for (j=0; j<N; j++)
			new_features[j*dimension+i] = XTM[i*N+j];
	}
	SG_FREE(XTM);
	return SGMatrix<float64_t>(new_features,dimension,N);
}

#endif /* HAVE_LAPACK */
