/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#include <shogun/converter/LinearLocalTangentSpaceAlignment.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CLinearLocalTangentSpaceAlignment::CLinearLocalTangentSpaceAlignment() :
		CLocalTangentSpaceAlignment()
{
}

CLinearLocalTangentSpaceAlignment::~CLinearLocalTangentSpaceAlignment()
{
}

const char* CLinearLocalTangentSpaceAlignment::get_name() const
{
	return "LinearLocalTangentSpaceAlignment";
}

SGMatrix<float64_t> CLinearLocalTangentSpaceAlignment::construct_embedding(CFeatures* features, SGMatrix<float64_t> matrix, int dimension)
{
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*)features;
	ASSERT(simple_features);
	int i,j;

	int N;
	int dim;
	float64_t* feature_matrix;
	simple_features->get_feature_matrix(&feature_matrix,&dim,&N);
	ASSERT(dimension<=dim);
	float64_t* XTM = SG_MALLOC(float64_t, dim*N);
	float64_t* lhs_M = SG_MALLOC(float64_t, dim*dim);
	float64_t* rhs_M = SG_MALLOC(float64_t, dim*dim);
	CMath::center_matrix(matrix.matrix,N,N);

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,dim,N,N,1.0,feature_matrix,dim,matrix.matrix,N,0.0,XTM,dim);
	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,dim,dim,N,1.0,XTM,dim,feature_matrix,dim,0.0,lhs_M,dim);

	float64_t* mean_vector = SG_CALLOC(float64_t, dim);
	for (i=0; i<N; i++)
		cblas_daxpy(dim,1.0,feature_matrix+i*dim,1,mean_vector,1);

	cblas_dscal(dim,1.0/N,mean_vector,1);

	for (i=0; i<N; i++)
		cblas_daxpy(dim,-1.0,mean_vector,1,feature_matrix+i*dim,1);

	SG_FREE(mean_vector);

	cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,dim,dim,N,1.0,feature_matrix,dim,feature_matrix,dim,0.0,rhs_M,dim);

	for (i=0; i<dim; i++) rhs_M[i*dim+i] += 1e-6;

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

	for (i=0; i<dimension/2; i++)
	{
		cblas_dswap(dim,evectors+i*dim,1,evectors+(dimension-i-1)*dim,1);
	}

	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,N,dimension,dim,1.0,feature_matrix,dim,evectors,dim,0.0,XTM,N);
	SG_FREE(evectors);
	SG_FREE(feature_matrix);

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
