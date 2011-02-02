/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Berlin Institute of Technology
 */

#include "lib/config.h"
#include "lib/Mathematics.h"

#include <string.h>
#include <stdlib.h>

#ifdef HAVE_LAPACK
#include "lib/lapack.h"

#include "lib/common.h"
#include "preproc/KernelPCACut.h"
#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

using namespace shogun;

CKernelPCACut::CKernelPCACut()
: CSimplePreProc<float64_t>("KernelPCACut", "KPCA"), T(NULL), num_dim(0),
	initialized(false), thresh(1e-6), kernel(NULL)
{
}

CKernelPCACut::CKernelPCACut(CKernel* k, float64_t thresh_)
: CSimplePreProc<float64_t>("KernelPCACut", "KPCA"), T(NULL), num_dim(0),
	initialized(false), thresh(thresh_)
{
	SG_REF(k);
	kernel=k;
}

CKernelPCACut::~CKernelPCACut()
{
	delete[] T;
	SG_UNREF(kernel);
}

/// initialize preprocessor from features
bool CKernelPCACut::init(CFeatures* f)
{
	if (!initialized && kernel)
	{
		ASSERT(f->get_feature_class()==C_SIMPLE);
		ASSERT(f->get_feature_type()==F_DREAL);

		int32_t n=0;
		int32_t m=0;
		kernel->init(f,f);
		float64_t* km = kernel->get_kernel_matrix(m, n, (float64_t*) NULL);
		ASSERT(n==m);

		//int64_t num_data= int64_t(n)*m;
		//% Centering kernel matrix (non-linearly mapped data).
		//J = ones(num_data,num_data)/num_data;
		//Kc = K - J*K - K*J + J*K*J;
		CMath::center_matrix(km, n, m);
		CMath::display_matrix(km,n,m, "kmc");

		//% eigen decomposition of the kernel marix
		//[U,D] = eig(Kc);
		//Lambda=real(diag(D));
		//% normalization of eigenvectors to be orthonormal 
		//	for k = 1:num_data,
		//		if Lambda(k) ~= 0,
		//			U(:,k)=U(:,k)/sqrt(Lambda(k));
		//end
		//	end
		eigenvalues=CMath::compute_eigenvectors(km, n, n);

		//% Sort the eigenvalues and the eigenvectors in descending order.
		//[Lambda,ordered]=sort(-Lambda);
		//Lambda=-Lambda;
		//U=U(:,ordered);
		//int32_t* index=new int32_t[n];
		//CMath::range_fill_vector(index, n);
		//CMath::qsort_backward_index(eigenvalues, index, n);
		T=km;


		//% use first new_dim principal components
		//A=U(:,1:options.new_dim);

		//% compute Alpha and compute bias (implicite centering)
		//	% of kernel projection
		//	model.Alpha = (eye(num_data,num_data)-J)*A;
		//Jt=ones(num_data,1)/num_data;
		//model.b = A'*(J'*K*Jt-K*Jt);
		initialized=true;
		SG_INFO("Done\n") ;
		return true ;
	}
	return 
		false;
}

/// initialize preprocessor from features
void CKernelPCACut::cleanup()
{
	delete[] T ;
	T=NULL ;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
float64_t* CKernelPCACut::apply_to_feature_matrix(CFeatures* f)
{
	/*
	int32_t num_vectors=0;
	int32_t num_features=0;

	float64_t* m=((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);
	SG_INFO("get Feature matrix: %ix%i\n", num_features, num_vectors) ;

	if (m)
	{
		SG_INFO("Preprocessing feature matrix\n");
		float64_t* res= new float64_t[num_dim];
		float64_t* sub_mean= new float64_t[num_features];

		for (int32_t vec=0; vec<num_vectors; vec++)
		{
			int32_t i;

			for (i=0; i<num_features; i++)
				sub_mean[i]=m[num_features*vec+i]-mean[i] ;

			int nd = (int) num_dim; // calling external lib
			cblas_dgemv(CblasColMajor, CblasNoTrans, nd, (int) num_features,
				1.0, T, nd, (double*) sub_mean, 1, 0, (double*) res, 1);

			float64_t* m_transformed=&m[num_dim*vec];
			for (i=0; i<num_dim; i++)
				m_transformed[i]=res[i];
		}
		delete[] res;
		delete[] sub_mean;

		((CSimpleFeatures<float64_t>*) f)->set_num_features(num_dim);
		((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_features, num_vectors);
		SG_INFO("new Feature matrix: %ix%i\n", num_vectors, num_features);
	}

	return m;
*/
	return NULL;
}

/// apply preproc on single feature vector
/// result in feature matrix
float64_t* CKernelPCACut::apply_to_feature_vector(float64_t* f, int32_t &len)
{
	/*
	float64_t *ret=new float64_t[num_dim];
	float64_t *sub_mean=new float64_t[len];
	for (int32_t i=0; i<len; i++)
		sub_mean[i]=f[i]-mean[i];

	int nd = (int) num_dim;  // calling external lib
	cblas_dgemv(CblasColMajor, CblasNoTrans, nd, (int) len, 1.0, (double*) T,
		nd, (double*) sub_mean, 1, 0, (double*) ret, 1);

	delete[] sub_mean ;
	len=num_dim ;
	//	  SG_DEBUG( "num_dim: %d\n", num_dim);
	return ret;
	*/
	return NULL;
}

void CKernelPCACut::get_transformation_matrix(float64_t** dst, int32_t* num_feat, int32_t* num_new_dim)
{
	ASSERT(T);

	int64_t num=int64_t(num_dim)*num_old_dim;
	*num_feat=num_old_dim;
	*num_new_dim=num_dim;
	*dst=(float64_t*) malloc(sizeof(float64_t)*num);
	if (!*dst)
		SG_ERROR("Allocating %ld bytes failes\n", sizeof(float64_t)*num);
	memcpy(*dst, T, num * sizeof(float64_t));
}

void CKernelPCACut::get_eigenvalues(float64_t** dst, int32_t* new_num_dim)
{
	ASSERT(eigenvalues);

	*new_num_dim=num_eigenvalues;
	*dst=(float64_t*) malloc(sizeof(float64_t)*num_eigenvalues);
	if (!*dst)
		SG_ERROR("Allocating %ld bytes failes\n", sizeof(float64_t)*num_eigenvalues);
	memcpy(*dst, eigenvalues, num_eigenvalues * sizeof(float64_t));
}
#endif
