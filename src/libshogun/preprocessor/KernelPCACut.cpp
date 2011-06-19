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
#include "preprocessor/KernelPCACut.h"
#include "preprocessor/SimplePreprocessor.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

using namespace shogun;

CKernelPCACut::CKernelPCACut()
: CSimplePreprocessor<float64_t>(), T(NULL), rows_T(0),
	initialized(false), thresh(1e-6), kernel(NULL)
{
}

CKernelPCACut::CKernelPCACut(CKernel* k, float64_t thresh_)
: CSimplePreprocessor<float64_t>(), T(NULL), rows_T(0),
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
bool CKernelPCACut::init(CFeatures* features)
{
	if (!initialized && kernel)
	{
		ASSERT(features->get_feature_class()==C_SIMPLE);
		ASSERT(features->get_feature_type()==F_DREAL);

		int32_t n=0;
		int32_t m=0;
		kernel->init(features,features);
		float64_t* km = kernel->get_kernel_matrix(m, n, (float64_t*) NULL);
		ASSERT(n==m);

		float64_t* bias_tmp=CMath::get_column_sum(km, n,n);
		CMath::scale_vector(-1.0/n, bias_tmp, n);
		float64_t s=CMath::sum(bias_tmp, n)/n;
		CMath::add_scalar(-s, bias_tmp, n);

		CMath::center_matrix(km, n, m);

		eigenvalues=CMath::compute_eigenvectors(km, n, n);
		num_eigenvalues=n;


		for (int32_t i=0; i<n; i++)
		{	
			//normalize and trap divide by zero and negative eigenvalues
			for (int32_t j=0; j<n; j++)
				km[i*n+j]/=CMath::sqrt(CMath::max(1e-16,eigenvalues[i]));
		}

		//% Sort the eigenvalues and the eigenvectors in descending order.
		//[Lambda,ordered]=sort(-Lambda);
		//Lambda=-Lambda;
		//U=U(:,ordered);
		//int32_t* index=new int32_t[n];
		//CMath::range_fill_vector(index, n);
		//CMath::qsort_backward_index(eigenvalues, index, n);
		T=km;
		rows_T=n;
		cols_T=n;

		bias=new float64_t[cols_T];
		CMath::fill_vector(bias, cols_T, 0.0);
		bias_len=cols_T;

		CMath::dgemv(1.0, T, rows_T, cols_T, CblasTrans, bias_tmp, 0.0, bias);

		float64_t* rowsum=CMath::get_row_sum(T, rows_T, cols_T);
		CMath::scale_vector(1.0/n, rowsum, cols_T);

		for (int32_t i=0; i<cols_T; i++)
		{	
			for (int32_t j=0; j<rows_T; j++)
				T[j+rows_T*i]-=rowsum[i];
		}
		delete[] rowsum;
		delete[] bias_tmp;

		initialized=true;
		SG_INFO("Done\n");
		return true;
	}
	return false;
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
SGMatrix<float64_t> CKernelPCACut::apply_to_feature_matrix(CFeatures* features)
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
	return ((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CKernelPCACut::apply_to_feature_vector(SGVector<float64_t> vector)
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
	for( i = 0; i < num_data; i++ ) {

		for( k = 0; k < new_dim; k++) {
			Y[k+i*new_dim] = b[k];
		}

		for( j = 0; j < nsv; j++ ) {
			k_ij = kernel(i,j);

			for( k = 0; k < new_dim; k++) {
				if(Alpha[j+k*nsv] != 0 )
					Y[k+i*new_dim] += k_ij*Alpha[j+k*nsv];  
			}
		}
	}
	*/

	return vector;
}

void CKernelPCACut::get_transformation_matrix(float64_t** dst, int32_t* num_feat, int32_t* num_new_dim)
{
	ASSERT(T);

	int64_t num=int64_t(rows_T)*cols_T;
	*num_feat=cols_T;
	*num_new_dim=rows_T;
	*dst=(float64_t*) SG_MALLOC(sizeof(float64_t)*num);
	memcpy(*dst, T, num * sizeof(float64_t));
}

void CKernelPCACut::get_bias(float64_t** dst, int32_t* new_num_dim)
{
	ASSERT(bias);

	*new_num_dim=bias_len;
	*dst=(float64_t*) SG_MALLOC(sizeof(float64_t)*bias_len);
	memcpy(*dst, bias, bias_len * sizeof(float64_t));
}

void CKernelPCACut::get_eigenvalues(float64_t** dst, int32_t* new_num_dim)
{
	ASSERT(eigenvalues);

	*new_num_dim=num_eigenvalues;
	*dst=(float64_t*) SG_MALLOC(sizeof(float64_t)*num_eigenvalues);
	memcpy(*dst, eigenvalues, num_eigenvalues * sizeof(float64_t));
}
#endif
