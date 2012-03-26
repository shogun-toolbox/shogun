/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/converter/libedrt.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/lib/CoverTree.h>
#include <shogun/base/DynArray.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

CLocallyLinearEmbedding::CLocallyLinearEmbedding() :
		CEmbeddingConverter()
{
	m_k = 10;
	m_max_k = 60;
	m_auto_k = false;
	m_nullspace_shift = -1e-9;
	m_reconstruction_shift = 1e-3;
#ifdef HAVE_ARPACK
	m_use_arpack = true;
#else
	m_use_arpack = false;
#endif
	init();
}

void CLocallyLinearEmbedding::init()
{
	m_parameters->add(&m_auto_k, "auto_k",
	                  "whether k should be determined automatically in range");
	m_parameters->add(&m_k, "k", "number of neighbors");
	m_parameters->add(&m_max_k, "max_k",
	                  "maximum number of neighbors used to compute optimal one");
	m_parameters->add(&m_nullspace_shift, "nullspace_shift",
	                  "nullspace finding regularization shift");
	m_parameters->add(&m_reconstruction_shift, "reconstruction_shift",
	                  "shift used to regularize reconstruction step");
	m_parameters->add(&m_use_arpack, "use_arpack",
	                  "whether arpack is being used or not");
}


CLocallyLinearEmbedding::~CLocallyLinearEmbedding()
{
}

void CLocallyLinearEmbedding::set_k(int32_t k)
{
	ASSERT(k>0);
	m_k = k;
}

int32_t CLocallyLinearEmbedding::get_k() const
{
	return m_k;
}

void CLocallyLinearEmbedding::set_max_k(int32_t max_k)
{
	ASSERT(max_k>=m_k);
	m_max_k = max_k;
}

int32_t CLocallyLinearEmbedding::get_max_k() const
{
	return m_max_k;
}

void CLocallyLinearEmbedding::set_auto_k(bool auto_k)
{
	m_auto_k = auto_k;
}

bool CLocallyLinearEmbedding::get_auto_k() const
{
	return m_auto_k;
}

void CLocallyLinearEmbedding::set_nullspace_shift(float64_t nullspace_shift)
{
	m_nullspace_shift = nullspace_shift;
}

float64_t CLocallyLinearEmbedding::get_nullspace_shift() const
{
	return m_nullspace_shift;
}

void CLocallyLinearEmbedding::set_reconstruction_shift(float64_t reconstruction_shift)
{
	m_reconstruction_shift = reconstruction_shift;
}

float64_t CLocallyLinearEmbedding::get_reconstruction_shift() const
{
	return m_reconstruction_shift;
}

void CLocallyLinearEmbedding::set_use_arpack(bool use_arpack)
{
	m_use_arpack = use_arpack;
}

bool CLocallyLinearEmbedding::get_use_arpack() const
{
	return m_use_arpack;
}

const char* CLocallyLinearEmbedding::get_name() const
{
	return "LocallyLinearEmbedding";
}

CFeatures* CLocallyLinearEmbedding::apply(CFeatures* features)
{
	ASSERT(features);
	// check features
	if (!(features->get_feature_class()==C_SIMPLE &&
	      features->get_feature_type()==F_DREAL))
	{
		SG_ERROR("Given features are not of SimpleRealFeatures type.\n");
	}
	// shorthand for simplefeatures
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;
	SG_REF(features);

	// get and check number of vectors
	int32_t N = simple_features->get_num_vectors();
	if (m_k>=N)
		SG_ERROR("Number of neighbors (%d) should be less than number of objects (%d).\n",
		         m_k, N);

	CKernel* kernel = new CLinearKernel(simple_features,simple_features);
	CKernel* custom_kernel = new CCustomKernel(kernel);

	float64_t* new_features = NULL;

	edrt_options_t options;
	options.method = KERNEL_LOCALLY_LINEAR_EMBEDDING;
	options.use_arpack = m_use_arpack;

	edrt_embedding(options,
	               m_target_dim,N,2,m_k,NULL,
	               &compute_kernel,NULL,NULL,(void*)custom_kernel,
	               &new_features);

	SG_UNREF(custom_kernel);
	SG_UNREF(kernel);

	SG_UNREF(features);
	return (CFeatures*)(new CSimpleFeatures<float64_t>(SGMatrix<float64_t>(new_features,m_target_dim,N)));
}

float64_t CLocallyLinearEmbedding::compute_kernel(int32_t i, int32_t j, void* user_data)
{
	return ((CKernel*)user_data)->kernel(i,j);
}

int32_t CLocallyLinearEmbedding::estimate_k(CSimpleFeatures<float64_t>* simple_features, SGMatrix<int32_t> neighborhood_matrix)
{
	int32_t right = m_max_k;
	int32_t left = m_k;
	int32_t left_third;
	int32_t right_third;
	ASSERT(right>=left);
	if (right==left) return left;
	int32_t dim;
	int32_t N;
	float64_t* feature_matrix= simple_features->get_feature_matrix(dim,N);
	float64_t* z_matrix = SG_MALLOC(float64_t,right*dim);
	float64_t* covariance_matrix = SG_MALLOC(float64_t,right*right);
	float64_t* resid_vector = SG_MALLOC(float64_t, right);
	float64_t* id_vector = SG_MALLOC(float64_t, right);
	while (right-left>2)
	{
		left_third = (left*2+right)/3;
		right_third = (right*2+left)/3;
		float64_t left_val = compute_reconstruction_error(left_third,dim,N,feature_matrix,z_matrix,
		                                                  covariance_matrix,resid_vector,
		                                                  id_vector,neighborhood_matrix);
		float64_t right_val = compute_reconstruction_error(right_third,dim,N,feature_matrix,z_matrix,
		                                                   covariance_matrix,resid_vector,
		                                                   id_vector,neighborhood_matrix);
		if (left_val<right_val)
			right = right_third;
		else
			left = left_third;
	}
	SG_FREE(z_matrix);
	SG_FREE(covariance_matrix);
	SG_FREE(resid_vector);
	SG_FREE(id_vector);
	return right;
}

float64_t CLocallyLinearEmbedding::compute_reconstruction_error(int32_t k, int dim, int N, float64_t* feature_matrix,
                                                                float64_t* z_matrix, float64_t* covariance_matrix,
                                                                float64_t* resid_vector, float64_t* id_vector,
                                                                SGMatrix<int32_t> neighborhood_matrix)
{
	// todo parse params
	int32_t i,j;
	float64_t total_residual_norm = 0.0;
	for (i=CMath::random(0,20); i<N; i+=N/20)
	{
		for (j=0; j<k; j++)
		{
			cblas_dcopy(dim,feature_matrix+neighborhood_matrix[i*m_k+j]*dim,1,z_matrix+j*dim,1);
			cblas_daxpy(dim,-1.0,feature_matrix+i*dim,1,z_matrix+j*dim,1);
		}
		cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,
			    k,k,dim,
			    1.0,z_matrix,dim,
			    z_matrix,dim,
			    0.0,covariance_matrix,k);
		for (j=0; j<k; j++)
		{
			resid_vector[j] = 1.0;
			id_vector[j] = 1.0;
		}
		if (k>dim)
		{
			float64_t trace = 0.0;
			for (j=0; j<k; j++)
				trace += covariance_matrix[j*k+j];
			for (j=0; j<m_k; j++)
				covariance_matrix[j*k+j] += m_reconstruction_shift*trace;
		}
		clapack_dposv(CblasColMajor,CblasLower,k,1,covariance_matrix,k,id_vector,k);
		float64_t norming=0.0;
		for (j=0; j<k; j++)
			norming += id_vector[j];
		cblas_dscal(k,-1.0/norming,id_vector,1);
		cblas_dsymv(CblasColMajor,CblasLower,k,-1.0,covariance_matrix,k,id_vector,1,1.0,resid_vector,1);
		total_residual_norm += cblas_dnrm2(k,resid_vector,1);
	}
	return total_residual_norm/k;
}
#endif /* HAVE_LAPACK */
