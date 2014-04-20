/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Berlin Institute of Technology
 */

#include <shogun/preprocessor/KernelPCA.h>
#ifdef HAVE_LAPACK
#include <shogun/lib/config.h>
#include <shogun/mathematics/Math.h>

#include <string.h>
#include <stdlib.h>

#include <shogun/mathematics/lapack.h>
#include <shogun/lib/common.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/preprocessor/DimensionReductionPreprocessor.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CKernelPCA::CKernelPCA() : CDimensionReductionPreprocessor()
{
	init();
}

CKernelPCA::CKernelPCA(CKernel* k) : CDimensionReductionPreprocessor()
{
	init();
	set_kernel(k);
}

void CKernelPCA::init()
{
	m_initialized = false;
	m_init_features = NULL;
	m_transformation_matrix = SGMatrix<float64_t>();
	m_bias_vector = SGVector<float64_t>();

	SG_ADD(&m_transformation_matrix, "transformation_matrix",
		"matrix used to transform data", MS_NOT_AVAILABLE);
	SG_ADD(&m_bias_vector, "bias_vector",
		"bias vector used to transform data", MS_NOT_AVAILABLE);
}

void CKernelPCA::cleanup()
{
	m_transformation_matrix = SGMatrix<float64_t>();
	m_bias_vector = SGVector<float64_t>();

	if (m_init_features)
		SG_UNREF(m_init_features);

	m_initialized = false;
}

CKernelPCA::~CKernelPCA()
{
	if (m_init_features)
		SG_UNREF(m_init_features);
}

bool CKernelPCA::init(CFeatures* features)
{
	if (!m_initialized && m_kernel)
	{
		SG_REF(features);
		m_init_features = features;

		m_kernel->init(features,features);
		SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();
		m_kernel->cleanup();
		int32_t n = kernel_matrix.num_cols;
		int32_t m = kernel_matrix.num_rows;
		ASSERT(n==m)

		float64_t* bias_tmp = SGMatrix<float64_t>::get_column_sum(kernel_matrix.matrix, n,n);
		SGVector<float64_t>::scale_vector(-1.0/n, bias_tmp, n);
		float64_t s = SGVector<float64_t>::sum(bias_tmp, n)/n;
		SGVector<float64_t>::add_scalar(-s, bias_tmp, n);

		SGMatrix<float64_t>::center_matrix(kernel_matrix.matrix, n, m);

		float64_t* eigenvalues=SGMatrix<float64_t>::compute_eigenvectors(kernel_matrix.matrix, n, n);

		for (int32_t i=0; i<n; i++)
		{
			//normalize and trap divide by zero and negative eigenvalues
			for (int32_t j=0; j<n; j++)
				kernel_matrix.matrix[i*n+j]/=CMath::sqrt(CMath::max(1e-16,eigenvalues[i]));
		}

		SG_FREE(eigenvalues);

		m_transformation_matrix = SGMatrix<float64_t>(kernel_matrix.matrix,n,n);
		kernel_matrix.matrix = NULL;
		m_bias_vector = SGVector<float64_t>(n);
		SGVector<float64_t>::fill_vector(m_bias_vector.vector, m_bias_vector.vlen, 0.0);

		cblas_dgemv(CblasColMajor, CblasTrans,
				n, n, 1.0, m_transformation_matrix.matrix, n,
				bias_tmp, 1, 0.0, m_bias_vector.vector, 1);

		float64_t* rowsum = SGMatrix<float64_t>::get_row_sum(m_transformation_matrix.matrix, n, n);
		SGVector<float64_t>::scale_vector(1.0/n, rowsum, n);

		for (int32_t i=0; i<n; i++)
		{
			for (int32_t j=0; j<n; j++)
				m_transformation_matrix.matrix[j+n*i] -= rowsum[i];
		}
		SG_FREE(rowsum);
		SG_FREE(bias_tmp);

		m_initialized=true;
		SG_INFO("Done\n")
		return true;
	}
	return false;
}


SGMatrix<float64_t> CKernelPCA::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(m_initialized)
	CDenseFeatures<float64_t>* simple_features = (CDenseFeatures<float64_t>*)features;

	int32_t num_vectors = simple_features->get_num_vectors();
	int32_t i,j,k;
	int32_t n = m_transformation_matrix.num_cols;

	m_kernel->init(features,m_init_features);

	float64_t* new_feature_matrix = SG_MALLOC(float64_t, m_target_dim*num_vectors);

	for (i=0; i<num_vectors; i++)
	{
		for (j=0; j<m_target_dim; j++)
			new_feature_matrix[i*m_target_dim+j] = m_bias_vector.vector[j];

		for (j=0; j<n; j++)
		{
			float64_t kij = m_kernel->kernel(i,j);

			for (k=0; k<m_target_dim; k++)
				new_feature_matrix[k+i*m_target_dim] += kij*m_transformation_matrix.matrix[(n-k-1)*n+j];
		}
	}

	m_kernel->cleanup();
	simple_features->set_feature_matrix(SGMatrix<float64_t>(new_feature_matrix,m_target_dim,num_vectors));
	return ((CDenseFeatures<float64_t>*)features)->get_feature_matrix();
}

SGVector<float64_t> CKernelPCA::apply_to_feature_vector(SGVector<float64_t> vector)
{
	ASSERT(m_initialized)
	SGVector<float64_t> result = SGVector<float64_t>(m_target_dim);
	m_kernel->init(new CDenseFeatures<float64_t>(SGMatrix<float64_t>(vector.vector,vector.vlen,1)),
	               m_init_features);

	int32_t j,k;
	int32_t n = m_transformation_matrix.num_cols;

	for (j=0; j<m_target_dim; j++)
		result.vector[j] = m_bias_vector.vector[j];

	for (j=0; j<n; j++)
	{
		float64_t kj = m_kernel->kernel(0,j);

		for (k=0; k<m_target_dim; k++)
			result.vector[k] += kj*m_transformation_matrix.matrix[(n-k-1)*n+j];
	}

	m_kernel->cleanup();
	return result;
}

CDenseFeatures<float64_t>* CKernelPCA::apply_to_string_features(CFeatures* features)
{
	ASSERT(m_initialized)

	int32_t num_vectors = features->get_num_vectors();
	int32_t i,j,k;
	int32_t n = m_transformation_matrix.num_cols;

	m_kernel->init(features,m_init_features);

	float64_t* new_feature_matrix = SG_MALLOC(float64_t, m_target_dim*num_vectors);

	for (i=0; i<num_vectors; i++)
	{
		for (j=0; j<m_target_dim; j++)
			new_feature_matrix[i*m_target_dim+j] = m_bias_vector.vector[j];

		for (j=0; j<n; j++)
		{
			float64_t kij = m_kernel->kernel(i,j);

			for (k=0; k<m_target_dim; k++)
				new_feature_matrix[k+i*m_target_dim] += kij*m_transformation_matrix.matrix[(n-k-1)*n+j];
		}
	}

	m_kernel->cleanup();

	return new CDenseFeatures<float64_t>(SGMatrix<float64_t>(new_feature_matrix,m_target_dim,num_vectors));
}

#endif
