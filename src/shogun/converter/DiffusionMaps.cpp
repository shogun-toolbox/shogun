/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/DiffusionMaps.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/arpack_d.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>

using namespace shogun;

CDiffusionMaps::CDiffusionMaps() :
		CEmbeddingConverter()
{
	m_t = 10;
	
	init();
}

void CDiffusionMaps::init()
{
	m_parameters->add(&m_t, "t", "number of steps");
}

CDiffusionMaps::~CDiffusionMaps()
{
}

void CDiffusionMaps::set_t(int32_t t)
{
	m_t = t;
}

int32_t CDiffusionMaps::get_t() const
{
	return m_t;
}

const char* CDiffusionMaps::get_name() const
{
	return "DiffusionMaps";
};

CFeatures* CDiffusionMaps::apply(CFeatures* features)
{
	ASSERT(features);
	if (!(features->get_feature_class()==C_SIMPLE &&
	      features->get_feature_type()==F_DREAL))
	{
		SG_ERROR("Given features are not of SimpleRealFeatures type.\n");
	}
	// shorthand for simplefeatures
	CSimpleFeatures<float64_t>* simple_features = (CSimpleFeatures<float64_t>*) features;
	SG_REF(features);

	// get dimensionality and number of vectors of data
	int32_t N = simple_features->get_num_vectors();
	int32_t dim = simple_features->get_num_features();

	// loop variables
	int32_t i,j;

	// compute distance matrix
	ASSERT(m_kernel);
	m_kernel->init(simple_features,simple_features);
	SGMatrix<float64_t> kernel_matrix = m_kernel->get_kernel_matrix();
	m_kernel->cleanup();
	
	float64_t* p_vector = SG_CALLOC(float64_t, N);
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			p_vector[i] += kernel_matrix.matrix[j*N+i];
		}
	}

	float64_t* p_matrix = SG_CALLOC(float64_t, N*N);
	cblas_dger(CblasColMajor,N,N,1.0,p_vector,1,p_vector,1,p_matrix,N);
	for (i=0; i<N*N; i++)
	{
		kernel_matrix.matrix[i] /= CMath::pow(p_matrix[i], m_t);
	}
	SG_FREE(p_matrix);

	for (i=0; i<N; i++)
	{
		p_vector[i] = 0.0;
		for (j=0; j<N; j++)
		{
			p_vector[i] += kernel_matrix.matrix[j*N+i];
		}
		p_vector[i] = CMath::sqrt(p_vector[i]);
	}
	float64_t ppt = cblas_ddot(N,p_vector,1,p_vector,1);
	SG_FREE(p_vector);

	for (i=0; i<N*N; i++)
	{
		kernel_matrix.matrix[i] /= ppt;
	}


	float64_t* s_values = SG_MALLOC(float64_t, N);

	float64_t* kkt_matrix = SG_MALLOC(float64_t, N*N);

	cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,
	            N,N,N,
	            1.0,kernel_matrix.matrix,N,
	            kernel_matrix.matrix,N,
	            0.0,kkt_matrix,N);

	int32_t info = 0;

	wrap_dsyevr('V','U',N,kkt_matrix,N,N-m_target_dim,N,s_values,kernel_matrix.matrix,&info);
	if (info)
		SG_ERROR("DGESVD failed with %d code", info);

	SG_FREE(s_values);
/*
	int32_t info = 0;
	wrap_dgesvd('O','N',N,N,kernel_matrix.matrix,N,s_values,NULL,1,NULL,1,&info);
*/

	SG_FREE(kkt_matrix);
	SGMatrix<float64_t> new_feature_matrix = SGMatrix<float64_t>(m_target_dim,N);

	for (i=0; i<m_target_dim; i++)
	{
		for (j=0; j<N; j++)
			new_feature_matrix[j*m_target_dim+i] = kernel_matrix.matrix[(m_target_dim-i-1)*N+j]/kernel_matrix.matrix[(m_target_dim)*N+j];
	}
	kernel_matrix.destroy_matrix();

	SG_UNREF(features);
	return (CFeatures*)(new CSimpleFeatures<float64_t>(new_feature_matrix));
}

#endif /* HAVE_LAPACK */
