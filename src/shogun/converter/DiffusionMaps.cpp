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
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>

using namespace shogun;

CDiffusionMaps::CDiffusionMaps() :
		CEmbeddingConverter()
{
	m_t = 10;
	set_kernel(new CGaussianKernel(10,1.0));

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
	// shorthand for simplefeatures
	SG_REF(features);
	// compute distance matrix
	ASSERT(m_kernel);
	m_kernel->init(features,features);
	CSimpleFeatures<float64_t>* embedding = embed_kernel(m_kernel);
	m_kernel->cleanup();
	SG_UNREF(features);
	return (CFeatures*)embedding;
}

CSimpleFeatures<float64_t>* CDiffusionMaps::embed_kernel(CKernel* kernel)
{
#ifdef HAVE_ARPACK
	bool use_arpack = true;
#else
	bool use_arpack = false;
#endif
	int32_t i,j;
	SGMatrix<float64_t> kernel_matrix = kernel->get_kernel_matrix();
	ASSERT(kernel_matrix.num_rows==kernel_matrix.num_cols);
	int32_t N = kernel_matrix.num_rows;

	float64_t* p_vector = SG_CALLOC(float64_t, N);
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			p_vector[i] += kernel_matrix.matrix[j*N+i];
		}
	}
	//CMath::display_matrix(kernel_matrix.matrix,N,N,"K");
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			kernel_matrix.matrix[i*N+j] /= CMath::pow(p_vector[i]*p_vector[j], m_t);
		}
	}
	//CMath::display_matrix(kernel_matrix.matrix,N,N,"K");

	for (i=0; i<N; i++)
	{
		p_vector[i] = 0.0;
		for (j=0; j<N; j++)
		{
			p_vector[i] += kernel_matrix.matrix[j*N+i];
		}
		p_vector[i] = CMath::sqrt(p_vector[i]);
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			kernel_matrix.matrix[i*N+j] /= p_vector[i]*p_vector[j];
		}
	}

	SG_FREE(p_vector);
	float64_t* s_values = SG_MALLOC(float64_t, N);

	int32_t info = 0;
	SGMatrix<float64_t> new_feature_matrix = SGMatrix<float64_t>(m_target_dim,N);

	if (use_arpack)
	{
#ifdef HAVE_ARPACK
		arpack_dsxupd(kernel_matrix.matrix,NULL,false,N,m_target_dim,"LA",false,1,false,true,0.0,0.0,s_values,kernel_matrix.matrix,info);
#endif /* HAVE_ARPACK */
		for (i=0; i<m_target_dim; i++)
		{
			for (j=0; j<N; j++)
				new_feature_matrix[j*m_target_dim+i] = kernel_matrix[j*m_target_dim+i];
		}
	}
	else
	{
		SG_WARNING("LAPACK does not provide efficient routines to construct embedding (this may take time). Consider installing ARPACK.");
		wrap_dgesvd('O','N',N,N,kernel_matrix.matrix,N,s_values,NULL,1,NULL,1,&info);
		for (i=0; i<m_target_dim; i++)
		{
			for (j=0; j<N; j++)
				new_feature_matrix[j*m_target_dim+i] =
				    kernel_matrix[(m_target_dim-i-1)*N+j];
		}
	}
	if (info)
		SG_ERROR("Eigenproblem solving  failed with %d code", info);

	kernel_matrix.destroy_matrix();
	SG_FREE(s_values);

	return new CSimpleFeatures<float64_t>(new_feature_matrix);
}
#endif /* HAVE_LAPACK */
