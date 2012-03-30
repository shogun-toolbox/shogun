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
#ifdef HAVE_SUPERLU
	m_use_superlu = true;
#else
	m_use_superlu = false;
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
	                  "whether arpack should be used or not");
	m_parameters->add(&m_use_superlu, "use_superlu",
	                  "whether superlu should be used or not");
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

void CLocallyLinearEmbedding::set_use_superlu(bool use_superlu)
{
#ifndef HAVE_SUPERLU
	if (use_superlu)
		SG_ERROR("No SUPERLU available in this configuration");
#endif 
	m_use_superlu = use_superlu;
}

bool CLocallyLinearEmbedding::get_use_superlu() const
{
	return m_use_superlu;
}

void CLocallyLinearEmbedding::set_use_arpack(bool use_arpack)
{
#ifndef HAVE_ARPACK
	if (use_arpack)
		SG_ERROR("No ARPACK available in this configuration");
#endif 
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
	CDotFeatures* dot_features = (CDotFeatures*)features;
	ASSERT(dot_features);
	SG_REF(dot_features);

	// get and check number of vectors
	int32_t N = dot_features->get_num_vectors();
	if (m_k>=N)
		SG_ERROR("Number of neighbors (%d) should be less than number of objects (%d).\n",
		         m_k, N);

	CKernel* kernel = new CLinearKernel(dot_features,dot_features);
	CKernel* custom_kernel = new CCustomKernel(kernel);

	float64_t* new_features = NULL;

	edrt_options_t options;
	options.method = get_edrt_method();
	options.use_arpack = m_use_arpack;
	options.use_superlu = m_use_superlu;
	options.klle_reconstruction_shift = m_reconstruction_shift;
	options.nullspace_shift = m_nullspace_shift;

	edrt_embedding(options, m_target_dim, N, 0, m_k, NULL,
	               &compute_kernel, NULL, NULL, (void*)kernel,
	               &new_features);

	SG_UNREF(custom_kernel);
	SG_UNREF(kernel);

	SG_UNREF(dot_features);
	return (CFeatures*)(new CSimpleFeatures<float64_t>(SGMatrix<float64_t>(new_features,m_target_dim,N)));
}

float64_t CLocallyLinearEmbedding::compute_kernel(int32_t i, int32_t j, const void* user_data)
{
	return ((CKernel*)user_data)->kernel(i,j);
}


#endif /* HAVE_LAPACK */
