/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/KernelLocallyLinearEmbedding.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

CKernelLocallyLinearEmbedding::CKernelLocallyLinearEmbedding() :
		CLocallyLinearEmbedding()
{
}

CKernelLocallyLinearEmbedding::CKernelLocallyLinearEmbedding(CKernel* kernel) :
		CLocallyLinearEmbedding()
{
	set_kernel(kernel);
}

const char* CKernelLocallyLinearEmbedding::get_name() const
{
	return "KernelLocallyLinearEmbedding";
};

CKernelLocallyLinearEmbedding::~CKernelLocallyLinearEmbedding()
{
}

CFeatures* CKernelLocallyLinearEmbedding::apply(CFeatures* features)
{
	ASSERT(features)
	SG_REF(features);

	// get dimensionality and number of vectors of data
	int32_t N = features->get_num_vectors();
	if (m_k>=N)
		SG_ERROR("Number of neighbors (%d) should be less than number of objects (%d).\n",
		         m_k, N);

	// compute kernel matrix
	ASSERT(m_kernel)
	m_kernel->init(features,features);
	CDenseFeatures<float64_t>* embedding = embed_kernel(m_kernel);
	m_kernel->cleanup();
	SG_UNREF(features);
	return (CFeatures*)embedding;
}

CDenseFeatures<float64_t>* CKernelLocallyLinearEmbedding::embed_kernel(CKernel* kernel)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_KERNEL_LOCALLY_LINEAR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

