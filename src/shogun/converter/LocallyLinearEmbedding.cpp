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
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CLocallyLinearEmbedding::CLocallyLinearEmbedding() :
		CEmbeddingConverter()
{
	m_k = 10;
	m_nullspace_shift = -1e-9;
	m_reconstruction_shift = 1e-3;
	init();
}

void CLocallyLinearEmbedding::init()
{
	SG_ADD(&m_k, "k", "number of neighbors", MS_AVAILABLE);
	SG_ADD(&m_nullspace_shift, "nullspace_shift",
      "nullspace finding regularization shift",MS_NOT_AVAILABLE);
	SG_ADD(&m_reconstruction_shift, "reconstruction_shift",
      "shift used to regularize reconstruction step", MS_NOT_AVAILABLE);
}


CLocallyLinearEmbedding::~CLocallyLinearEmbedding()
{
}

void CLocallyLinearEmbedding::set_k(int32_t k)
{
	ASSERT(k>0)
	m_k = k;
}

int32_t CLocallyLinearEmbedding::get_k() const
{
	return m_k;
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

const char* CLocallyLinearEmbedding::get_name() const
{
	return "LocallyLinearEmbedding";
}

CFeatures* CLocallyLinearEmbedding::apply(CFeatures* features)
{
	// oh my let me dirty cast it
	CKernel* kernel = new CLinearKernel((CDotFeatures*)features,(CDotFeatures*)features);
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_LOCALLY_LINEAR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	SG_UNREF(kernel);
	return embedding;
}

