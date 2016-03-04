/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Sergey Lisitsyn
 */

#include <shogun/converter/NeighborhoodPreservingEmbedding.h>
#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CNeighborhoodPreservingEmbedding::CNeighborhoodPreservingEmbedding() :
		CLocallyLinearEmbedding()
{
}

CNeighborhoodPreservingEmbedding::~CNeighborhoodPreservingEmbedding()
{
}

const char* CNeighborhoodPreservingEmbedding::get_name() const
{
	return "NeighborhoodPreservingEmbedding";
}

CFeatures* CNeighborhoodPreservingEmbedding::apply(CFeatures* features)
{
	CKernel* kernel = new CLinearKernel((CDotFeatures*)features,(CDotFeatures*)features);
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.eigenshift = m_nullspace_shift;
	parameters.method = SHOGUN_NEIGHBORHOOD_PRESERVING_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.kernel = kernel;
	parameters.features = (CDotFeatures*)features;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	SG_UNREF(kernel);
	return embedding;
}

