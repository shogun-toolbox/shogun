/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Sergey Lisitsyn
 */

#include <converter/NeighborhoodPreservingEmbedding.h>
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <io/SGIO.h>
#include <kernel/LinearKernel.h>
#include <lib/tapkee/tapkee_shogun.hpp>

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

#endif /* HAVE_EIGEN3 */
