/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/LocalityPreservingProjections.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CLocalityPreservingProjections::CLocalityPreservingProjections() :
		CLaplacianEigenmaps()
{
}

CLocalityPreservingProjections::~CLocalityPreservingProjections()
{
}

const char* CLocalityPreservingProjections::get_name() const
{
	return "LocalityPreservingProjections";
};

CFeatures* CLocalityPreservingProjections::apply(CFeatures* features)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	m_distance->init(features,features);
	parameters.n_neighbors = m_k;
	parameters.gaussian_kernel_width = m_tau;
	parameters.method = SHOGUN_LOCALITY_PRESERVING_PROJECTIONS;
	parameters.target_dimension = m_target_dim;
	parameters.distance = m_distance;
	parameters.features = (CDotFeatures*)features;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

