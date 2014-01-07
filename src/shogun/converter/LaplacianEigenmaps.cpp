/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#include <converter/LaplacianEigenmaps.h>
#include <converter/EmbeddingConverter.h>
#ifdef HAVE_EIGEN3
#include <distance/EuclideanDistance.h>
#include <lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CLaplacianEigenmaps::CLaplacianEigenmaps() :
		CEmbeddingConverter()
{
	m_k = 3;
	m_tau = 1.0;

	init();
}

void CLaplacianEigenmaps::init()
{
	SG_ADD(&m_k, "k", "number of neighbors", MS_AVAILABLE);
	SG_ADD(&m_tau, "tau", "heat distribution coefficient", MS_AVAILABLE);
}

CLaplacianEigenmaps::~CLaplacianEigenmaps()
{
}

void CLaplacianEigenmaps::set_k(int32_t k)
{
	ASSERT(k>0)
	m_k = k;
}

int32_t CLaplacianEigenmaps::get_k() const
{
	return m_k;
}

void CLaplacianEigenmaps::set_tau(float64_t tau)
{
	m_tau = tau;
}

float64_t CLaplacianEigenmaps::get_tau() const
{
	return m_tau;
}

const char* CLaplacianEigenmaps::get_name() const
{
	return "LaplacianEigenmaps";
};

CFeatures* CLaplacianEigenmaps::apply(CFeatures* features)
{
	// shorthand for simplefeatures
	SG_REF(features);

	// get dimensionality and number of vectors of data
	int32_t N = features->get_num_vectors();
	ASSERT(m_k<N)
	ASSERT(m_target_dim<N)

	// compute distance matrix
	ASSERT(m_distance)
	m_distance->init(features,features);
	CDenseFeatures<float64_t>* embedding = embed_distance(m_distance);
	m_distance->remove_lhs_and_rhs();
	SG_UNREF(features);
	return (CFeatures*)embedding;
}

CDenseFeatures<float64_t>* CLaplacianEigenmaps::embed_distance(CDistance* distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.gaussian_kernel_width = m_tau;
	parameters.method = SHOGUN_LAPLACIAN_EIGENMAPS;
	parameters.target_dimension = m_target_dim;
	parameters.distance = distance;
	return tapkee_embed(parameters);
}
#endif /* HAVE_EIGEN3 */
