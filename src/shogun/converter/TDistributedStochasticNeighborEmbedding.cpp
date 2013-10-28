/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Vladyslav S. Gorbatiuk
 * Copyright (C) 2011-2013 Vladyslav S. Gorbatiuk
 */

#include <shogun/converter/TDistributedStochasticNeighborEmbedding.h>
#ifdef HAVE_EIGEN3
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CTDistributedStochasticNeighborEmbedding::CTDistributedStochasticNeighborEmbedding() :
		CEmbeddingConverter()
{
	// Default values
	m_perplexity = 30.0;
	m_theta = 0.5;
	init();
}

void CTDistributedStochasticNeighborEmbedding::init()
{
	SG_ADD(&m_perplexity, "perplexity", "perplexity", MS_NOT_AVAILABLE);
	SG_ADD(&m_theta, "theta", "learning rate", MS_NOT_AVAILABLE);
}

CTDistributedStochasticNeighborEmbedding::~CTDistributedStochasticNeighborEmbedding()
{
}

const char* CTDistributedStochasticNeighborEmbedding::get_name() const
{
	return "TDistributedStochasticNeighborEmbedding";
}

void CTDistributedStochasticNeighborEmbedding::set_theta(const float64_t theta)
{

	m_theta = theta;
}

float64_t CTDistributedStochasticNeighborEmbedding::get_theta() const
{
	return m_theta;
}

void CTDistributedStochasticNeighborEmbedding::set_perplexity(const float64_t perplexity)
{
	m_perplexity = perplexity;
}

float64_t CTDistributedStochasticNeighborEmbedding::get_perplexity() const
{
	return m_perplexity;
}

CFeatures* CTDistributedStochasticNeighborEmbedding::apply(CFeatures* features)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.sne_theta = m_theta;
	parameters.sne_perplexity = m_perplexity;
	parameters.features = (CDotFeatures*)features;

	parameters.method = SHOGUN_TDISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

#endif /* HAVE_EIGEN */
