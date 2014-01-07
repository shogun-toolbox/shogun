/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Fernando José Iglesias García
 * Copyright (C) 2012-2013 Fernando José Iglesias García
 */

#include <converter/StochasticProximityEmbedding.h>
#include <lib/config.h>
#ifdef HAVE_EIGEN3
#include <io/SGIO.h>
#include <lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CStochasticProximityEmbedding::CStochasticProximityEmbedding() :
	CEmbeddingConverter()
{
	// Initialize to default values
	m_k         = 12;
	m_nupdates  = 100;
	m_strategy  = SPE_GLOBAL;
	m_tolerance = 1e-5;
	m_max_iteration  = 0;

	init();
}

void CStochasticProximityEmbedding::init()
{
	SG_ADD(&m_k, "m_k", "Number of neighbors", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_strategy, "m_strategy", "SPE strategy",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "m_tolerance", "Regularization parameter",
			MS_NOT_AVAILABLE);
	SG_ADD(&m_max_iteration, "max_iteration", "maximum number of iterations",
			MS_NOT_AVAILABLE);
}

CStochasticProximityEmbedding::~CStochasticProximityEmbedding()
{
}

void CStochasticProximityEmbedding::set_k(int32_t k)
{
	if ( k <= 0 )
		SG_ERROR("Number of neighbors k must be greater than 0")

	m_k = k;
}

int32_t CStochasticProximityEmbedding::get_k() const
{
	return m_k;
}

void CStochasticProximityEmbedding::set_strategy(ESPEStrategy strategy)
{
	m_strategy = strategy;
}

ESPEStrategy CStochasticProximityEmbedding::get_strategy() const
{
	return m_strategy;
}

void CStochasticProximityEmbedding::set_tolerance(float32_t tolerance)
{
	if ( tolerance <= 0 )
		SG_ERROR("Tolerance regularization parameter must be greater "
			 "than 0");

	m_tolerance = tolerance;
}

int32_t CStochasticProximityEmbedding::get_tolerance() const
{
	return m_tolerance;
}

void CStochasticProximityEmbedding::set_nupdates(int32_t nupdates)
{
	if ( nupdates <= 0 )
		SG_ERROR("The number of updates must be greater than 0")

	m_nupdates = nupdates;
}

int32_t CStochasticProximityEmbedding::get_nupdates() const
{
	return m_nupdates;
}

void CStochasticProximityEmbedding::set_max_iteration(const int32_t max_iteration)
{
	m_max_iteration = max_iteration;
}

int32_t CStochasticProximityEmbedding::get_max_iteration() const
{
	return m_max_iteration;
}

const char * CStochasticProximityEmbedding::get_name() const
{
	return "StochasticProximityEmbedding";
}

CFeatures* CStochasticProximityEmbedding::apply(CFeatures* features)
{
	if ( !features )
		SG_ERROR("Features are required to apply SPE\n")

	// Shorthand for the DenseFeatures
	CDenseFeatures< float64_t >* simple_features =
		(CDenseFeatures< float64_t >*) features;
	SG_REF(features);

	// Get and check the number of vectors
	int32_t N = simple_features->get_num_vectors();
	if ( m_strategy == SPE_LOCAL && m_k >= N )
		SG_ERROR("The number of neighbors (%d) must be less than "
		         "the number of vectors (%d)\n", m_k, N);

	if ( 2*m_nupdates > N )
		SG_ERROR("The number of vectors (%d) must be at least two times "
			 "the number of updates (%d)\n", N, m_nupdates);

	m_distance->init(simple_features, simple_features);
	CDenseFeatures< float64_t >* embedding = embed_distance(m_distance);
	m_distance->remove_lhs_and_rhs();

	SG_UNREF(features);
	return (CFeatures*)embedding;
}

CDenseFeatures< float64_t >* CStochasticProximityEmbedding::embed_distance(CDistance* distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_neighbors = m_k;
	parameters.method = SHOGUN_STOCHASTIC_PROXIMITY_EMBEDDING;
	parameters.target_dimension = m_target_dim;
	parameters.spe_num_updates = m_nupdates;
	parameters.spe_tolerance = m_tolerance;
	parameters.distance = distance;
	parameters.spe_global_strategy = (m_strategy==SPE_GLOBAL);
	parameters.max_iteration = m_max_iteration;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

#endif /* HAVE_EIGEN3 */
