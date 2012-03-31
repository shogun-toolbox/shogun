/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/converter/StochasticProximityEmbedding.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/lib/CoverTree.h>
#include <shogun/mathematics/Math.h>
#include <shogun/distance/Distance.h>

using namespace shogun;

CStochasticProximityEmbedding::CStochasticProximityEmbedding() : 
	CEmbeddingConverter()
{
	m_k = 12;
	m_strategy = SPE_GLOBAL;
	m_tolerance = 1e-5;

	init();
}

void CStochasticProximityEmbedding::init()
{
	SG_ADD(&m_k, "m_k", "Number of neighbours", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &m_strategy, "m_strategy", "SPE strategy", MS_NOT_AVAILABLE);
	SG_ADD(&m_tolerance, "m_tolerance", "Regularization parameter", MS_NOT_AVAILABLE);
}

CStochasticProximityEmbedding::~CStochasticProximityEmbedding()
{
}

inline void CStochasticProximityEmbedding::set_k(int32_t k)
{
	if ( k <= 0 )
		SG_ERROR("Number of neighbours k must be greater than 0");

	m_k = k;
}

inline int32_t CStochasticProximityEmbedding::get_k() const
{
	return m_k;
}

inline void CStochasticProximityEmbedding::set_tolerance(float32_t tolerance)
{
	if ( tolerance <= 0 )
		SG_ERROR("Tolerance regularization parameter must be greater than 0");

	m_tolerance = tolerance;
}

inline int32_t CStochasticProximityEmbedding::get_tolerance() const
{
	return m_tolerance;
}

inline const char * CStochasticProximityEmbedding::get_name() const
{
	return "StochasticProximityEmbedding";
}

CFeatures* CStochasticProximityEmbedding::apply(CFeatures* features)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}

#endif /* HAVE_LAPACK */
