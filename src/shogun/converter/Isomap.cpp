/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Sergey Lisitsyn
 * Copyright (C) 2011-2013 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/Isomap.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CIsomap::CIsomap() : CMultidimensionalScaling()
{
	m_k = 3;

	init();
}

void CIsomap::init()
{
	SG_ADD(&m_k, "k", "number of neighbors", MS_AVAILABLE);
}

CIsomap::~CIsomap()
{
}

void CIsomap::set_k(int32_t k)
{
	ASSERT(k>0)
	m_k = k;
}

int32_t CIsomap::get_k() const
{
	return m_k;
}

const char* CIsomap::get_name() const
{
	return "Isomap";
}

CDenseFeatures<float64_t>* CIsomap::embed_distance(CDistance* distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	if (m_landmark)
	{
		parameters.method = SHOGUN_LANDMARK_ISOMAP;
		parameters.landmark_ratio = float64_t(m_landmark_number)/distance->get_num_vec_lhs();
	}
	else
	{
		parameters.method = SHOGUN_ISOMAP;
	}
	parameters.n_neighbors = m_k;
	parameters.target_dimension = m_target_dim;
	parameters.distance = distance;
	CDenseFeatures<float64_t>* embedding = tapkee_embed(parameters);
	return embedding;
}

