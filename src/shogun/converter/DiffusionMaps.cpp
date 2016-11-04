/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/DiffusionMaps.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/lib/config.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CDiffusionMaps::CDiffusionMaps() :
		CEmbeddingConverter()
{
	m_t = 10;
	m_width = 1.0;
	set_distance(new CEuclideanDistance());

	init();
}

void CDiffusionMaps::init()
{
	SG_ADD(&m_t, "t", "number of steps", MS_AVAILABLE);
	SG_ADD(&m_width, "width", "gaussian kernel width", MS_AVAILABLE);
}

CDiffusionMaps::~CDiffusionMaps()
{
}

void CDiffusionMaps::set_t(int32_t t)
{
	m_t = t;
}

int32_t CDiffusionMaps::get_t() const
{
	return m_t;
}

void CDiffusionMaps::set_width(float64_t width)
{
	m_width = width;
}

float64_t CDiffusionMaps::get_width() const
{
	return m_width;
}

const char* CDiffusionMaps::get_name() const
{
	return "DiffusionMaps";
};

CFeatures* CDiffusionMaps::apply(CFeatures* features)
{
	ASSERT(features)
	// shorthand for simplefeatures
	SG_REF(features);
	// compute distance matrix
	ASSERT(m_distance)
	m_distance->init(features,features);
	CDenseFeatures<float64_t>* embedding = embed_distance(m_distance);
	m_distance->cleanup();
	SG_UNREF(features);
	return (CFeatures*)embedding;
}

CDenseFeatures<float64_t>* CDiffusionMaps::embed_distance(CDistance* distance)
{
	TAPKEE_PARAMETERS_FOR_SHOGUN parameters;
	parameters.n_timesteps = m_t;
	parameters.gaussian_kernel_width = m_width;
	parameters.method = SHOGUN_DIFFUSION_MAPS;
	parameters.target_dimension = m_target_dim;
	parameters.distance = distance;
	return tapkee_embed(parameters);
}
