/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/distance/EuclideanDistance.h>

using namespace shogun;

namespace shogun
{
CEmbeddingConverter::CEmbeddingConverter()
: CConverter()
{
	m_target_dim = 1;
	m_distance = new CEuclideanDistance();
	m_kernel = new CLinearKernel();

	init();
}

CEmbeddingConverter::~CEmbeddingConverter()
{
	SG_UNREF(m_distance);
	SG_UNREF(m_kernel);
}

CDenseFeatures<float64_t>* CEmbeddingConverter::embed(CFeatures* features)
{
	return (CDenseFeatures<float64_t>*)apply(features);
}

void CEmbeddingConverter::set_target_dim(int32_t dim)
{
	ASSERT(dim>0)
	m_target_dim = dim;
}

int32_t CEmbeddingConverter::get_target_dim() const
{
	return m_target_dim;
}

void CEmbeddingConverter::set_distance(CDistance* distance)
{
	SG_REF(distance);
	SG_UNREF(m_distance);
	m_distance = distance;
}

CDistance* CEmbeddingConverter::get_distance() const
{
	SG_REF(m_distance);
	return m_distance;
}

void CEmbeddingConverter::set_kernel(CKernel* kernel)
{
	SG_REF(kernel);
	SG_UNREF(m_kernel);
	m_kernel = kernel;
}

CKernel* CEmbeddingConverter::get_kernel() const
{
	SG_REF(m_kernel);
	return m_kernel;
}

void CEmbeddingConverter::init()
{
	SG_ADD(&m_target_dim, "target_dim",
      "target dimensionality of preprocessor", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_distance, "distance",
	    "distance to be used for embedding", MS_AVAILABLE);
	SG_ADD((CSGObject**)&m_kernel, "kernel", "kernel to be used for embedding",
	    MS_AVAILABLE);
}
}
