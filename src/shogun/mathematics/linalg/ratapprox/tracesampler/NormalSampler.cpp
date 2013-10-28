/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Random.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>

namespace shogun
{

CNormalSampler::CNormalSampler()
	: CTraceSampler()
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CNormalSampler::CNormalSampler(index_t dimension)
	: CTraceSampler(dimension)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

CNormalSampler::~CNormalSampler()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void CNormalSampler::precompute()
{
	m_num_samples=1;
}

SGVector<float64_t> CNormalSampler::sample(index_t idx) const
{
	// ignore idx since it doesnt matter, all samples are independent
	SGVector<float64_t> s(m_dimension);

	for (index_t i=0; i<m_dimension; ++i)
		s[i]=sg_rand->std_normal_distrib();

	return s;
}

}
