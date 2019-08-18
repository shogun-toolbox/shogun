/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Bjoern Esser
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/ratapprox/tracesampler/NormalSampler.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/RandomNamespace.h>

namespace shogun
{

CNormalSampler::CNormalSampler()
	: RandomMixin<CTraceSampler>()
{
	SG_GCDEBUG("{} created ({})", this->get_name(), fmt::ptr(this))
}

CNormalSampler::CNormalSampler(index_t dimension)
	: RandomMixin<CTraceSampler>(dimension)
{
	SG_GCDEBUG("{} created ({})", this->get_name(), fmt::ptr(this))
}

CNormalSampler::~CNormalSampler()
{
	SG_GCDEBUG("{} destroyed ({})", this->get_name(), fmt::ptr(this))
}

void CNormalSampler::precompute()
{
	m_num_samples=1;
}

SGVector<float64_t> CNormalSampler::sample(index_t idx) const
{
	// ignore idx since it doesnt matter, all samples are independent
	SGVector<float64_t> s(m_dimension);
	random::fill_array(s, NormalDistribution<float64_t>(), m_prng);
	return s;
}

}
