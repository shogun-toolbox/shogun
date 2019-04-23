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

NormalSampler::NormalSampler()
	: RandomMixin<TraceSampler>()
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

NormalSampler::NormalSampler(index_t dimension)
	: RandomMixin<TraceSampler>(dimension)
{
	SG_GCDEBUG("%s created (%p)\n", this->get_name(), this)
}

NormalSampler::~NormalSampler()
{
	SG_GCDEBUG("%s destroyed (%p)\n", this->get_name(), this)
}

void NormalSampler::precompute()
{
	m_num_samples=1;
}

SGVector<float64_t> NormalSampler::sample(index_t idx) const
{
	// ignore idx since it doesnt matter, all samples are independent
	SGVector<float64_t> s(m_dimension);
	random::fill_array(s, NormalDistribution<float64_t>(), m_prng);
	return s;
}

}
