/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Björn Esser
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
