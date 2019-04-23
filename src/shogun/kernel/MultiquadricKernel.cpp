/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/MultiquadricKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

MultiquadricKernel::MultiquadricKernel(): Kernel(0), m_distance(NULL), m_coef(0.0001)
{
	init();
}

MultiquadricKernel::MultiquadricKernel(int32_t cache, float64_t coef, std::shared_ptr<Distance> dist)
: Kernel(cache), m_distance(dist), m_coef(coef)
{
	ASSERT(m_distance)
	
	init();
}

MultiquadricKernel::MultiquadricKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t coef, std::shared_ptr<Distance> dist)
: Kernel(10), m_distance(dist), m_coef(coef)
{
	ASSERT(m_distance)
	
	init(l, r);
	init();
}

MultiquadricKernel::~MultiquadricKernel()
{
	cleanup();
	
}

bool MultiquadricKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(m_distance)
	Kernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

float64_t MultiquadricKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	return sqrt(Math::sq(dist) + Math::sq(m_coef));
}

void MultiquadricKernel::init()
{
	SG_ADD(&m_coef, "coef", "Kernel coefficient.", ParameterProperties::HYPER);
	SG_ADD(&m_distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}
