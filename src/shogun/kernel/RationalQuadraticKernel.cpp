/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/RationalQuadraticKernel.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

RationalQuadraticKernel::RationalQuadraticKernel(): Kernel(0), m_distance(NULL), m_coef(0.001)
{
	init();
}

RationalQuadraticKernel::RationalQuadraticKernel(int32_t cache, float64_t coef, std::shared_ptr<Distance> distance)
: Kernel(cache), m_distance(std::move(distance)), m_coef(coef)
{
	ASSERT(m_distance)
	
	init();
}

RationalQuadraticKernel::RationalQuadraticKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t coef, std::shared_ptr<Distance> dist)
: Kernel(10), m_distance(std::move(dist)), m_coef(coef)
{
	ASSERT(m_distance)
	
	init();
	init(std::move(l), std::move(r));
}

RationalQuadraticKernel::~RationalQuadraticKernel()
{
	cleanup();
	
}

bool RationalQuadraticKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(m_distance)
	Kernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

float64_t RationalQuadraticKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	float64_t pDist = dist * dist;
	return 1-pDist/(pDist+m_coef);
}

void RationalQuadraticKernel::init()
{
	SG_ADD(&m_coef, "coef", "Kernel coefficient.", ParameterProperties::HYPER);
	SG_ADD(&m_distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}

