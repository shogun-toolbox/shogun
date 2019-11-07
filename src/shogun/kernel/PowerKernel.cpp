/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evan Shelhamer
 */

#include <shogun/kernel/PowerKernel.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

PowerKernel::PowerKernel(): Kernel(0), distance(NULL), m_degree(1.8)
{
	init();
}

PowerKernel::PowerKernel(int32_t cache, float64_t degree, std::shared_ptr<Distance> dist)
: Kernel(cache), distance(std::move(dist)), m_degree(degree)
{
	init();
	ASSERT(distance)
	
}

PowerKernel::PowerKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t degree, std::shared_ptr<Distance> dist)
: Kernel(10), distance(std::move(dist)), m_degree(degree)
{
	init();
	ASSERT(distance)
	
	init(std::move(l), std::move(r));
}

PowerKernel::~PowerKernel()
{
	cleanup();
	
}

bool PowerKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(distance)
	Kernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void PowerKernel::init()
{
	SG_ADD(&m_degree, "degree", "Degree kernel parameter.", ParameterProperties::HYPER);
	SG_ADD(&distance, "distance", "Distance to be used.",
			ParameterProperties::HYPER);
}

float64_t PowerKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	float64_t temp = pow(dist, m_degree);
	return -temp;
}
