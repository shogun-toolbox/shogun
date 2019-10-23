/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/WaveKernel.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

WaveKernel::WaveKernel(): Kernel(0), m_distance(NULL), m_theta(1.0)
{
	init();
}

WaveKernel::WaveKernel(int32_t cache, float64_t theta, std::shared_ptr<Distance> dist)
: Kernel(cache), m_distance(std::move(dist)), m_theta(theta)
{
	init();
	ASSERT(m_distance)
	
}

WaveKernel::WaveKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t theta, std::shared_ptr<Distance> dist)
: Kernel(10), m_distance(std::move(dist)), m_theta(theta)
{
	init();
	ASSERT(m_distance)
	
	init(std::move(l), std::move(r));
}

WaveKernel::~WaveKernel()
{
	cleanup();
	
}

bool WaveKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(m_distance)
	Kernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void WaveKernel::init()
{
	SG_ADD(&m_theta, "theta", "Theta kernel parameter.", ParameterProperties::HYPER);
	SG_ADD(&m_distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}

float64_t WaveKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);

	if (dist==0.0)
		return 1.0;

	return (m_theta/dist)*sin(dist/m_theta);
}
