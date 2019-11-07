/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/LogKernel.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

LogKernel::LogKernel(): Kernel(0), m_distance(NULL), m_degree(1.8)
{
	init();
}

LogKernel::LogKernel(int32_t cache, float64_t degree, std::shared_ptr<Distance> dist)
: Kernel(cache), m_distance(std::move(dist)), m_degree(degree)
{
	init();
	ASSERT(m_distance)
	
}

LogKernel::LogKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t degree, std::shared_ptr<Distance> dist)
: Kernel(10), m_distance(std::move(dist)), m_degree(degree)
{
	init();
	ASSERT(m_distance)
	
	init(std::move(l), std::move(r));
}

LogKernel::~LogKernel()
{
	cleanup();
	
}

bool LogKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(m_distance)
	Kernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void LogKernel::init()
{
	SG_ADD(&m_degree, "degree", "Degree kernel parameter.", ParameterProperties::HYPER);
	SG_ADD(&m_distance, "distance", "Distance to be used.", ParameterProperties::HYPER);
}

float64_t LogKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	float64_t temp = pow(dist, m_degree);
	temp = log(temp + 1);
	return -temp;
}
