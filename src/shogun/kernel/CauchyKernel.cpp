/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/CauchyKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CauchyKernel::CauchyKernel(): Kernel(0), m_distance(NULL), m_sigma(1.0)
{
	init();
}

CauchyKernel::CauchyKernel(int32_t cache, float64_t sigma, std::shared_ptr<Distance> dist)
: Kernel(cache), m_distance(dist), m_sigma(sigma)
{
	init();
	ASSERT(m_distance)
	
}

CauchyKernel::CauchyKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sigma, std::shared_ptr<Distance> dist)
: Kernel(10), m_distance(dist), m_sigma(sigma)
{
	init();
	ASSERT(m_distance)
	
	init(l, r);
}

CauchyKernel::~CauchyKernel()
{
	cleanup();
	
}

bool CauchyKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(m_distance)
	Kernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void CauchyKernel::init()
{
	SG_ADD(&m_sigma, "sigma", "Sigma kernel parameter.", ParameterProperties::HYPER);
	SG_ADD(&m_distance, "distance", "Distance to be used.", ParameterProperties::HYPER);
}

float64_t CauchyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	return 1.0/(1.0+dist*dist/m_sigma);
}
