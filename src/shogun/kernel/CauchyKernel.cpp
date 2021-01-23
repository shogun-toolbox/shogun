/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/distance/EuclideanDistance.h>
#include <shogun/kernel/CauchyKernel.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

CauchyKernel::CauchyKernel(): ShiftInvariantKernel(), m_sigma(1.0)
{
	init();
}

CauchyKernel::CauchyKernel(int32_t cache, float64_t sigma)
: ShiftInvariantKernel(), m_sigma(sigma)
{
	init();
	set_cache_size(cache);
	ASSERT(m_distance)
	
}

CauchyKernel::CauchyKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sigma)
: ShiftInvariantKernel(), m_sigma(sigma)
{
	init();
	set_cache_size(10);
	ASSERT(m_distance)
	
	init(std::move(l), std::move(r));
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
	auto dist = std::make_shared<EuclideanDistance>();
	m_distance = std::make_shared<EuclideanDistance>();

	SG_ADD(&m_sigma, "sigma", "Sigma kernel parameter.", ParameterProperties::HYPER);
}

float64_t CauchyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);
	return 1.0/(1.0+dist*dist/m_sigma);
}
