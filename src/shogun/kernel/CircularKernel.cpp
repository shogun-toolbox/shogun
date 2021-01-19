/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/CircularKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/distance/ManhattanMetric.h>

#include <utility>

using namespace shogun;

CircularKernel::CircularKernel(): ShiftInvariantKernel()
{
	init();
	set_cache_size(0);
	set_sigma(1.0);
}

CircularKernel::CircularKernel(int32_t size, float64_t sig)
: ShiftInvariantKernel()
{
	ASSERT(m_distance)
	
	set_cache_size(size);
	set_sigma(sig);
	init();
}

CircularKernel::CircularKernel(
	std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sig)
: ShiftInvariantKernel()
{
	ASSERT(m_distance)
	
	set_sigma(sig);
	set_cache_size(10);
	init();
	init(std::move(l), std::move(r));
}

CircularKernel::~CircularKernel()
{
	cleanup();
	
}

bool CircularKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(m_distance);
	Kernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void CircularKernel::load_serializable_post() noexcept(false)
{
	Kernel::load_serializable_post();
}

void CircularKernel::init()
{
	auto dist = std::make_shared<ManhattanMetric>();
	m_distance = dist;

	SG_ADD(&sigma, "sigma", "Sigma kernel parameter.", ParameterProperties::HYPER);
}

float64_t CircularKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=m_distance->distance(idx_a, idx_b);
	float64_t ds_ratio=dist/sigma;

	if (dist < sigma)
		return (2/M_PI)*acos(-ds_ratio) - (2/M_PI)*ds_ratio*sqrt(1-ds_ratio*ds_ratio);
	else
		return 0;
}
