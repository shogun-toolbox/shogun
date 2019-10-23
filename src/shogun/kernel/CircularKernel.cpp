/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/CircularKernel.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

CircularKernel::CircularKernel(): Kernel(0), distance(NULL)
{
	init();
	set_sigma(1.0);
}

CircularKernel::CircularKernel(int32_t size, float64_t sig, std::shared_ptr<Distance> dist)
: Kernel(size), distance(std::move(dist))
{
	ASSERT(distance)
	

	set_sigma(sig);
	init();
}

CircularKernel::CircularKernel(
	std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sig, std::shared_ptr<Distance> dist)
: Kernel(10), distance(std::move(dist))
{
	ASSERT(distance)
	
	set_sigma(sig);
	init();
	init(std::move(l), std::move(r));
}

CircularKernel::~CircularKernel()
{
	cleanup();
	
}

bool CircularKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(distance)
	Kernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CircularKernel::load_serializable_post() noexcept(false)
{
	Kernel::load_serializable_post();
}

void CircularKernel::init()
{
	SG_ADD(&distance, "distance", "Distance to be used.", ParameterProperties::HYPER);
	SG_ADD(&sigma, "sigma", "Sigma kernel parameter.", ParameterProperties::HYPER);
}

float64_t CircularKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=distance->distance(idx_a, idx_b);
	float64_t ds_ratio=dist/sigma;

	if (dist < sigma)
		return (2/M_PI)*acos(-ds_ratio) - (2/M_PI)*ds_ratio*sqrt(1-ds_ratio*ds_ratio);
	else
		return 0;
}
