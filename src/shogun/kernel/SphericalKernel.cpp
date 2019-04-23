/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/kernel/SphericalKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

SphericalKernel::SphericalKernel(): Kernel(0), distance(NULL)
{
	register_params();
	set_sigma(1.0);
}

SphericalKernel::SphericalKernel(int32_t size, float64_t sig, std::shared_ptr<Distance> dist)
: Kernel(size), distance(dist)
{
	ASSERT(distance)
	
	register_params();
	set_sigma(sig);
}

SphericalKernel::SphericalKernel(
	std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t sig, std::shared_ptr<Distance> dist)
: Kernel(10), distance(dist)
{
	ASSERT(distance)
	
	register_params();
	set_sigma(sig);
	init(l, r);
}

SphericalKernel::~SphericalKernel()
{
	cleanup();
	
}

bool SphericalKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(distance)
	Kernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void SphericalKernel::register_params()
{
	SG_ADD(&distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
	SG_ADD(&sigma, "sigma", "Sigma kernel parameter.", ParameterProperties::HYPER);
}

float64_t SphericalKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=distance->distance(idx_a, idx_b);
	float64_t ds_ratio=dist/sigma;

	if (dist < sigma)
		return 1.0-1.5*(ds_ratio)+0.5*(ds_ratio*ds_ratio*ds_ratio);
	else
		return 0;
}
