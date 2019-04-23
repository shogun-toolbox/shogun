/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/InverseMultiQuadricKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

InverseMultiQuadricKernel::InverseMultiQuadricKernel(): Kernel(0), distance(NULL), coef(0.0001)
{
	init();
}

InverseMultiQuadricKernel::InverseMultiQuadricKernel(int32_t cache, float64_t coefficient, std::shared_ptr<Distance> dist)
: Kernel(cache), distance(dist), coef(coefficient)
{
	
	init();
}

InverseMultiQuadricKernel::InverseMultiQuadricKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t coefficient, std::shared_ptr<Distance> dist)
: Kernel(10), distance(dist), coef(coefficient)
{
	
	init();
	init(l, r);
}

InverseMultiQuadricKernel::~InverseMultiQuadricKernel()
{
	cleanup();
	
}

bool InverseMultiQuadricKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	Kernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void InverseMultiQuadricKernel::load_serializable_post() noexcept(false)
{
	Kernel::load_serializable_post();
}

void InverseMultiQuadricKernel::init()
{
	SG_ADD(&coef, "coef", "Kernel Coefficient.", ParameterProperties::HYPER);
	SG_ADD(&distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}

float64_t InverseMultiQuadricKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return 1/sqrt(dist*dist + coef*coef);
}
