/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evan Shelhamer
 */

#include <shogun/kernel/BesselKernel.h>
#include <shogun/mathematics/Math.h>
#include <math.h>

using namespace shogun;

BesselKernel::BesselKernel():DistanceKernel(),order(0.0),degree(0)
{
	init();
}

BesselKernel::BesselKernel(int32_t size, float64_t v, float64_t w,
		int32_t n, std::shared_ptr<Distance> dist) :
	DistanceKernel(size,w,dist), order(v), degree(n)
{
	ASSERT(distance)
	
	init();
}

BesselKernel::BesselKernel(std::shared_ptr<Features> l, std::shared_ptr<Features> r, float64_t v,
		float64_t w, int32_t n, std::shared_ptr<Distance> dist, int32_t size) :
	DistanceKernel(size,w,dist), order(v), degree(n)
{
	init();
	ASSERT(distance)
	
	init(l,r);
}

BesselKernel::~BesselKernel()
{
	cleanup();
	
}

void BesselKernel::cleanup()
{
}

bool BesselKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(distance)
	DistanceKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void BesselKernel::init()
{
	SG_ADD(&order, "order", "Kernel order.", ParameterProperties::HYPER);
	SG_ADD(&degree, "degree", "Kernel degree.", ParameterProperties::HYPER);
}

float64_t BesselKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return jn(order,dist/width)/Math::pow(dist,-degree*order);
}
