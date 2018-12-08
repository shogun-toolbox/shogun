/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evan Shelhamer
 */

#include <shogun/kernel/BesselKernel.h>
#include <shogun/mathematics/Math.h>
#include <math.h>

using namespace shogun;

CBesselKernel::CBesselKernel():CDistanceKernel(),order(0.0),degree(0)
{
	init();
}

CBesselKernel::CBesselKernel(int32_t size, float64_t v, float64_t w,
		int32_t n, CDistance* dist) :
	CDistanceKernel(size,w,dist), order(v), degree(n)
{
	ASSERT(distance)
	SG_REF(distance);
	init();
}

CBesselKernel::CBesselKernel(CFeatures* l, CFeatures* r, float64_t v,
		float64_t w, int32_t n, CDistance* dist, int32_t size) :
	CDistanceKernel(size,w,dist), order(v), degree(n)
{
	init();
	ASSERT(distance)
	SG_REF(distance);
	init(l,r);
}

CBesselKernel::~CBesselKernel()
{
	cleanup();
	SG_UNREF(distance);
}

void CBesselKernel::cleanup()
{
}

bool CBesselKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance)
	CDistanceKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CBesselKernel::init()
{
	SG_ADD(&order, "order", "Kernel order.", ParameterProperties::HYPER);
	SG_ADD(&degree, "degree", "Kernel degree.", ParameterProperties::HYPER);
}

float64_t CBesselKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return jn(order,dist/width)/CMath::pow(dist,-degree*order);
}
