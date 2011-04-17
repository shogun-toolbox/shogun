/*
 * BesselKernel.cpp
 *
 *  Created on: Apr 17, 2011
 *      Author: ziyuan
 */

#include "BesselKernel.h"
#include "lib/Mathematics.h"
#include <boost/math/special_functions/bessel.hpp>

using namespace boost::math;
using namespace shogun;

CBesselKernel::CBesselKernel():CDistanceKernel(),order(0.0),degree(0)
{
	init();
}

CBesselKernel::CBesselKernel(int32_t size, float64_t v, float64_t w, int32_t n,
		CDistance* dist):CDistanceKernel(size,w,dist), order(v), degree(n)
{
	ASSERT(distance);
	SG_REF(distance);
	init();
}

CBesselKernel::CBesselKernel(CFeatures* l, CFeatures* r, int32_t size, float64_t v, float64_t w, int32_t n,
		CDistance* dist):CDistanceKernel(size,w,dist), order(v), degree(n)
{
	init();
	ASSERT(distance);
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
	ASSERT(distance);
	CDistanceKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CBesselKernel::init()
{
	m_parameters->add(&order, "order", "Kernel order.");
	m_parameters->add(&width, "width", "Kernel width.");
	m_parameters->add(&degree, "degree", "Kernel degree.");
	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
}

float64_t CBesselKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return cyl_bessel_j(order,dist/width)/pow(dist,-degree*order);
}
