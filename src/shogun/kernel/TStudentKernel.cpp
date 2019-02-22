/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Viktor Gal
 */

#include <shogun/kernel/TStudentKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

void CTStudentKernel::init()
{
	SG_ADD(&degree, "degree", "Kernel degree.", ParameterProperties::HYPER);
	SG_ADD(&distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}

CTStudentKernel::CTStudentKernel(): CKernel(0), distance(NULL), degree(1.0)
{
	init();
}

CTStudentKernel::CTStudentKernel(int32_t cache, float64_t d, CDistance* dist)
: CKernel(cache), distance(dist), degree(d)
{
	init();
	ASSERT(distance)
	SG_REF(distance);
}

CTStudentKernel::CTStudentKernel(CFeatures *l, CFeatures *r, float64_t d, CDistance* dist)
: CKernel(10), distance(dist), degree(d)
{
	init();
	ASSERT(distance)
	SG_REF(distance);
	init(l, r);
}

CTStudentKernel::~CTStudentKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CTStudentKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance)
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

float64_t CTStudentKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return 1.0/(1.0+CMath::pow(dist, this->degree));
}
