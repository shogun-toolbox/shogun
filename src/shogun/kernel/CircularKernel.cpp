/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/CircularKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CCircularKernel::CCircularKernel(): CKernel(0), distance(NULL)
{
	init();
	set_sigma(1.0);
}

CCircularKernel::CCircularKernel(int32_t size, float64_t sig, CDistance* dist)
: CKernel(size), distance(dist)
{
	ASSERT(distance)
	SG_REF(distance);

	set_sigma(sig);
	init();
}

CCircularKernel::CCircularKernel(
	CFeatures *l, CFeatures *r, float64_t sig, CDistance* dist)
: CKernel(10), distance(dist)
{
	ASSERT(distance)
	SG_REF(distance);
	set_sigma(sig);
	init();
	init(l, r);
}

CCircularKernel::~CCircularKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CCircularKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance)
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CCircularKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
}

void CCircularKernel::init()
{
	SG_ADD((CSGObject**) &distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
	SG_ADD(&sigma, "sigma", "Sigma kernel parameter.", MS_AVAILABLE);
}

float64_t CCircularKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=distance->distance(idx_a, idx_b);
	float64_t ds_ratio=dist/sigma;

	if (dist < sigma)
		return (2/M_PI)*acos(-ds_ratio) - (2/M_PI)*ds_ratio*sqrt(1-ds_ratio*ds_ratio);
	else
		return 0;
}
