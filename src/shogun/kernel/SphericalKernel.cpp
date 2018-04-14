/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Bjoern Esser
 */

#include <shogun/kernel/SphericalKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CSphericalKernel::CSphericalKernel(): CKernel(0), distance(NULL)
{
	register_params();
	set_sigma(1.0);
}

CSphericalKernel::CSphericalKernel(int32_t size, float64_t sig, CDistance* dist)
: CKernel(size), distance(dist)
{
	ASSERT(distance)
	SG_REF(distance);
	register_params();
	set_sigma(sig);
}

CSphericalKernel::CSphericalKernel(
	CFeatures *l, CFeatures *r, float64_t sig, CDistance* dist)
: CKernel(10), distance(dist)
{
	ASSERT(distance)
	SG_REF(distance);
	register_params();
	set_sigma(sig);
	init(l, r);
}

CSphericalKernel::~CSphericalKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CSphericalKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(distance)
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CSphericalKernel::register_params()
{
	SG_ADD((CSGObject**) &distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
	SG_ADD(&sigma, "sigma", "Sigma kernel parameter.", MS_AVAILABLE);
}

float64_t CSphericalKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist=distance->distance(idx_a, idx_b);
	float64_t ds_ratio=dist/sigma;

	if (dist < sigma)
		return 1.0-1.5*(ds_ratio)+0.5*(ds_ratio*ds_ratio*ds_ratio);
	else
		return 0;
}
