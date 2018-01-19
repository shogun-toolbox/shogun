/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/InverseMultiQuadricKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CInverseMultiQuadricKernel::CInverseMultiQuadricKernel(): CKernel(0), distance(NULL), coef(0.0001)
{
	init();
}

CInverseMultiQuadricKernel::CInverseMultiQuadricKernel(int32_t cache, float64_t coefficient, CDistance* dist)
: CKernel(cache), distance(dist), coef(coefficient)
{
	SG_REF(distance);
	init();
}

CInverseMultiQuadricKernel::CInverseMultiQuadricKernel(CFeatures *l, CFeatures *r, float64_t coefficient, CDistance* dist)
: CKernel(10), distance(dist), coef(coefficient)
{
	SG_REF(distance);
	init();
	init(l, r);
}

CInverseMultiQuadricKernel::~CInverseMultiQuadricKernel()
{
	cleanup();
	SG_UNREF(distance);
}

bool CInverseMultiQuadricKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CInverseMultiQuadricKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
}

void CInverseMultiQuadricKernel::init()
{
	SG_ADD(&coef, "coef", "Kernel Coefficient.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}

float64_t CInverseMultiQuadricKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return 1/sqrt(dist*dist + coef*coef);
}
