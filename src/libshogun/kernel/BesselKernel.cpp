#include "BesselKernel.h"
#include "lib/Mathematics.h"
#include <math.h>

using namespace shogun;

CBesselKernel::CBesselKernel():CKernel(0),distance(NULL)
{
	init();
	set_sigma(1.0);
}

CBesselKernel::CBesselKernel(int32_t cache, float64_t sigma, CDistance* dist)
: CKernel(cache), distance(dist), sigma(sigma)
{
	init();
	ASSERT(distance);
	SG_REF(distance);
}

CBesselKernel::CBesselKernel(CFeatures *l, CFeatures *r, float64_t sigma, CDistance* dist)
: CKernel(10), distance(dist), sigma(sigma)
{
	init();
	ASSERT(distance);
	SG_REF(distance);
	init(l, r);
}

CBesselKernel::~CBesselKernel()
{
	cleanup();
	SG_UNREF(distance);
}


bool CBesselKernel::init(CFeatures* l, CFeatures* r)
{
	
	CKernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

void CBesselKernel::load_serializable_post(void) throw (ShogunException)
{
	CKernel::load_serializable_post();
}


void CBesselKernel::init()
{
        ASSERT(distance);
	m_parameters->add(&sigma, "sigma", "Sigma kernel parameter.");
	m_parameters->add((CSGObject**) &distance, "distance", "Distance to be used.");
}

float64_t CBesselKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return -j1(sigma*(dist*dist));
}

