/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/kernel/ExponentialKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CExponentialKernel::CExponentialKernel()
	: CDotKernel(), m_distance(NULL), m_width(1)
{
	init();
}

CExponentialKernel::CExponentialKernel(
	CDotFeatures* l, CDotFeatures* r, float64_t width, CDistance* distance, int32_t size)
: CDotKernel(size), m_distance(distance), m_width(width)
{
	init();
	ASSERT(distance)
	SG_REF(distance);
	init(l,r);
}

CExponentialKernel::~CExponentialKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

void CExponentialKernel::cleanup()
{
	CKernel::cleanup();
}

bool CExponentialKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(m_distance)
	CDotKernel::init(l, r);
	m_distance->init(l, r);
	return init_normalizer();
}

float64_t CExponentialKernel::compute(int32_t idx_a, int32_t idx_b)
{
	ASSERT(m_distance)
	float64_t dist=m_distance->distance(idx_a, idx_b);
	return exp(-dist/m_width);
}

void CExponentialKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
}


void CExponentialKernel::init()
{
	SG_ADD(&m_width, "width", "Kernel width.", MS_AVAILABLE);
	SG_ADD((CSGObject**) &m_distance, "distance", "Distance to be used.",
	    MS_AVAILABLE);
}
