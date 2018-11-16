/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer
 */

#include <shogun/kernel/WaveKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CWaveKernel::CWaveKernel(): CKernel(0), m_distance(NULL), m_theta(1.0)
{
	init();
}

CWaveKernel::CWaveKernel(int32_t cache, float64_t theta, CDistance* dist)
: CKernel(cache), m_distance(dist), m_theta(theta)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
}

CWaveKernel::CWaveKernel(CFeatures *l, CFeatures *r, float64_t theta, CDistance* dist)
: CKernel(10), m_distance(dist), m_theta(theta)
{
	init();
	ASSERT(m_distance)
	SG_REF(m_distance);
	init(l, r);
}

CWaveKernel::~CWaveKernel()
{
	cleanup();
	SG_UNREF(m_distance);
}

bool CWaveKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(m_distance)
	CKernel::init(l,r);
	m_distance->init(l,r);
	return init_normalizer();
}

void CWaveKernel::init()
{
	SG_ADD(&m_theta, "theta", "Theta kernel parameter.", ParameterProperties::HYPER);
	SG_ADD((CSGObject**) &m_distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}

float64_t CWaveKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = m_distance->distance(idx_a, idx_b);

	if (dist==0.0)
		return 1.0;

	return (m_theta/dist)*sin(dist/m_theta);
}
