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

ExponentialKernel::ExponentialKernel()
	: DotKernel(), m_distance(NULL), m_width(1)
{
	init();
}

ExponentialKernel::ExponentialKernel(
	std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r, float64_t width, std::shared_ptr<Distance> distance, int32_t size)
: DotKernel(size), m_distance(distance), m_width(width)
{
	init();
	ASSERT(distance)
	
	init(l,r);
}

ExponentialKernel::~ExponentialKernel()
{
	cleanup();
	
}

void ExponentialKernel::cleanup()
{
	Kernel::cleanup();
}

bool ExponentialKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(m_distance)
	DotKernel::init(l, r);
	m_distance->init(l, r);
	return init_normalizer();
}

float64_t ExponentialKernel::compute(int32_t idx_a, int32_t idx_b)
{
	ASSERT(m_distance)
	float64_t dist=m_distance->distance(idx_a, idx_b);
	return exp(-dist/m_width);
}

void ExponentialKernel::load_serializable_post() noexcept(false)
{
	Kernel::load_serializable_post();
}


void ExponentialKernel::init()
{
	SG_ADD(&m_width, "width", "Kernel width.", ParameterProperties::HYPER);
	SG_ADD(&m_distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}
