/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Viktor Gal
 */

#include <shogun/kernel/TStudentKernel.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

void TStudentKernel::init()
{
	SG_ADD(&degree, "degree", "Kernel degree.", ParameterProperties::HYPER);
	SG_ADD(&distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}

TStudentKernel::TStudentKernel(): Kernel(0), distance(NULL), degree(1.0)
{
	init();
}

TStudentKernel::TStudentKernel(int32_t cache, float64_t d, std::shared_ptr<Distance> dist)
: Kernel(cache), distance(std::move(dist)), degree(d)
{
	init();
	ASSERT(distance)
	
}

TStudentKernel::TStudentKernel(std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t d, std::shared_ptr<Distance> dist)
: Kernel(10), distance(std::move(dist)), degree(d)
{
	init();
	ASSERT(distance)
	
	init(std::move(l), std::move(r));
}

TStudentKernel::~TStudentKernel()
{
	cleanup();
	
}

bool TStudentKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(distance)
	Kernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

float64_t TStudentKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t dist = distance->distance(idx_a, idx_b);
	return 1.0/(1.0+Math::pow(dist, this->degree));
}
