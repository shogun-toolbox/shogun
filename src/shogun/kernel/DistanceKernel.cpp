/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Bjoern Esser, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/DistanceKernel.h>
#include <shogun/features/DenseFeatures.h>

#include <utility>

using namespace shogun;

DistanceKernel::DistanceKernel()
: Kernel(0), distance(NULL), width(0.0)
{
	register_params();
}

DistanceKernel::DistanceKernel(int32_t size, float64_t w, std::shared_ptr<Distance> d)
: Kernel(size), distance(std::move(d))
{
	ASSERT(distance)
	set_width(w);
	
	register_params();
}

DistanceKernel::DistanceKernel(
	std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t w , std::shared_ptr<Distance> d)
: Kernel(10), distance(std::move(d))
{
	set_width(w);
	ASSERT(distance)
	
	init(std::move(l), std::move(r));
	register_params();
}

DistanceKernel::~DistanceKernel()
{
	// important to have the cleanup of Kernel first, it calls get_name which
	// uses the distance
	cleanup();
	
}

bool DistanceKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(distance)
	Kernel::init(l,r);
	distance->init(l,r);
	return init_normalizer();
}

float64_t DistanceKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=distance->distance(idx_a, idx_b);
	return exp(-result/width);
}

void DistanceKernel::register_params()
{
	SG_ADD(&width, "width", "Kernel width.", ParameterProperties::HYPER);
	SG_ADD(&distance, "distance", "Distance to be used.",
	    ParameterProperties::HYPER);
}
