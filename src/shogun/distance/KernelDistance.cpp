/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/KernelDistance.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

KernelDistance::KernelDistance() : Distance()
{
	init();
}

KernelDistance::KernelDistance(float64_t w, std::shared_ptr<Kernel> k)
: Distance()
{
	init();

	kernel=k;
	width=w;
	ASSERT(kernel)
	
}

KernelDistance::KernelDistance(
	std::shared_ptr<Features >l, std::shared_ptr<Features >r, float64_t w , std::shared_ptr<Kernel> k)
: Distance()
{
	init();

	kernel=k;
	width=w;
	ASSERT(kernel)
	

	init(l, r);
}

KernelDistance::~KernelDistance()
{
	// important to have the cleanup of Distance first, it calls get_name which
	// uses the distance
	cleanup();
	
}

bool KernelDistance::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(kernel)
	kernel->init(l,r);
	return Distance::init(l,r);
}

float64_t KernelDistance::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=kernel->kernel(idx_a, idx_b);
	return exp(-result/width);
}

void KernelDistance::init()
{
	kernel = NULL;
	width = 0.0;

	SG_ADD(&width, "width", "Width of RBF Kernel", ParameterProperties::HYPER);
	SG_ADD(&kernel, "kernel", "Kernel.");
}
