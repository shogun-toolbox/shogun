/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Evan Shelhamer
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Features.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/kernel/LinearKernel.h>

using namespace shogun;

LinearKernel::LinearKernel()
: DotKernel(0)
{
	properties |= KP_LINADD;
}

LinearKernel::LinearKernel(std::shared_ptr<DotFeatures> l, std::shared_ptr<DotFeatures> r)
: DotKernel(0)
{
	properties |= KP_LINADD;
	init(l,r);
}

LinearKernel::~LinearKernel()
{
	cleanup();
}

bool LinearKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);

	return init_normalizer();
}

void LinearKernel::cleanup()
{
	delete_optimization();

	Kernel::cleanup();
}

void LinearKernel::add_to_normal(int32_t idx, float64_t weight)
{
	lhs->as<DotFeatures>()->add_to_dense_vec(
		normalizer->normalize_lhs(weight, idx), idx, normal.vector, normal.size());
	set_is_initialized(true);
}

bool LinearKernel::init_optimization(
	int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas)
{
	clear_normal();

	for (int32_t i=0; i<num_suppvec; i++)
		add_to_normal(sv_idx[i], alphas[i]);

	set_is_initialized(true);
	return true;
}

bool LinearKernel::init_optimization(std::shared_ptr<KernelMachine> km)
{
	clear_normal();

	int32_t num_suppvec=km->get_num_support_vectors();

	for (int32_t i=0; i<num_suppvec; i++)
		add_to_normal(km->get_support_vector(i), km->get_alpha(i));

	set_is_initialized(true);
	return true;
}

bool LinearKernel::delete_optimization()
{
	normal = SGVector<float64_t>();
	set_is_initialized(false);

	return true;
}

float64_t LinearKernel::compute_optimized(int32_t idx)
{
	ASSERT(get_is_initialized())
	float64_t result = rhs->as<DotFeatures>()->dot(idx, normal);
	return normalizer->normalize_rhs(result, idx);
}
