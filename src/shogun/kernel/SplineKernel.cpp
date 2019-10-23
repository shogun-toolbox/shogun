/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/SplineKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

SplineKernel::SplineKernel() : DotKernel()
{
}

SplineKernel::SplineKernel(const std::shared_ptr<DotFeatures>& l, const std::shared_ptr<DotFeatures>& r) : DotKernel()
{
	init(l,r);
}

SplineKernel::~SplineKernel()
{
	cleanup();
}

bool SplineKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	ASSERT(l->get_feature_type()==F_DREAL)
	ASSERT(l->get_feature_type()==r->get_feature_type())

	ASSERT(l->get_feature_class()==C_DENSE)
	ASSERT(l->get_feature_class()==r->get_feature_class())

	DotKernel::init(l,r);
	return init_normalizer();
}

void SplineKernel::cleanup()
{
	Kernel::cleanup();
}

float64_t SplineKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec = (std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec = (std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen == blen)

	float64_t result = 0;
	for (int32_t i = 0; i < alen; i++) {
		const float64_t x = avec[i], y = bvec[i];
		const float64_t min = Math::min(avec[i], bvec[i]);
		result += 1 + x*y + x*y*min - ((x+y)/2)*min*min + min*min*min/3;
	}

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
