/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/GaussianShortRealKernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

GaussianShortRealKernel::GaussianShortRealKernel()
: DotKernel(0), width(0.0)
{
	register_params();
}

GaussianShortRealKernel::GaussianShortRealKernel(int32_t size, float64_t w)
: DotKernel(size), width(w)
{
	register_params();
}

GaussianShortRealKernel::GaussianShortRealKernel(
	const std::shared_ptr<DenseFeatures<float32_t>>& l, const std::shared_ptr<DenseFeatures<float32_t>>& r, float64_t w, int32_t size)
: DotKernel(size), width(w)
{
	init(l,r);
	register_params();
}

GaussianShortRealKernel::~GaussianShortRealKernel()
{
}

bool GaussianShortRealKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);
	return init_normalizer();
}

float64_t GaussianShortRealKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float32_t* avec=(std::static_pointer_cast<DenseFeatures<float32_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float32_t* bvec=(std::static_pointer_cast<DenseFeatures<float32_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
		result+=Math::sq(avec[i]-bvec[i]);

	result=exp(-result/width);

	(std::static_pointer_cast<DenseFeatures<float32_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float32_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void GaussianShortRealKernel::register_params()
{
	SG_ADD(&width, "width", "kernel width", ParameterProperties::HYPER);
}
