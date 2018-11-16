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

CGaussianShortRealKernel::CGaussianShortRealKernel()
: CDotKernel(0), width(0.0)
{
	register_params();
}

CGaussianShortRealKernel::CGaussianShortRealKernel(int32_t size, float64_t w)
: CDotKernel(size), width(w)
{
	register_params();
}

CGaussianShortRealKernel::CGaussianShortRealKernel(
	CDenseFeatures<float32_t>* l, CDenseFeatures<float32_t>* r, float64_t w, int32_t size)
: CDotKernel(size), width(w)
{
	init(l,r);
	register_params();
}

CGaussianShortRealKernel::~CGaussianShortRealKernel()
{
}

bool CGaussianShortRealKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	return init_normalizer();
}

float64_t CGaussianShortRealKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float32_t* avec=((CDenseFeatures<float32_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float32_t* bvec=((CDenseFeatures<float32_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
		result+=CMath::sq(avec[i]-bvec[i]);

	result=exp(-result/width);

	((CDenseFeatures<float32_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float32_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void CGaussianShortRealKernel::register_params()
{
	SG_ADD(&width, "width", "kernel width", ParameterProperties::HYPER);
}
