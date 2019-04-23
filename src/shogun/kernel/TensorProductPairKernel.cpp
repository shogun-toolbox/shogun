/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evan Shelhamer, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/TensorProductPairKernel.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

TensorProductPairKernel::TensorProductPairKernel()
: DotKernel(0), subkernel(NULL)
{
	register_params();
}

TensorProductPairKernel::TensorProductPairKernel(int32_t size, std::shared_ptr<Kernel> s)
: DotKernel(size), subkernel(s)
{

	register_params();
}

TensorProductPairKernel::TensorProductPairKernel(std::shared_ptr<DenseFeatures<int32_t>> l, std::shared_ptr<DenseFeatures<int32_t>> r, std::shared_ptr<Kernel> s)
: DotKernel(10), subkernel(s)
{

	init(l, r);
	register_params();
}

TensorProductPairKernel::~TensorProductPairKernel()
{

	cleanup();
}

bool TensorProductPairKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	DotKernel::init(l, r);
	init_normalizer();
	return true;
}

float64_t TensorProductPairKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	int32_t* avec=(std::static_pointer_cast<DenseFeatures<int32_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	int32_t* bvec=(std::static_pointer_cast<DenseFeatures<int32_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==2)
	ASSERT(blen==2)

	auto k=subkernel;
	ASSERT(k && k->has_features())

	int32_t a=avec[0];
	int32_t b=avec[1];
	int32_t c=bvec[0];
	int32_t d=bvec[1];

	float64_t result = k->kernel(a,c)*k->kernel(b,d) + k->kernel(a,d)*k->kernel(b,c);

	(std::static_pointer_cast<DenseFeatures<int32_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<int32_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void TensorProductPairKernel::register_params()
{
	SG_ADD((std::shared_ptr<SGObject>*)&subkernel, "subkernel", "the subkernel", ParameterProperties::HYPER);
}
