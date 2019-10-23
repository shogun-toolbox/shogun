/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Evan Shelhamer,
 *          Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/Chi2Kernel.h>
#include <shogun/features/Features.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

void
Chi2Kernel::init()
{
	SG_ADD(&width, "width", "Kernel width.", ParameterProperties::HYPER);
}

Chi2Kernel::Chi2Kernel()
: DotKernel(0), width(1)
{
	init();
}

Chi2Kernel::Chi2Kernel(int32_t size, float64_t w)
: DotKernel(size), width(w)
{
	init();
}

Chi2Kernel::Chi2Kernel(
	const std::shared_ptr<DenseFeatures<float64_t>>& l, const std::shared_ptr<DenseFeatures<float64_t>>& r, float64_t w, int32_t size)
: DotKernel(size), width(w)
{
	init();
	init(l,r);
}

Chi2Kernel::~Chi2Kernel()
{
	cleanup();
}

bool Chi2Kernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result=DotKernel::init(l,r);
	init_normalizer();
	return result;
}

float64_t Chi2Kernel::compute(int32_t idx_a, int32_t idx_b)
{
	require(width>0,
		"width not set to positive value. Current width {} ", width);
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
	{
		float64_t n=avec[i]-bvec[i];
		float64_t d=avec[i]+bvec[i];
		if (d!=0)
			result+=n*n/d;
	}

	result=exp(-result/width);

	(std::static_pointer_cast<DenseFeatures<float64_t>>(lhs))->free_feature_vector(avec, idx_a, afree);
	(std::static_pointer_cast<DenseFeatures<float64_t>>(rhs))->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

float64_t Chi2Kernel::get_width()
{
	return width;
}

std::shared_ptr<Chi2Kernel> Chi2Kernel::obtain_from_generic(const std::shared_ptr<Kernel>& kernel)
{
	if (kernel->get_kernel_type()!=K_CHI2)
	{
		error("Provided kernel is "
				"not of type CChi2Kernel!");
	}

	/* since an additional reference is returned */

	return std::static_pointer_cast<Chi2Kernel>(kernel);
}

void Chi2Kernel::set_width(int32_t w)
{
	require(w>0, "Parameter width should be > 0");
	width=w;
}
