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
CChi2Kernel::init()
{
	SG_ADD(&width, "width", "Kernel width.", ParameterProperties::HYPER);
}

CChi2Kernel::CChi2Kernel()
: CDotKernel(0), width(1)
{
	init();
}

CChi2Kernel::CChi2Kernel(int32_t size, float64_t w)
: CDotKernel(size), width(w)
{
	init();
}

CChi2Kernel::CChi2Kernel(
	CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r, float64_t w, int32_t size)
: CDotKernel(size), width(w)
{
	init();
	init(l,r);
}

CChi2Kernel::~CChi2Kernel()
{
	cleanup();
}

bool CChi2Kernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CDotKernel::init(l,r);
	init_normalizer();
	return result;
}

float64_t CChi2Kernel::compute(int32_t idx_a, int32_t idx_b)
{
	REQUIRE(width>0,
		"width not set to positive value. Current width %d \n", width);
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
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

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

float64_t CChi2Kernel::get_width()
{
	return width;
}

CChi2Kernel* CChi2Kernel::obtain_from_generic(CKernel* kernel)
{
	if (kernel->get_kernel_type()!=K_CHI2)
	{
		SG_SERROR("Provided kernel is "
				"not of type CChi2Kernel!\n");
	}

	/* since an additional reference is returned */
	SG_REF(kernel);
	return (CChi2Kernel*)kernel;
}

void CChi2Kernel::set_width(int32_t w)
{
	REQUIRE(w>0, "Parameter width should be > 0");
	width=w;
}