/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/PolyMatchStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

PolyMatchStringKernel::PolyMatchStringKernel()
: StringKernel<char>()
{
	init();
}

PolyMatchStringKernel::PolyMatchStringKernel(int32_t size, int32_t d, bool i)
: StringKernel<char>(size)
{
	init();

	degree=d;
	inhomogene=i;
}

PolyMatchStringKernel::PolyMatchStringKernel(
	const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r, int32_t d, bool i)
: StringKernel<char>(10)
{
	init();

	degree=d;
	inhomogene=i;

	init(l, r);
}

PolyMatchStringKernel::~PolyMatchStringKernel()
{
	cleanup();
}

bool PolyMatchStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	StringKernel<char>::init(l, r);
	return init_normalizer();
}

void PolyMatchStringKernel::cleanup()
{
	Kernel::cleanup();
}

float64_t PolyMatchStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t i, alen, blen, sum;
	bool free_avec, free_bvec;

	char* avec = std::static_pointer_cast<StringFeatures<char>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = std::static_pointer_cast<StringFeatures<char>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen)
	for (i = 0, sum = inhomogene; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			sum++;
	}
	float64_t result = ((float64_t) sum);

	if (rescaling)
		result/=alen;

	std::static_pointer_cast<StringFeatures<char>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<char>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return Math::pow(result , degree);
}

void PolyMatchStringKernel::init()
{
	degree=0;
	inhomogene=false;
	rescaling=false;
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());

	SG_ADD(&degree, "degree", "Degree of poly-kernel.", ParameterProperties::HYPER);
	SG_ADD(&inhomogene, "inhomogene", "True for inhomogene poly-kernel.");
	SG_ADD(&rescaling, "rescaling",
	    "True to rescale kernel with string length.", ParameterProperties::HYPER);
}
