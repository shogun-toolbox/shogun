/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/MatchWordStringKernel.h>
#include <shogun/kernel/normalizer/AvgDiagKernelNormalizer.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

MatchWordStringKernel::MatchWordStringKernel() : StringKernel<uint16_t>()
{
	init();
}

MatchWordStringKernel::MatchWordStringKernel(int32_t size, int32_t d)
: StringKernel<uint16_t>(size)
{
	init();
	degree=d;
}

MatchWordStringKernel::MatchWordStringKernel(
		const std::shared_ptr<StringFeatures<uint16_t>>& l, const std::shared_ptr<StringFeatures<uint16_t>>& r, int32_t d)
: StringKernel<uint16_t>()
{
	init();
	degree=d;
	init(l, r);
}

MatchWordStringKernel::~MatchWordStringKernel()
{
	cleanup();
}

bool MatchWordStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	StringKernel<uint16_t>::init(l, r);
	return init_normalizer();
}

float64_t MatchWordStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint16_t* avec=std::static_pointer_cast<StringFeatures<uint16_t>>(lhs)->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=std::static_pointer_cast<StringFeatures<uint16_t>>(rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen)

	float64_t sum=0;
	for (int32_t i=0; i<alen; i++)
		sum+= (avec[i]==bvec[i]) ? 1 : 0;

	std::static_pointer_cast<StringFeatures<uint16_t>>(lhs)->free_feature_vector(avec, idx_a, free_avec);
	std::static_pointer_cast<StringFeatures<uint16_t>>(rhs)->free_feature_vector(bvec, idx_b, free_bvec);

	return Math::pow(sum, degree);
}

void MatchWordStringKernel::init()
{
	degree=0;
	set_normalizer(std::make_shared<AvgDiagKernelNormalizer>());
	SG_ADD(&degree, "degree", "Degree of poly kernel", ParameterProperties::HYPER);
}
