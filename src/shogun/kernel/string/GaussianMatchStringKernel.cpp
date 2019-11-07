/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/GaussianMatchStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

GaussianMatchStringKernel::GaussianMatchStringKernel()
: StringKernel<char>(0), width(0.0)
{
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	register_params();
}

GaussianMatchStringKernel::GaussianMatchStringKernel(int32_t size, float64_t w)
: StringKernel<char>(size), width(w)
{
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	register_params();
}

GaussianMatchStringKernel::GaussianMatchStringKernel(
	const std::shared_ptr<StringFeatures<char>>& l, const std::shared_ptr<StringFeatures<char>>& r, float64_t w)
: StringKernel<char>(10), width(w)
{
	set_normalizer(std::make_shared<SqrtDiagKernelNormalizer>());
	init(l, r);
	register_params();
}

GaussianMatchStringKernel::~GaussianMatchStringKernel()
{
	cleanup();
}

bool GaussianMatchStringKernel::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	StringKernel<char>::init(l, r);
	return init_normalizer();
}

void GaussianMatchStringKernel::cleanup()
{
	Kernel::cleanup();
}

float64_t GaussianMatchStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t i, alen, blen ;
	bool free_avec, free_bvec;

	char* avec = lhs->as<StringFeatures<char>>()->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = rhs->as<StringFeatures<char>>()->get_feature_vector(idx_b, blen, free_bvec);

	float64_t result=0;

	ASSERT(alen==blen)

	for (i = 0;  i<alen; i++)
		result+=(avec[i]==bvec[i]) ? 0:4;

	result=exp(-result/width);


	lhs->as<StringFeatures<char>>()->free_feature_vector(avec, idx_a, free_avec);
	rhs->as<StringFeatures<char>>()->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

void GaussianMatchStringKernel::register_params()
{
	SG_ADD(&width, "width", "kernel width", ParameterProperties::HYPER);
}
