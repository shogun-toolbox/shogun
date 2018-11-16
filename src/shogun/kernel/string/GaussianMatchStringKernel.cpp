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

CGaussianMatchStringKernel::CGaussianMatchStringKernel()
: CStringKernel<char>(0), width(0.0)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CGaussianMatchStringKernel::CGaussianMatchStringKernel(int32_t size, float64_t w)
: CStringKernel<char>(size), width(w)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	register_params();
}

CGaussianMatchStringKernel::CGaussianMatchStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, float64_t w)
: CStringKernel<char>(10), width(w)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l, r);
	register_params();
}

CGaussianMatchStringKernel::~CGaussianMatchStringKernel()
{
	cleanup();
}

bool CGaussianMatchStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CGaussianMatchStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CGaussianMatchStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t i, alen, blen ;
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	float64_t result=0;

	ASSERT(alen==blen)

	for (i = 0;  i<alen; i++)
		result+=(avec[i]==bvec[i]) ? 0:4;

	result=exp(-result/width);


	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

void CGaussianMatchStringKernel::register_params()
{
	SG_ADD(&width, "width", "kernel width", ParameterProperties::HYPER);
}
