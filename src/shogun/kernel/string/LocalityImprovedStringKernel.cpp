/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/LocalityImprovedStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

CLocalityImprovedStringKernel::CLocalityImprovedStringKernel()
: CStringKernel<char>()
{
	init();
}

CLocalityImprovedStringKernel::CLocalityImprovedStringKernel(
	int32_t size, int32_t l, int32_t id, int32_t od)
: CStringKernel<char>(size)
{
	init();

	length=l;
	inner_degree=id;
	outer_degree=od;

	SG_DEBUG("LIK with parms: l=%d, id=%d, od=%d created!\n", l, id, od)
}

CLocalityImprovedStringKernel::CLocalityImprovedStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t len,
	int32_t id, int32_t od)
: CStringKernel<char>()
{
	init();

	length=len;
	inner_degree=id;
	outer_degree=od;

	SG_DEBUG("LIK with parms: l=%d, id=%d, od=%d created!\n", len, id, od)

	init(l, r);
}

CLocalityImprovedStringKernel::~CLocalityImprovedStringKernel()
{
	cleanup();
}

bool CLocalityImprovedStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l,r);
	return init_normalizer();
}

float64_t CLocalityImprovedStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen && alen>0)

	int32_t i,t;
	float64_t* match=SG_MALLOC(float64_t, alen);

	// initialize match table 1 -> match;  0 -> no match
	for (i = 0; i<alen; i++)
		match[i] = (avec[i] == bvec[i])? 1 : 0;

	float64_t outer_sum = 0;

	for (t = 0; t<alen-length; t++)
	{
		float64_t sum = 0;
		for (i = 0; i<length && t+i+length+1<alen; i++)
			sum += (i+1)*match[t+i]+(length-i)*match[t+i+length+1];
		//add middle element + normalize with sum_i=0^2l+1 i = (2l+1)(l+1)
		float64_t inner_sum = (sum + (length+1)*match[t+length]) / ((2*length+1)*(length+1));
		inner_sum = pow(inner_sum, inner_degree + 1);
		outer_sum += inner_sum;
	}
	SG_FREE(match);

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return pow(outer_sum, outer_degree + 1);
}

void CLocalityImprovedStringKernel::init()
{
	set_normalizer(new CSqrtDiagKernelNormalizer());

	length = 0;
	inner_degree = 0;
	outer_degree = 0;

	SG_ADD(&length, "length", "Window Length.", ParameterProperties::HYPER);
	SG_ADD(&inner_degree, "inner_degree", "Inner degree.", ParameterProperties::HYPER);
	SG_ADD(&outer_degree, "outer_degree", "Outer degree.", ParameterProperties::HYPER);
}
