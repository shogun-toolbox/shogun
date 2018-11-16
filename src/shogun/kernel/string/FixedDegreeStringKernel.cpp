/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/kernel/string/FixedDegreeStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

void
CFixedDegreeStringKernel::init()
{
	SG_ADD(&degree, "degree", "The degree.", ParameterProperties::HYPER);
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel()
: CStringKernel<char>(0), degree(0)
{
	init();
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel(int32_t size, int32_t d)
: CStringKernel<char>(size), degree(d)
{
	init();
}

CFixedDegreeStringKernel::CFixedDegreeStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t d)
: CStringKernel<char>(10), degree(d)
{
	init();
	init(l, r);
}

CFixedDegreeStringKernel::~CFixedDegreeStringKernel()
{
	cleanup();
}

bool CFixedDegreeStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<char>::init(l, r);
	return init_normalizer();
}

void CFixedDegreeStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CFixedDegreeStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec = ((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec = ((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	// can only deal with strings of same length
	ASSERT(alen==blen)

	int64_t sum = 0;
	for (int32_t i = 0; i<alen-degree+1; i++)
	{
		bool match = true;

		for (int32_t j = i; j<i+degree && match; j++)
			match = avec[j]==bvec[j];
		if (match)
			sum++;
	}
	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);

	return sum;
}
