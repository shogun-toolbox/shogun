/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/string/PolyMatchWordStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

CPolyMatchWordStringKernel::CPolyMatchWordStringKernel()
: CStringKernel<uint16_t>()
{
	init();
}

CPolyMatchWordStringKernel::CPolyMatchWordStringKernel(int32_t size, int32_t d, bool i)
: CStringKernel<uint16_t>(size)
{
	init();

	degree=d;
	inhomogene=i;
}

CPolyMatchWordStringKernel::CPolyMatchWordStringKernel(
	CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r, int32_t d, bool i)
: CStringKernel<uint16_t>()
{
	init();

	degree=d;
	inhomogene=i;

	init(l, r);
}

CPolyMatchWordStringKernel::~CPolyMatchWordStringKernel()
{
	cleanup();
}

bool CPolyMatchWordStringKernel::init(CFeatures* l, CFeatures* r)
{
	CStringKernel<uint16_t>::init(l,r);
	return init_normalizer();
}

void CPolyMatchWordStringKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CPolyMatchWordStringKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint16_t* avec=((CStringFeatures<uint16_t>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	uint16_t* bvec=((CStringFeatures<uint16_t>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen)

	int32_t sum=0;

	for (int32_t i=0; i<alen; i++)
		sum+= (avec[i]==bvec[i]) ? 1 : 0;

	if (inhomogene)
		sum+=1;

	float64_t result=sum;

	for (int32_t j=1; j<degree; j++)
		result*=sum;

	((CStringFeatures<uint16_t>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<uint16_t>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);
	return result;
}

void CPolyMatchWordStringKernel::init()
{
	degree=0;
	inhomogene=false;
	set_normalizer(new CSqrtDiagKernelNormalizer());

	SG_ADD(&degree, "degree", "Degree of poly-kernel.", ParameterProperties::HYPER);
	SG_ADD(&inhomogene, "inhomogene", "True for inhomogene poly-kernel.");
}
