/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/SplineKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CSplineKernel::CSplineKernel() : CDotKernel()
{
}

CSplineKernel::CSplineKernel(CDotFeatures* l, CDotFeatures* r) : CDotKernel()
{
	init(l,r);
}

CSplineKernel::~CSplineKernel()
{
	cleanup();
}

bool CSplineKernel::init(CFeatures* l, CFeatures* r)
{
	ASSERT(l->get_feature_type()==F_DREAL)
	ASSERT(l->get_feature_type()==r->get_feature_type())

	ASSERT(l->get_feature_class()==C_DENSE)
	ASSERT(l->get_feature_class()==r->get_feature_class())

	CDotKernel::init(l,r);
	return init_normalizer();
}

void CSplineKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CSplineKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec = ((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec = ((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen == blen)

	float64_t result = 0;
	for (int32_t i = 0; i < alen; i++) {
		const float64_t x = avec[i], y = bvec[i];
		const float64_t min = CMath::min(avec[i], bvec[i]);
		result += 1 + x*y + x*y*min - ((x+y)/2)*min*min + min*min*min/3;
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
