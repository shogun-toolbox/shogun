/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Chiyuan Zhang
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/AttenuatedEuclideanDistance.h>
#include <shogun/features/Features.h>

using namespace shogun;

CAttenuatedEuclideanDistance::CAttenuatedEuclideanDistance() : CRealDistance()
{
	init();
}

CAttenuatedEuclideanDistance::CAttenuatedEuclideanDistance(CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r)
: CRealDistance()
{
	init();
	init(l, r);
}

CAttenuatedEuclideanDistance::~CAttenuatedEuclideanDistance()
{
	cleanup();
}

bool CAttenuatedEuclideanDistance::init(CFeatures* l, CFeatures* r)
{
	CRealDistance::init(l, r);
	return true;
}

void CAttenuatedEuclideanDistance::cleanup()
{
}

float64_t CAttenuatedEuclideanDistance::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;
	float64_t result=0;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->
		get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->
		get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	for (int32_t i=0; i<alen; i++)
		result+=(CMath::abs(avec[i])*CMath::abs(bvec[i]))*CMath::pow(avec[i] - bvec[i],2);

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	if (disable_sqrt)
		return result;

	return std::sqrt(result);
}

void CAttenuatedEuclideanDistance::init()
{
	disable_sqrt=false;

	SG_ADD(
	    &disable_sqrt, "disable_sqrt", "If sqrt shall not be applied.");
}
