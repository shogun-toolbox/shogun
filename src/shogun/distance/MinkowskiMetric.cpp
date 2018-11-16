/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#include <shogun/base/Parameter.h>

#include <shogun/distance/MinkowskiMetric.h>
#include <shogun/features/Features.h>

using namespace shogun;

CMinkowskiMetric::CMinkowskiMetric() : CDenseDistance<float64_t>()
{
	init();
}

CMinkowskiMetric::CMinkowskiMetric(float64_t k_)
: CDenseDistance<float64_t>()
{
	init();
	k=k_;
}

CMinkowskiMetric::CMinkowskiMetric(
	CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r, float64_t k_)
: CDenseDistance<float64_t>()
{
	init();
	k=k_;
	init(l, r);
}

CMinkowskiMetric::~CMinkowskiMetric()
{
	cleanup();
}

bool CMinkowskiMetric::init(CFeatures* l, CFeatures* r)
{
	return CDenseDistance<float64_t>::init(l,r);
}

void CMinkowskiMetric::cleanup()
{
}

float64_t CMinkowskiMetric::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

	ASSERT(avec)
	ASSERT(bvec)
	ASSERT(alen==blen)

	float64_t absTmp = 0;
	float64_t result=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			absTmp=fabs(avec[i]-bvec[i]);
			result+=pow(absTmp,k);
		}

	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return pow(result,1/k);
}

void CMinkowskiMetric::init()
{
	k = 2.0;
	SG_ADD(&k, "k", "L_k norm.", ParameterProperties::HYPER);
}
