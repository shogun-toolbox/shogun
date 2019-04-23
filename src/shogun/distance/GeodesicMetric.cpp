/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/GeodesicMetric.h>
#include <shogun/features/Features.h>

using namespace shogun;

GeodesicMetric::GeodesicMetric() : DenseDistance<float64_t>()
{
}

GeodesicMetric::GeodesicMetric(std::shared_ptr<DenseFeatures<float64_t>> l, std::shared_ptr<DenseFeatures<float64_t>> r)
: DenseDistance<float64_t>()
{
	init(l, r);
}

GeodesicMetric::~GeodesicMetric()
{
	cleanup();
}

bool GeodesicMetric::init(std::shared_ptr<Features> l, std::shared_ptr<Features> r)
{
	bool result=DenseDistance<float64_t>::init(l,r);

	return result;
}

void GeodesicMetric::cleanup()
{
}

float64_t GeodesicMetric::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		lhs->as<DenseFeatures<float64_t>>()->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		rhs->as<DenseFeatures<float64_t>>()->get_feature_vector(idx_b, blen, bfree);

	ASSERT(alen==blen)

	float64_t s=0;
	float64_t d=0;
	float64_t nx=0;
	float64_t ny=0;
	{
		for (int32_t i=0; i<alen; i++)
		{
			d+=avec[i]*bvec[i];
			nx+=avec[i]*avec[i];
			ny+=bvec[i]*bvec[i];
			s+=avec[i]+bvec[i];
		}
	}

	lhs->as<DenseFeatures<float64_t>>()->free_feature_vector(avec, idx_a, afree);
	rhs->as<DenseFeatures<float64_t>>()->free_feature_vector(bvec, idx_b, bfree);


	// trap division by zero
	if(s==0 || nx==0 || ny==0)
		return 0;

	d /= std::sqrt(nx * ny);

	// can only happen due to numerical problems
	if (Math::abs(d)>1.0)
		d=Math::sign(d);

	return acos(d);
}
