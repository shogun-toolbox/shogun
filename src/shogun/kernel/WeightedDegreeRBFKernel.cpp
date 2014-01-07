/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <kernel/WeightedDegreeRBFKernel.h>
#include <features/Features.h>
#include <io/SGIO.h>

using namespace shogun;

CWeightedDegreeRBFKernel::CWeightedDegreeRBFKernel()
: CDotKernel(), width(1), degree(1), weights(0)
{
}


CWeightedDegreeRBFKernel::CWeightedDegreeRBFKernel(int32_t size, float64_t w, int32_t d, int32_t nof_prop)
: CDotKernel(size), width(w), degree(d), nof_properties(nof_prop), weights(0)
{
	init_wd_weights();
}

CWeightedDegreeRBFKernel::CWeightedDegreeRBFKernel(
	CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r, float64_t w, int32_t d, int32_t nof_prop, int32_t size)
: CDotKernel(size), width(w), degree(d), nof_properties(nof_prop), weights(0)
{
	init_wd_weights();
	init(l,r);
}

CWeightedDegreeRBFKernel::~CWeightedDegreeRBFKernel()
{
	SG_FREE(weights);
	weights=NULL;
}

bool CWeightedDegreeRBFKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	SG_DEBUG("Initialized WeightedDegreeRBFKernel (%p).\n", this)
	return init_normalizer();
}

bool CWeightedDegreeRBFKernel::init_wd_weights()
{
	ASSERT(degree>0)

	if (weights!=0)	SG_FREE(weights);
	weights=SG_MALLOC(float64_t, degree);
	if (weights)
	{
		int32_t i;
		float64_t sum=0;
		for (i=0; i<degree; i++)
		{
			weights[i]=degree-i;
			sum+=weights[i];
		}
		for (i=0; i<degree; i++)
			weights[i]/=sum;

		SG_DEBUG("Initialized weights for WeightedDegreeRBFKernel (%p).\n", this)
		return true;
	}
	else
		return false;
}


float64_t CWeightedDegreeRBFKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)
	ASSERT(alen%nof_properties == 0)

	float64_t result=0;

	for (int32_t i=0; i<alen; i+=nof_properties)
	{
		float64_t resulti = 0.0;

		for (int32_t d=0; (i+(d*nof_properties)<alen) && (d<degree); d++)
		{
			float64_t resultid = 0.0;
			int32_t limit = (d + 1 ) * nof_properties;
			for (int32_t k=0; k < limit; k++)
			{
				resultid+=CMath::sq(avec[i+k]-bvec[i+k]);
			}

			resulti += weights[d] * exp(-resultid/width);
		}

		result+=resulti ;
	}

	return result;
}
