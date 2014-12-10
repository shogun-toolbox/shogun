/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Andrew Tereskin
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/kernel/ANOVAKernel.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CANOVAKernel::CANOVAKernel(): CDotKernel(0), cardinality(1.0)
{
	register_params();
}

CANOVAKernel::CANOVAKernel(int32_t cache, int32_t d)
: CDotKernel(cache), cardinality(d)
{
	register_params();
}

CANOVAKernel::CANOVAKernel(
	CDenseFeatures<float64_t>* l, CDenseFeatures<float64_t>* r, int32_t d, int32_t cache)
  : CDotKernel(cache), cardinality(d)
{
	register_params();
	init(l, r);
}

CANOVAKernel::~CANOVAKernel()
{
	cleanup();
}

bool CANOVAKernel::init(CFeatures* l, CFeatures* r)
{
	cleanup();

	bool result = CDotKernel::init(l,r);

	init_normalizer();
	return result;
}

float64_t CANOVAKernel::compute(int32_t idx_a, int32_t idx_b)
{
	return compute_rec1(idx_a, idx_b);
}

float64_t CANOVAKernel::compute_rec1(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result = compute_recursive1(avec, bvec, alen);

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

float64_t CANOVAKernel::compute_rec2(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result = compute_recursive2(avec, bvec, alen);

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}

void CANOVAKernel::register_params()
{
	SG_ADD(&cardinality, "cardinality", "Kernel cardinality.", MS_AVAILABLE);
}


float64_t CANOVAKernel::compute_recursive1(float64_t* avec, float64_t* bvec, int32_t len)
{
	int32_t DP_len=(cardinality+1)*(len+1);
	float64_t* DP = SG_MALLOC(float64_t, DP_len);

	ASSERT(DP)
	int32_t d=cardinality;
	int32_t offs=cardinality+1;

	ASSERT(DP_len==(len+1)*offs)

	for (int32_t j=0; j < len+1; j++)
		DP[j] = 1.0;

	for (int32_t k=1; k < d+1; k++)
	{
		// TRAP d>len case
		if (k-1>=len)
			return 0.0;

		DP[k*offs+k-1] = 0;
		for (int32_t j=k; j < len+1; j++)
			DP[k*offs+j]=DP[k*offs+j-1]+avec[j-1]*bvec[j-1]*DP[(k-1)*offs+j-1];
	}

	float64_t result=DP[d*offs+len];

	SG_FREE(DP);

	return result;
}

float64_t CANOVAKernel::compute_recursive2(float64_t* avec, float64_t* bvec, int32_t len)
{
	float64_t* KD = SG_MALLOC(float64_t, cardinality+1);
	float64_t* KS = SG_MALLOC(float64_t, cardinality+1);
	float64_t* vec_pow = SG_MALLOC(float64_t, len);

	ASSERT(vec_pow)
	ASSERT(KS)
	ASSERT(KD)

	int32_t d=cardinality;
	for (int32_t i=0; i < len; i++)
		vec_pow[i] = 1;

	for (int32_t k=1; k < d+1; k++)
	{
		KS[k] = 0;
		for (int32_t i=0; i < len; i++)
		{
			vec_pow[i] *= avec[i]*bvec[i];
			KS[k] += vec_pow[i];
		}
	}

	KD[0] = 1;
	for (int32_t k=1; k < d+1; k++)
	{
		float64_t sum = 0;
		for (int32_t s=1; s < k+1; s++)
		{
			float64_t sign = 1.0;
			if (s % 2 == 0)
				sign = -1.0;

			sum += sign*KD[k-s]*KS[s];
		}

		KD[k] = sum / k;
	}
	float64_t result=KD[d];
	SG_FREE(vec_pow);
	SG_FREE(KS);
	SG_FREE(KD);

	return result;
}
