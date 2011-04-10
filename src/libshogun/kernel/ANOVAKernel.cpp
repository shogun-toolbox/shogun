/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Andrew Tereskin
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <math.h>
#include "ANOVAKernel.h"
#include "lib/Mathematics.h"

using namespace shogun;

void CANOVAKernel::init()
{
	m_parameters->add(&cardinality, "cardinality", "Kernel cardinality.");
}

CANOVAKernel::CANOVAKernel(): CDotKernel(0), cardinality(1.0)
{
	init();
}

CANOVAKernel::CANOVAKernel(int32_t cache, int32_t d)
: CDotKernel(cache), cardinality(d)
{
	init();
}

CANOVAKernel::CANOVAKernel(
	CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r, int32_t d, int32_t cache)
  : CDotKernel(cache), cardinality(d)
{
	init();
	init(l, r);
}

CANOVAKernel::~CANOVAKernel()
{
	cleanup();
}

bool CANOVAKernel::init(CFeatures* l, CFeatures* r)
{
	bool result = CDotKernel::init(l,r);
	init_normalizer();
	return result;
}

float64_t CANOVAKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen, length;
	bool afree, bfree;

	float64_t* avec=
		((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen);
	length = alen;

	float64_t result1 = compute_recursive1(avec, bvec, length, cardinality);
	//float64_t result2 = compute_recursive2(avec, bvec, length, cardinality);
		
	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);
        
	return result1;
}

float64_t CANOVAKernel::compute_recursive1(float64_t* avec, float64_t* bvec, int32_t len, int32_t d) {
	float64_t** DP = new float64_t*[(d+1)];
	for(int32_t k=0; k < d+1; k++) {
		DP[k] = new float64_t[len+1];
	}
	
	for(int32_t j=0; j < len+1; j++) {
		DP[0][j] = 1.0;
	}

	for(int32_t k=1; k < d+1; k++) {
		DP[k][k-1] = 0;
		for(int32_t j=k; j < len+1; j++) {
			DP[k][j]=DP[k][j-1]+avec[j-1]*bvec[j-1]*DP[k-1][j-1];
		}
	}

	float64_t result=DP[d][len];
	
	for(int32_t k=0; k < d+1; k++) {
		delete[] DP[k];
	}
	delete[] DP;

	return result;
}

float64_t CANOVAKernel::compute_recursive2(float64_t* avec, float64_t* bvec, int32_t len, int32_t d) {
	float64_t* KD = new float64_t[d+1];
	float64_t* KS = new float64_t[d+1];
	float64_t* vec_pow = new float64_t[len];
	
	for(int32_t i=0; i < len; i++) {
		vec_pow[i] = 1;
	}
	for(int32_t k=1; k < d+1; k++){
		for(int32_t i=0; i < len; i++) {
			vec_pow[i] *= avec[i]*bvec[i];
			KS[k] += vec_pow[i];
		}
	}
	KD[0] = 1;
	for(int32_t k=1; k < d+1; k++){
		float64_t sum = 0;
		for(int32_t s=1; s < k+1; s++) {
			float64_t sign = 1;
			if (s & 1) sign = -1;
			sum += sign*KD[k-s]*KS[s];
		}
		KD[k] = sum / k;
	}
	float64_t result=KD[d];
	
	delete[] KD;
	delete[] KS;

	return result;
}
