/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Siddharth Kherada
 * Copyright (C) 2007-2011 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <lib/common.h>
#include <kernel/WaveletKernel.h>
#include <features/DenseFeatures.h>

using namespace shogun;

CWaveletKernel::CWaveletKernel() : CDotKernel(), Wdilation(0.0), Wtranslation(0.0)
{
	init();
}

CWaveletKernel::CWaveletKernel(int32_t size, float64_t a, float64_t c)
: CDotKernel(size), Wdilation(a), Wtranslation(c)
{
	init();
}

CWaveletKernel::CWaveletKernel(
	CDotFeatures* l, CDotFeatures* r, int32_t size, float64_t a, float64_t c)
: CDotKernel(size), Wdilation(a), Wtranslation(c)
{
	init();
	init(l,r);
}

CWaveletKernel::~CWaveletKernel()
{
	cleanup();
}

void CWaveletKernel::cleanup()
{
}

bool CWaveletKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l, r);
	return init_normalizer();
}

void CWaveletKernel::init()
{
	SG_ADD(&Wdilation, "Wdilation", "Dilation coefficient", MS_AVAILABLE);
	SG_ADD(&Wtranslation, "Wtranslaton", "Translation coefficient", MS_AVAILABLE);
}

float64_t CWaveletKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=
		((CDenseFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=
		((CDenseFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen)

	float64_t result=1;

	for (int32_t i=0; i<alen; i++)
	{
		if (Wtranslation !=0)
		{
			float64_t h1=(avec[i]-Wdilation)/Wtranslation;
			float64_t h2=(bvec[i]-Wdilation)/Wtranslation;
			float64_t res1=MotherWavelet(h1);
			float64_t res2=MotherWavelet(h2);
			result=result*res1*res2;
		}
	}

	((CDenseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CDenseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
