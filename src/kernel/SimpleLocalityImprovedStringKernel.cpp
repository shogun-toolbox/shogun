/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/SimpleLocalityImprovedStringKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

CSimpleLocalityImprovedStringKernel::CSimpleLocalityImprovedStringKernel(
	INT size, INT l, INT id, INT od)
: CStringKernel<CHAR>(size), length(l), inner_degree(id), outer_degree(od),
	pyramid_weights(NULL)
{
}

CSimpleLocalityImprovedStringKernel::CSimpleLocalityImprovedStringKernel(
	CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r,
	INT len, INT id, INT od)
: CStringKernel<CHAR>(10), length(len), inner_degree(id), outer_degree(od),
	pyramid_weights(NULL)
{
	init(l, r);
}

CSimpleLocalityImprovedStringKernel::~CSimpleLocalityImprovedStringKernel()
{
	cleanup();
}

bool CSimpleLocalityImprovedStringKernel::init(CFeatures* l, CFeatures* r)
{
	bool result = CStringKernel<CHAR>::init(l,r);

	if (!result)
		return false;
	INT num_features = ((CStringFeatures<CHAR>*) l)->get_max_vector_length();
	pyramid_weights = new DREAL[num_features];
	ASSERT(pyramid_weights);
	SG_INFO("initializing pyramid weights: size=%ld length=%i\n",
		num_features, length);

	const INT PYRAL = 2 * length - 1; // total window length
	DREAL PYRAL_pot;
	INT DEGREE1_1  = (inner_degree & 0x1)==0;
	INT DEGREE1_1n = (inner_degree & ~0x1)!=0;
	INT DEGREE1_2  = (inner_degree & 0x2)!=0;
	INT DEGREE1_3  = (inner_degree & ~0x3)!=0;
	INT DEGREE1_4  = (inner_degree & 0x4)!=0;
	{
	DREAL PYRAL_ = PYRAL;
	PYRAL_pot = DEGREE1_1 ? 1.0 : PYRAL_;
	if (DEGREE1_1n)
	{
		PYRAL_ *= PYRAL_;
		if (DEGREE1_2)
			PYRAL_pot *= PYRAL_;
		if (DEGREE1_3)
		{
			PYRAL_ *= PYRAL_;
			if (DEGREE1_4)
				PYRAL_pot *= PYRAL_;
		}
	}
	}

	INT pyra_len  = num_features-PYRAL+1;
	INT pyra_len2 = (int) pyra_len/2;
	{
	INT j;
	for (j = 0; j < pyra_len; j++)
		pyramid_weights[j] = 4*((DREAL)((j < pyra_len2)? j+1 : pyra_len-j))/((DREAL)pyra_len);
	for (j = 0; j < pyra_len; j++)
		pyramid_weights[j] /= PYRAL_pot;
	}
	return true;
}

void CSimpleLocalityImprovedStringKernel::cleanup()
{
	delete[] pyramid_weights;
	pyramid_weights = NULL;

	CKernel::cleanup();
}

bool CSimpleLocalityImprovedStringKernel::load_init(FILE* src)
{
	return false;
}

bool CSimpleLocalityImprovedStringKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CSimpleLocalityImprovedStringKernel::dot_pyr (const CHAR* const x1,
	     const CHAR* const x2, const INT NOF_NTS, const INT NTWIDTH,
	     const INT DEGREE1, const INT DEGREE2, DREAL *pyra)
{
	const INT PYRAL = 2*NTWIDTH-1; // total window length
	INT pyra_len, pyra_len2;
	DREAL pot, PYRAL_pot;
	DREAL sum;
	INT DEGREE1_1 = (DEGREE1 & 0x1)==0;
	INT DEGREE1_1n = (DEGREE1 & ~0x1)!=0;
	INT DEGREE1_2 = (DEGREE1 & 0x2)!=0;
	INT DEGREE1_3 = (DEGREE1 & ~0x3)!=0;
	INT DEGREE1_4 = (DEGREE1 & 0x4)!=0;
	{
	DREAL PYRAL_ = PYRAL;
	PYRAL_pot = DEGREE1_1 ? 1.0 : PYRAL_;
	if (DEGREE1_1n)
	{
		PYRAL_ *= PYRAL_;
		if (DEGREE1_2) PYRAL_pot *= PYRAL_;
		if (DEGREE1_3)
		{
			PYRAL_ *= PYRAL_;
			if (DEGREE1_4) PYRAL_pot *= PYRAL_;
		}
	}
	}

	ASSERT((DEGREE1 & ~0x7) == 0);
	ASSERT((DEGREE2 & ~0x7) == 0);

	pyra_len = NOF_NTS-PYRAL+1;
	pyra_len2 = (int) pyra_len/2;
	{
	INT j;
	for (j = 0; j < pyra_len; j++)
		pyra[j] = 4*((DREAL)((j < pyra_len2) ? j+1 : pyra_len-j))/((DREAL)pyra_len);
	for (j = 0; j < pyra_len; j++)
		pyra[j] /= PYRAL_pot;
	}

	register INT conv;
	register INT i;
	register INT j;

	sum = 0.0;
	conv = 0;
	for (j = 0; j < PYRAL; j++)
		conv += (x1[j] == x2[j]) ? 1 : 0;

	for (i = 0; i < NOF_NTS-PYRAL+1; i++)
	{
		register DREAL pot2;
		if (i>0)
			conv += ((x1[i+PYRAL-1] == x2[i+PYRAL-1]) ? 1 : 0 ) - 
				((x1[i-1] == x2[i-1]) ? 1 : 0);
		{ /* potencing of conv -- double is faster*/
		register DREAL conv2 = conv;
		pot2 = (DEGREE1_1) ? 1.0 : conv2;
			if (DEGREE1_1n)
			{
				conv2 *= conv2;
				if (DEGREE1_2)
					pot2 *= conv2;
				if (DEGREE1_3 && DEGREE1_4)
					pot2 *= conv2*conv2;
			}
		}
		sum += pot2*pyra[i];
	}

	pot = ((DEGREE2 & 0x1) == 0) ? 1.0 : sum;
	if ((DEGREE2 & ~0x1) != 0)
	{
		sum *= sum;
		if ((DEGREE2 & 0x2) != 0)
			pot *= sum;
		if ((DEGREE2 & ~0x3) != 0)
		{
			sum *= sum;
			if ((DEGREE2 & 0x4) != 0)
				pot *= sum;
		}
	}
	return pot;
}

DREAL CSimpleLocalityImprovedStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	CHAR* avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length
	ASSERT(alen==blen);

	DREAL dpt;

	dpt = dot_pyr(avec, bvec, alen, length, inner_degree, outer_degree, pyramid_weights);
	dpt = dpt / pow((double)alen, (double)outer_degree);
	return (DREAL) dpt;
}
