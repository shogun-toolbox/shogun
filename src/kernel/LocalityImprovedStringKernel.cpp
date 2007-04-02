/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/LocalityImprovedStringKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

CLocalityImprovedStringKernel::CLocalityImprovedStringKernel(LONG size, INT l, INT d1, INT d2)
  : CStringKernel<CHAR>(size),length(l),inner_degree(d1),outer_degree(d2),match(NULL)
{
	SG_INFO( "LIK with parms: l=%d, d1=%d, d2=%d created!\n", l, d1, d2);
}

CLocalityImprovedStringKernel::~CLocalityImprovedStringKernel()
{
	cleanup();
}

bool CLocalityImprovedStringKernel::init(CFeatures* l, CFeatures* r)
{
	bool result = CStringKernel<CHAR>::init(l,r);

	if (!result)
		return false;
	match = new CHAR[((CStringFeatures<CHAR>*) l)->get_max_vector_length()];
	return match? true : false;
}

void CLocalityImprovedStringKernel::cleanup()
{
	delete[] match;
	match = NULL;
}

bool CLocalityImprovedStringKernel::load_init(FILE* src)
{
	return false;
}

bool CLocalityImprovedStringKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CLocalityImprovedStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	CHAR* avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length
	ASSERT(alen == blen);

	INT i,j,t;

	// initialize match table 1 -> match;  0 -> no match
	for (i = 0; i<alen; i++)
		match[i] = (avec[i] == bvec[i])? 1 : 0;

	DREAL outer_sum = 0;

	for (t = 0; t<alen-length; t++)
	{
		INT sum = 0;
		for (i = 0; i<length; i++)
			sum += (i+1)*match[t+i]+(length-i)*match[t+i+length+1];
		//add middle element + normalize with sum_i=0^2l+1 i = (2l+1)(l+1)
		DREAL inner_sum = ((DREAL) sum + (length+1)*match[t+length]) / ((2*length+1)*(length+1));
		inner_sum = pow(inner_sum, inner_degree + 1);
		outer_sum += inner_sum;
	}
	return pow(outer_sum, outer_degree + 1);
}
