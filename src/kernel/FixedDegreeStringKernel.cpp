/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/FixedDegreeStringKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

CFixedDegreeStringKernel::CFixedDegreeStringKernel(LONG size, INT d)
  : CStringKernel<CHAR>(size),degree(d), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false)
{
}

CFixedDegreeStringKernel::~CFixedDegreeStringKernel()
{
	cleanup();
}

bool CFixedDegreeStringKernel::init(CFeatures* l, CFeatures* r)
{
	bool result = CStringKernel<CHAR>::init(l, r);
	initialized = false;
	INT i;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = NULL;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = new DREAL[lhs->get_num_vectors()];
	ASSERT(sqrtdiag_lhs);

	if (l==r)
		sqrtdiag_rhs = sqrtdiag_lhs;
	else
	{
		sqrtdiag_rhs = new DREAL[rhs->get_num_vectors()];
		ASSERT(sqrtdiag_rhs);
	}

	this->lhs = (CStringFeatures<CHAR>*) l;
	this->rhs = (CStringFeatures<CHAR>*) l;

	//compute normalize to 1 values
	for (i = 0; i<lhs->get_num_vectors(); i++)
	{
		sqrtdiag_lhs[i] = sqrt(compute(i,i));

		//trap divide by zero exception
		if (sqrtdiag_lhs[i]==0)
			sqrtdiag_lhs[i] = 1e-16;
	}

	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if (sqrtdiag_lhs!=sqrtdiag_rhs)
	{
		this->lhs = (CStringFeatures<CHAR>*) r;
		this->rhs = (CStringFeatures<CHAR>*) r;

		//compute normalize to 1 values
		for (i = 0; i<rhs->get_num_vectors(); i++)
		{
		  sqrtdiag_rhs[i] = sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i] = 1e-16;
		}
	}

	this->lhs = (CStringFeatures<CHAR>*) l;
	this->rhs = (CStringFeatures<CHAR>*) r;

	initialized = true;
	return result;
}

void CFixedDegreeStringKernel::cleanup()
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = NULL;

	initialized = false;
}

bool CFixedDegreeStringKernel::load_init(FILE* src)
{
	return false;
}

bool CFixedDegreeStringKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CFixedDegreeStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	CHAR* avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length
	ASSERT(alen==blen);

	DREAL sqrt = initialized? sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b] : 1;
	LONG sum = 0;
	for (INT i = 0; i<alen-degree; i++)
	{
		bool match = true;

		for (INT j = i; j<i+degree && match; j++)
			match = avec[j]==bvec[j];
		if (match)
			sum++;
	}
	return (DREAL) sum/sqrt;
}
