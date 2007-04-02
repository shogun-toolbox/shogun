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
#include "lib/io.h"
#include "kernel/PolyMatchStringKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

CPolyMatchStringKernel::CPolyMatchStringKernel(LONG size, INT d, bool inhom,
		bool use_norm)
: CStringKernel<CHAR>(size),degree(d),inhomogene(inhom),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	use_normalization(use_norm)
{
}

CPolyMatchStringKernel::~CPolyMatchStringKernel()
{
	cleanup();
}

bool CPolyMatchStringKernel::init(CFeatures* l, CFeatures* r)
{
	bool result = CStringKernel<CHAR>::init(l, r);

	initialized = false;
	INT i;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = NULL;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = NULL;

	if (use_normalization)
	{
		sqrtdiag_lhs = new DREAL[lhs->get_num_vectors()];

		if (l==r)
			sqrtdiag_rhs = sqrtdiag_lhs;
		else
		{
			sqrtdiag_rhs = new DREAL[rhs->get_num_vectors()];
		}

		ASSERT(sqrtdiag_lhs);
		ASSERT(sqrtdiag_rhs);

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
	}

	this->lhs = (CStringFeatures<CHAR>*) l;
	this->rhs = (CStringFeatures<CHAR>*) r;

	initialized = true;
	return result;
}

void CPolyMatchStringKernel::cleanup()
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = NULL;

	initialized = false;
}

bool CPolyMatchStringKernel::load_init(FILE *src)
{
	return false;
}

bool CPolyMatchStringKernel::save_init(FILE *dest)
{
	return false;
}

DREAL CPolyMatchStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	//fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b);
	CHAR* avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	ASSERT(alen==blen);

	DREAL sqrt_a = 1;
	DREAL sqrt_b = 1;

	if (initialized && use_normalization)
	{
		sqrt_a = sqrtdiag_lhs[idx_a];
		sqrt_b = sqrtdiag_rhs[idx_b];
	}

	DREAL sqrt_both = sqrt_a*sqrt_b;
	INT ialen = (int) alen;
	INT sum = 0;
	for (INT i = 0; i<ialen; i++)
		sum += (avec[i]==bvec[i])? 1 : 0;
	if (inhomogene)
		sum += 1;
	DREAL result = sum;
	for (INT j = 1; j<degree; j++)
		result *= sum;
	result /= sqrt_both;
	return result;
}
