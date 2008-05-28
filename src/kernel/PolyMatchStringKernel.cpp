/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/PolyMatchStringKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

CPolyMatchStringKernel::CPolyMatchStringKernel(
	INT size, INT d, bool i, bool un)
: CStringKernel<CHAR>(size), degree(d), inhomogene(i),
	use_normalization(un), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL),
	initialized(false)
{
}

CPolyMatchStringKernel::CPolyMatchStringKernel(
	CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r, INT d, bool i, bool un)
: CStringKernel<CHAR>(10), degree(d), inhomogene(i), use_normalization(un),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false)
{
	init(l, r);
}

CPolyMatchStringKernel::~CPolyMatchStringKernel()
{
	cleanup();
}

bool CPolyMatchStringKernel::init(CFeatures* l, CFeatures* r)
{
	bool result=CStringKernel<CHAR>::init(l, r);

	initialized=false;

	if (sqrtdiag_lhs!=sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	if (use_normalization)
	{
		sqrtdiag_lhs=new DREAL[lhs->get_num_vectors()];

		if (l==r)
			sqrtdiag_rhs=sqrtdiag_lhs;
		else
			sqrtdiag_rhs=new DREAL[rhs->get_num_vectors()];

		this->lhs=(CStringFeatures<CHAR>*) l;
		this->rhs=(CStringFeatures<CHAR>*) l;

		CKernel::init_sqrt_diag(sqrtdiag_lhs, lhs->get_num_vectors());

		// if lhs is different from rhs (train/test data)
		// compute also the normalization for rhs
		if (sqrtdiag_lhs!=sqrtdiag_rhs)
		{
			this->lhs=(CStringFeatures<CHAR>*) r;
			this->rhs=(CStringFeatures<CHAR>*) r;

			CKernel::init_sqrt_diag(sqrtdiag_rhs, rhs->get_num_vectors());
		}
	}

	this->lhs=(CStringFeatures<CHAR>*) l;
	this->rhs=(CStringFeatures<CHAR>*) r;

	initialized=true;
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
	INT i, alen, blen, sum;

	//fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b);
	CHAR* avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	ASSERT(alen==blen);
	DREAL sqrt = (initialized && use_normalization)?
		sqrtdiag_lhs[idx_a]*sqrtdiag_rhs[idx_b] : 1;
	for (i = 0, sum = inhomogene; i<alen; i++)
		if (avec[i]==bvec[i])
			sum++;
	return pow(sum, degree) / sqrt;
}
