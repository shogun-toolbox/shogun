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
#include "kernel/SparseGaussianKernel.h"
#include "features/Features.h"
#include "features/SparseFeatures.h"

CSparseGaussianKernel::CSparseGaussianKernel(INT size, double w)
: CSparseKernel<DREAL>(size), width(w), sq_lhs(NULL), sq_rhs(NULL)
{
}

CSparseGaussianKernel::CSparseGaussianKernel(
	CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r, double w)
: CSparseKernel<DREAL>(10), width(w), sq_lhs(NULL), sq_rhs(NULL)
{
	init(l, r);
}

CSparseGaussianKernel::~CSparseGaussianKernel()
{
	cleanup();
}

bool CSparseGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	///free sq_{r,l}hs first
	cleanup();

	CSparseKernel<DREAL>::init(l, r);

	sq_lhs=new DREAL[lhs->get_num_vectors()];
	sq_lhs=((CSparseFeatures<DREAL>*) lhs)->compute_squared(sq_lhs);
	if (lhs==rhs)
		sq_rhs=sq_lhs;
	else
	{
		sq_rhs=new DREAL[rhs->get_num_vectors()];
		sq_rhs=((CSparseFeatures<DREAL>*) rhs)->compute_squared(sq_rhs);
	}

	return true;
}

void CSparseGaussianKernel::cleanup()
{
	if (sq_lhs != sq_rhs)
		delete[] sq_rhs;
	sq_rhs = NULL;

	delete[] sq_lhs;
	sq_lhs = NULL;

	CKernel::cleanup();
}

bool CSparseGaussianKernel::load_init(FILE* src)
{
	return false;
}

bool CSparseGaussianKernel::save_init(FILE* dest)
{
	return false;
}

DREAL CSparseGaussianKernel::compute(INT idx_a, INT idx_b)
{
	//DREAL result = sq_lhs[idx_a] + sq_rhs[idx_b];
	DREAL result=((CSparseFeatures<DREAL>*) lhs)->compute_squared_norm((CSparseFeatures<DREAL>*) lhs, sq_lhs, idx_a, (CSparseFeatures<DREAL>*) rhs, sq_rhs, idx_b);
	return exp(-result/width);
}
