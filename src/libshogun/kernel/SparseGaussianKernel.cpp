/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/SparseGaussianKernel.h"
#include "features/Features.h"
#include "features/SparseFeatures.h"

CSparseGaussianKernel::CSparseGaussianKernel(int32_t size, float64_t w)
: CSparseKernel<float64_t>(size), width(w), sq_lhs(NULL), sq_rhs(NULL)
{
}

CSparseGaussianKernel::CSparseGaussianKernel(
	CSparseFeatures<float64_t>* l, CSparseFeatures<float64_t>* r, float64_t w)
: CSparseKernel<float64_t>(10), width(w), sq_lhs(NULL), sq_rhs(NULL)
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

	CSparseKernel<float64_t>::init(l, r);

	sq_lhs=new float64_t[lhs->get_num_vectors()];
	sq_lhs=((CSparseFeatures<float64_t>*) lhs)->compute_squared(sq_lhs);
	if (lhs==rhs)
		sq_rhs=sq_lhs;
	else
	{
		sq_rhs=new float64_t[rhs->get_num_vectors()];
		sq_rhs=((CSparseFeatures<float64_t>*) rhs)->compute_squared(sq_rhs);
	}

	return init_normalizer();
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

float64_t CSparseGaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
	//float64_t result = sq_lhs[idx_a] + sq_rhs[idx_b];
	float64_t result=((CSparseFeatures<float64_t>*) lhs)->compute_squared_norm(
		(CSparseFeatures<float64_t>*) lhs, sq_lhs, idx_a,
		(CSparseFeatures<float64_t>*) rhs, sq_rhs, idx_b);
	return exp(-result/width);
}
