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
#include "kernel/GaussianShiftKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

CGaussianShiftKernel::CGaussianShiftKernel(int32_t size, double w, int32_t ms, int32_t ss)
: CGaussianKernel(size, w), max_shift(ms), shift_step(ss)
{
}

CGaussianShiftKernel::CGaussianShiftKernel(
	CRealFeatures* l, CRealFeatures* r, double w, int32_t ms, int32_t ss, int32_t size)
: CGaussianKernel(l, r, w, size), max_shift(ms), shift_step(ss)
{
	init(l,r);
}

CGaussianShiftKernel::~CGaussianShiftKernel()
{
}

DREAL CGaussianShiftKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen);

	DREAL result = 0.0 ;
	DREAL sum=0.0 ;
	for (int32_t i=0; i<alen; i++)
		sum+=(avec[i]-bvec[i])*(avec[i]-bvec[i]);
	result += exp(-sum/width) ;

	for (int32_t shift = shift_step, s=1; shift<max_shift; shift+=shift_step, s++)
	{
		sum=0.0 ;
		for (int32_t i=0; i<alen-shift; i++)
			sum+=(avec[i+shift]-bvec[i])*(avec[i+shift]-bvec[i]);
		result += exp(-sum/width)/(2*s) ;

		sum=0.0 ;
		for (int32_t i=0; i<alen-shift; i++)
			sum+=(avec[i]-bvec[i+shift])*(avec[i]-bvec[i+shift]);
		result += exp(-sum/width)/(2*s) ;
	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
