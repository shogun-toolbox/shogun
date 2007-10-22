/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _GAUSSIANSHIFTKERNEL_H___
#define _GAUSSIANSHIFTKERNEL_H___

#include "lib/common.h"
#include "kernel/GaussianKernel.h"

class CGaussianShiftKernel: public CGaussianKernel
{
public:
	CGaussianShiftKernel(INT size, double width, int max_shift, int shift_step);
	CGaussianShiftKernel(CRealFeatures* l, CRealFeatures* r, double width, int max_shift, int shift_step, INT size=10);
	virtual ~CGaussianShiftKernel();
	
	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_GAUSSIANSHIFT; }
	
	// return the name of a kernel
	virtual const CHAR* get_name() { return "GaussianShift" ; } ;
	
protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual DREAL compute(INT idx_a, INT idx_b);
	/*    compute_kernel*/
	
protected:
	int max_shift, shift_step ;
};

#endif
