/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _POLYMATCHSTRINGKERNEL_H___
#define _POLYMATCHSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

class CPolyMatchStringKernel: public CStringKernel<CHAR>
{
public:
	CPolyMatchStringKernel(INT size, INT degree, bool inhomogene,
		bool use_normalization = true);
	~CPolyMatchStringKernel();

	virtual bool init(CFeatures* l, CFeatures* r);
	virtual void cleanup();

	/// load and save kernel init_data
	virtual bool load_init(FILE* src);
	virtual bool save_init(FILE* dest);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type()
	{
		return K_POLYMATCH;
	}

	// return the name of a kernel
	virtual const CHAR* get_name()
	{
		return "PolyMatchString";
	}

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual DREAL compute(INT idx_a, INT idx_b);

protected:
	INT degree;
	bool inhomogene;

	double* sqrtdiag_lhs;
	double* sqrtdiag_rhs;

	bool initialized;
	bool use_normalization;
};
#endif /* _POLYMATCHSTRINGKERNEL_H___ */
