/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SIMPLELOCALITYIMPROVEDSTRINGKERNEL_H___
#define _SIMPLELOCALITYIMPROVEDSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

class CSimpleLocalityImprovedStringKernel: public CStringKernel<CHAR>
{
public:
	CSimpleLocalityImprovedStringKernel(int size, INT length,
		INT inner_degree, INT outer_degree);
	~CSimpleLocalityImprovedStringKernel();

	virtual bool init(CFeatures *l, CFeatures *r);
	virtual void cleanup();

	/// load and save kernel init_data
	bool load_init(FILE *src);
	bool save_init(FILE *dest);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type()
	{
		return K_SIMPLELOCALITYIMPROVED;
	}

	// return the name of a kernel
	virtual const CHAR *get_name()
	{
		return "SimpleLocalityImproved";
	}

private:
	DREAL dot_pyr (const CHAR* const x1, const CHAR* const x2,
		const INT NOF_NTS,
	const INT NTWIDTH, const INT DEGREE1, const INT DEGREE2, CHAR *stage1,
		DREAL *pyra);

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	DREAL compute(INT idx_a, INT idx_b); /* compute_kernel*/

protected:
	INT length;
	INT inner_degree;
	INT outer_degree;
	CHAR *match;
	DREAL *pyramid_weights;
};
#endif /* _SIMPLELOCALITYIMPROVEDSTRINGKERNEL_H___ */
