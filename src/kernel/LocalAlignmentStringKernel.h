/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LOCALALIGNMENTSTRINGKERNEL_H___
#define _LOCALALIGNMENTSTRINGKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"

#define LOGSUM_TBL 10000      /* span of the logsum table */ 

class CLocalAlignmentStringKernel: public CStringKernel<CHAR>
{
public:
	CLocalAlignmentStringKernel(INT size);
	~CLocalAlignmentStringKernel();

	virtual bool init(CFeatures* l, CFeatures* r);
	virtual void cleanup();

	/// load and save kernel init_data
	virtual bool load_init(FILE* src) { return false; }
	virtual bool save_init(FILE* dest) { return false; }

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type()
	{
		return K_LOCALALIGNMENT;
	}

	// return the name of a kernel
	virtual const CHAR* get_name()
	{
		return "LocalAlignment";
	}

private:
	void init_logsum(void);
	int LogSum(int p1, int p2);
	float LogSum2(float p1, float p2);
	void initialize(void);
	DREAL LAkernelcompute(int* aaX, int* aaY, int nX, int nY);

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual DREAL compute(INT idx_a, INT idx_b);

protected:
	bool initialized;

	int *isAA;                /* Indicates whether a char is an amino-acid */

	int *aaIndex;             /* The correspondance between amino-acid letter and index */

	int opening,extension; /* Gap penalties */

	static int logsum_lookup[LOGSUM_TBL];
};
#endif // _LOCALALIGNMENTKERNEL

