/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Alexander Binder
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef PYRAMIDCHI2_H_
#define PYRAMIDCHI2_H_

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

//pyramid classifier over Chi2 matched histograms
//TODO: port to CCombinedKernel (if it is the appropriate) as the pyramid is a weighted linear combination of kernels

class CPyramidChi2 : public CSimpleKernel<DREAL>
{
public:

	CPyramidChi2(INT size, DREAL width2,
		INT* pyramidlevels2, INT numlevels2,
		INT  numbinsinhistogram2, DREAL* weights2, INT numweights2);

	virtual bool init(CFeatures* l, CFeatures* r);

	CPyramidChi2(CRealFeatures* l, CRealFeatures* r, INT size, DREAL width2,
		INT* pyramidlevels2, INT numlevels2,
		INT  numbinsinhistogram2, DREAL* weights2, INT numweights2);

	virtual ~CPyramidChi2();

	virtual void cleanup();

	/// load and save kernel init_data
	virtual bool load_init(FILE* src);
	virtual bool save_init(FILE* dest);

	/// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type()
	{
		//preliminary output
		return K_PYRAMIDCHI2;
	}

	/// return the name of a kernel
	virtual const CHAR* get_name()
	{
		return("PyramidoverChi2\0");
	}

	/// sets standard weights
	void setstandardweights(); 

	/// performs a weak check, does not test for correct feature length
	bool sanitycheck_weak(); 

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual DREAL compute(INT idx_a, INT idx_b);

protected:
	DREAL width;
	INT* pyramidlevels;

	/// length of vector pyramidlevels
	INT numlevels; 
	INT numbinsinhistogram;
	DREAL* weights;

	/// length of vector weights
	INT numweights; 
	//bool sanitycheckbit;
};

#endif /*PYRAMIDCHI2_H_*/
