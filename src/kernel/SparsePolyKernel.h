/*
		 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSEPOLYKERNEL_H___
#define _SPARSEPOLYKERNEL_H___

#include "lib/common.h"
#include "kernel/SparseKernel.h"
#include "features/SparseFeatures.h"

class CSparsePolyKernel: public CSparseKernel<DREAL>
{
public:
	CSparsePolyKernel(CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r, INT size, INT d, bool inhom, bool use_norm);
	CSparsePolyKernel(INT size, INT degree, bool inhomogene=true, bool use_normalization=true);
	virtual ~CSparsePolyKernel();
	
	virtual bool init(CFeatures* l, CFeatures* r);
	virtual void cleanup();

	/// load and save kernel init_data
	virtual bool load_init(FILE* src);
	virtual bool save_init(FILE* dest);

	/** return feature type the kernel can deal with
	*/
	inline virtual EFeatureType get_feature_type() { return F_DREAL; }

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_POLY; }

	// return the name of a kernel
	virtual const CHAR* get_name() { return "SparsePoly" ; } ;

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual DREAL compute(INT idx_a, INT idx_b);

protected:
	INT degree;
	bool inhomogene ;

	double* sqrtdiag_lhs;
	double* sqrtdiag_rhs;

	bool initialized ;
	bool use_normalization;
};

#endif /* _SPARSEPOLYKERNEL_H__ */
