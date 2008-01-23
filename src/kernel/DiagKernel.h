/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _DIAGKERNEL_H___
#define _DIAGKERNEL_H___

#include "lib/common.h"
#include "kernel/Kernel.h"

class CDiagKernel: public CKernel
{
public:
	CDiagKernel(INT size, DREAL diag=1.0);
	CDiagKernel(CFeatures* l, CFeatures* r, DREAL diag=1.0);
	virtual ~CDiagKernel();

	virtual void cleanup();

	/// load and save kernel init_data
	virtual bool load_init(FILE* src);
	virtual bool save_init(FILE* dest);

	/** return feature type the kernel can deal with
	*/
	inline virtual EFeatureType get_feature_type()
	{
		return F_ANY;
	}

	/** return feature class the kernel can deal with
	*/
	inline virtual EFeatureClass get_feature_class()
	{
		return C_ANY;
	}

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_DIAG; }

	// return the name of a kernel
	virtual const CHAR* get_name() { return "Diagonal" ; } ;

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	inline virtual DREAL compute(INT idx_a, INT idx_b)
	{
		if (idx_a==idx_b)
			return diag;
		else
			return 0;
	}

protected:
	double diag;
};

#endif /* _DIAGKERNEL_H__ */
