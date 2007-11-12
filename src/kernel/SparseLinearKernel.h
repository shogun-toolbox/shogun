/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SPARSELINEARKERNEL_H___
#define _SPARSELINEARKERNEL_H___

#include "lib/common.h"
#include "kernel/SparseKernel.h"
#include "features/SparseFeatures.h"

class CSparseLinearKernel: public CSparseKernel<DREAL>
{
public:
	CSparseLinearKernel(INT size, DREAL scale=1.0);
	CSparseLinearKernel(CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r, DREAL scale=1.0, INT size=10);
	virtual ~CSparseLinearKernel();
	
	virtual bool init(CFeatures* l, CFeatures* r);
	virtual void cleanup();

	/// load and save kernel init_data
	virtual bool load_init(FILE* src);
	virtual bool save_init(FILE* dest);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_SPARSELINEAR; }

	// return the name of a kernel
	virtual const CHAR* get_name() { return "SparseLinear" ; } ;

	///optimizable kernel, i.e. precompute normal vector and as phi(x)=x
	///do scalar product in input space
	virtual bool init_optimization(INT num_suppvec, INT* sv_idx, DREAL* alphas);
	virtual bool delete_optimization();
	virtual DREAL compute_optimized(INT idx);

	virtual void clear_normal();
	virtual void add_to_normal(INT idx, DREAL weight);

	inline const double* get_normal(INT& len)
	{
		len=normal_length;
		return normal;
	}

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	virtual DREAL compute(INT idx_a, INT idx_b);

	virtual void init_rescale();

protected:
	double scale ;
	bool initialized;

	/// normal vector (used in case of optimized kernel)
	INT normal_length;
	DREAL* normal;
};

#endif /* _SPARSELINEARKERNEL_H__ */
