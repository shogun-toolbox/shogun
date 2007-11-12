/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Written (W) 1999-2007 Gunnar Raetsch
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _HISTOGRAMWORDKERNEL_H___
#define _HISTOGRAMWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "classifier/PluginEstimate.h"
#include "features/WordFeatures.h"

class CHistogramWordKernel: public CSimpleKernel<WORD>
{
public:
	CHistogramWordKernel(INT size, CPluginEstimate* pie);
	CHistogramWordKernel(CWordFeatures* l, CWordFeatures* r, CPluginEstimate* pie);
	virtual ~CHistogramWordKernel();
	
	virtual bool init(CFeatures* l, CFeatures* r);
	virtual void cleanup();

	/// load and save kernel init_data
	bool load_init(FILE* src);
	bool save_init(FILE* dest);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_HISTOGRAM; }

	// return the name of a kernel
	virtual const CHAR* get_name() { return "Histogram" ; } ;

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	DREAL compute(INT idx_a, INT idx_b);
	//	DREAL compute_slow(LONG idx_a, LONG idx_b);

	inline INT compute_index(INT position, WORD symbol)
	{
		return position*num_symbols+symbol+1;
	}

protected:
	CPluginEstimate* estimate;

	DREAL* mean;
	DREAL* variance;

	DREAL* sqrtdiag_lhs;
	DREAL* sqrtdiag_rhs;

	DREAL* ld_mean_lhs;
	DREAL* ld_mean_rhs;

	DREAL* plo_lhs;
	DREAL* plo_rhs;

	INT num_params;
	INT num_params1;
	INT num_symbols;
	DREAL sum_m2_s2;

	bool initialized;
};

#endif /* _HISTOGRAMWORDKERNEL_H__ */
