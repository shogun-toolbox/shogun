/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SALZBERGWORDKERNEL_H___
#define _SALZBERGWORDKERNEL_H___

#include "lib/common.h"
#include "kernel/StringKernel.h"
#include "classifier/PluginEstimate.h"
#include "features/StringFeatures.h"

class CSalzbergWordKernel: public CStringKernel<WORD>
{
public:
	CSalzbergWordKernel(INT size, CPluginEstimate* pie);
	CSalzbergWordKernel(CStringFeatures<WORD>* l, CStringFeatures<WORD>* r, CPluginEstimate *pie);
	virtual ~CSalzbergWordKernel() ;
	
	void set_prior_probs(DREAL pos_prior_, DREAL neg_prior_)
		{
			pos_prior=pos_prior_ ;
			neg_prior=neg_prior_ ;
			if (fabs(pos_prior+neg_prior-1)>1e-6)
				SG_WARNING( "priors don't sum to 1: %f+%f-1=%f\n", pos_prior, neg_prior, pos_prior+neg_prior-1) ;
		};
	virtual bool init(CFeatures* l, CFeatures* r);
	virtual void cleanup();

	/// load and save kernel init_data
	bool load_init(FILE* src);
	bool save_init(FILE* dest);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_SALZBERG; }

	// return the name of a kernel
	virtual const CHAR* get_name() { return "Salzberg" ; } ;

protected:
	/// compute kernel function for features a and b
	/// idx_{a,b} denote the index of the feature vectors
	/// in the corresponding feature object
	DREAL compute(INT idx_a, INT idx_b);
	//	DREAL compute_slow(LONG idx_a, LONG idx_b);

	inline INT compute_index(INT position, WORD symbol)
	{
		return position*num_symbols+symbol;
	}

protected:
	CPluginEstimate* estimate;

	DREAL* mean;
	DREAL* variance;

	DREAL* sqrtdiag_lhs;
	DREAL* sqrtdiag_rhs;

	DREAL* ld_mean_lhs ;
	DREAL* ld_mean_rhs ;

	INT num_params;
	INT num_symbols;
	DREAL sum_m2_s2 ;
	DREAL pos_prior, neg_prior ;
	bool initialized ;
};

#endif /* _SALZBERGWORDKERNEL_H__ */
