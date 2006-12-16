/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _WEIGHTEDDEGREEPOSITIONPHYLCHARKERNE_H___
#define _WEIGHTEDDEGREEPOSITIONPHYLCHARKERNEL_H___

#include "lib/common.h"
#include "kernel/SimpleKernel.h"
#include "kernel/WeightedDegreePositionCharKernel.h"

#include "lib/Trie.h"

class CWeightedDegreePositionPhylCharKernel: public CWeightedDegreePositionCharKernel
{
public:
	CWeightedDegreePositionPhylCharKernel(LONG size, DREAL* weights, INT degree, INT max_mismatch, 
										  INT * shift, INT shift_len, bool use_norm=false,
										  INT mkl_stepsize=1) ;
	~CWeightedDegreePositionPhylCharKernel() ;
	
	virtual bool init(CFeatures* l, CFeatures* r, bool do_init);

	// return what type of kernel we are Linear,Polynomial, Gaussian,...
	virtual EKernelType get_kernel_type() { return K_WEIGHTEDDEGREEPHYLPOS; }
	
	// return the name of a kernel
	virtual const CHAR* get_name() { return "WeightedDegreePhylPos" ; } ;
	
	// set conservation weights (length x num_examples)
	virtual bool set_weights(DREAL* weights, INT len, INT num_examples);
	virtual bool set_position_weights(DREAL* position_weights, INT len)
	{
		CIO::message(M_ERROR, "not implemented\n") ; 
		return false ;
	}

	DREAL compute_by_tree(INT idx);
	void compute_by_tree(INT idx, DREAL* LevelContrib); // not implemented
	
	DREAL* compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* target, INT num_suppvec, INT* IDX, DREAL* weights);
	virtual void compute_batch(INT num_vec, INT* vec_idx, DREAL* target, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor=1.0);

	/// 
	virtual DREAL compute(INT idx_a, INT idx_b);
	DREAL compute_without_mismatch(CHAR* avec, DREAL* aphyl, INT alen, CHAR* bvec, DREAL *bphyl, INT blen) ;
	
protected:
	virtual void add_example_to_tree(INT idx, DREAL weight);
	void add_example_to_single_tree(INT idx, DREAL weight, INT tree_num);

	DREAL* lhs_phyl_weights, *rhs_phyl_weights ;
	DREAL* weights_buffer ;
};


#endif
