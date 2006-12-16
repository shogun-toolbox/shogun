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

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Trie.h"
#include "lib/Parallel.h"

#include "kernel/WeightedDegreePositionPhylCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"

#ifndef WIN32
#include <pthread.h>
#endif

struct S_THREAD_PARAM 
{
	INT* vec;
	DREAL* result;
	DREAL* weights;
	CWeightedDegreePositionPhylCharKernel* kernel;
	CTrie* tries;
	DREAL factor;
	INT j;
	INT start;
	INT end;
	INT length;
	INT max_shift;
	INT* shift;
	INT* vec_idx;
	DREAL* sqrtdiag_rhs;
};

CWeightedDegreePositionPhylCharKernel::CWeightedDegreePositionPhylCharKernel(LONG size, DREAL* w, INT d, 
																			 INT max_mismatch_, INT * shift_, 
																			 INT shift_len_, bool use_norm,
																			 INT mkl_stepsize_)
	: CWeightedDegreePositionCharKernel(size, w, d, max_mismatch, shift_, shift_len_, use_norm, mkl_stepsize_), 
	  lhs_phyl_weights(NULL), rhs_phyl_weights(NULL), weights_buffer(NULL)
{
}

CWeightedDegreePositionPhylCharKernel::~CWeightedDegreePositionPhylCharKernel() 
{
    delete[] lhs_phyl_weights ;
    lhs_phyl_weights = NULL ;

    delete[] rhs_phyl_weights ;
    rhs_phyl_weights = NULL ;

	delete[] weights_buffer ;
	weights_buffer = NULL ;
}

bool CWeightedDegreePositionPhylCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
    INT lhs_changed = (lhs!=l) ;
    INT rhs_changed = (rhs!=r) ;
	
    CIO::message(M_DEBUG, "lhs_changed: %i\n", lhs_changed) ;
    CIO::message(M_DEBUG, "rhs_changed: %i\n", rhs_changed) ;
	
	ASSERT(l && ((((CCharFeatures*) l)->get_alphabet()->get_alphabet()==DNA) || 
				 (((CCharFeatures*) l)->get_alphabet()->get_alphabet()==RNA)));
	ASSERT(r && ((((CCharFeatures*) r)->get_alphabet()->get_alphabet()==DNA) || 
				 (((CCharFeatures*) r)->get_alphabet()->get_alphabet()==RNA)));
	
    delete[] position_mask ;
    position_mask = NULL ;
	
    if (lhs_changed) 
    {
		INT alen ;
		bool afree ;
		CHAR* avec=((CCharFeatures*) l)->get_feature_vector(0, alen, afree);		
		seq_length = alen ;
		((CCharFeatures*) l)->free_feature_vector(avec, 0, afree);
		
		tries.destroy() ;
		if (opt_type==SLOWBUTMEMEFFICIENT)
			tries.create(alen, true); 
		else if (opt_type==FASTBUTMEMHUNGRY)
			tries.create(alen, false);  // still buggy
		else {
#ifdef HAVE_PYTHON
			throw KernelException("unknown optimization type\n");
#else
			CIO::message(M_ERROR, "unknown optimization type\n");
#endif
		}
		if (!lhs_phyl_weights)
		{
			CIO::message(M_DEBUG, "initializing lhs_phyl_weights\n") ;
			lhs_phyl_weights = new DREAL[seq_length * l->get_num_vectors()] ;
			for (INT i=0; i<alen*l->get_num_vectors(); i++)
				lhs_phyl_weights[i]=1.0 ;
		}
    } 
    if (rhs_changed)
	{
		if (!rhs_phyl_weights)
		{
			CIO::message(M_DEBUG, "initializing rhs_phyl_weights\n") ;
			rhs_phyl_weights = new DREAL[seq_length * r->get_num_vectors()] ;
			for (INT i=0; i<seq_length * r->get_num_vectors(); i++)
				rhs_phyl_weights[i]=1.0 ;
		}
	}	
	
    bool result=CSimpleKernel<CHAR>::init(l,r,do_init);
    initialized = false ;
    INT i;
	
    CIO::message(M_DEBUG, "use normalization:%d\n", (use_normalization) ? 1 : 0);
	
    if (use_normalization)
    {
		if (rhs_changed)
		{
			if (sqrtdiag_lhs != sqrtdiag_rhs)
				delete[] sqrtdiag_rhs;
			sqrtdiag_rhs=NULL ;
		}
		if (lhs_changed)
		{
			delete[] sqrtdiag_lhs;
			sqrtdiag_lhs=NULL ;
			sqrtdiag_lhs= new DREAL[lhs->get_num_vectors()];
			ASSERT(sqrtdiag_lhs) ;
			for (i=0; i<lhs->get_num_vectors(); i++)
				sqrtdiag_lhs[i]=1;
		}
		
		if (l==r)
			sqrtdiag_rhs=sqrtdiag_lhs;
		else if (rhs_changed)
		{
			sqrtdiag_rhs= new DREAL[rhs->get_num_vectors()];
			ASSERT(sqrtdiag_rhs) ;
			
			for (i=0; i<rhs->get_num_vectors(); i++)
				sqrtdiag_rhs[i]=1;
		}
		
		ASSERT(sqrtdiag_lhs);
		ASSERT(sqrtdiag_rhs);
		
		if (lhs_changed)
		{
			this->lhs=(CCharFeatures*) l;
			this->rhs=(CCharFeatures*) l;
			
			//compute normalize to 1 values
			for (i=0; i<lhs->get_num_vectors(); i++)
			{
				sqrtdiag_lhs[i]=sqrt(compute(i,i));
				
				//trap divide by zero exception
				if (sqrtdiag_lhs[i]==0)
					sqrtdiag_lhs[i]=1e-16;
			}
		}
		
		// if lhs is different from rhs (train/test data)
		// compute also the normalization for rhs
		if ((sqrtdiag_lhs!=sqrtdiag_rhs) & rhs_changed)
		{
			this->lhs=(CCharFeatures*) r;
			this->rhs=(CCharFeatures*) r;
			
			//compute normalize to 1 values
			for (i=0; i<rhs->get_num_vectors(); i++)
			{
				sqrtdiag_rhs[i]=sqrt(compute(i,i));
				
				//trap divide by zero exception
				if (sqrtdiag_rhs[i]==0)
					sqrtdiag_rhs[i]=1e-16;
			}
		}
    }
	
    this->lhs=(CCharFeatures*) l;
    this->rhs=(CCharFeatures*) r;

    initialized = true ;
    return result;
}

DREAL CWeightedDegreePositionPhylCharKernel::compute_without_mismatch(CHAR* avec, DREAL *aphyl, INT alen, 
																	  CHAR* bvec, DREAL *bphyl, INT blen) 
{
    DREAL sum0=0 ;
    DREAL *sum1=new DREAL[max_shift] ;
    for (INT i=0; i<max_shift; i++)
		sum1[i]=0 ;
	
    // no shift
    for (INT i=0; i<alen; i++)
    {
		DREAL sumi = 0.0 ;
		DREAL phyl = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			phyl += aphyl[i+j] * bphyl[i+j] ;
			sumi += weights[j] * phyl / (j+1.0) ;
		}
		sum0 += sumi ;
    } ;
	
    for (INT i=0; i<alen; i++)
    {
		for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			DREAL sumi = 0.0 ;
			DREAL phyl = 0.0 ;
			// shift in sequence a
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				phyl += aphyl[i+j+k] * bphyl[i+j] ;
				sumi += weights[j] * phyl / (j+1.0) ;
			}
			// shift in sequence b
			phyl = 0.0 ;
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				phyl += aphyl[i+j] * bphyl[i+j+k] ;
				sumi += weights[j] * phyl / (j+1.0) ;
			}
			sum1[k-1] += sumi ;
		} ;
    }
	
    DREAL result = sum0 ;
    for (INT i=0; i<max_shift; i++)
		result += sum1[i]/(2*(i+1)) ;
	
    delete[] sum1 ;
    return result ;
}

DREAL CWeightedDegreePositionPhylCharKernel::compute(INT idx_a, INT idx_b)
{
    INT alen, blen;
    bool afree, bfree;
	
    CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
    CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
	DREAL * aphyl = &lhs_phyl_weights[seq_length*idx_a] ;
	DREAL * bphyl = &rhs_phyl_weights[seq_length*idx_b] ;

    // can only deal with strings of same length
    ASSERT(alen == blen);
    ASSERT(shift_len == alen) ;
	
    DREAL sqrt_a=1;
    DREAL sqrt_b=1;
	
    if (initialized && use_normalization)
    {
		sqrt_a=sqrtdiag_lhs[idx_a];
		sqrt_b=sqrtdiag_rhs[idx_b];
    }
	
    DREAL sqrt_both=sqrt_a*sqrt_b;
	
    DREAL result = 0 ;
	ASSERT(max_mismatch==0) ;
	result = compute_without_mismatch(avec, aphyl, alen, bvec, bphyl, blen) ;
	
    ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
    ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
	
    result/=sqrt_both;
	
    return result ;
}

void CWeightedDegreePositionPhylCharKernel::add_example_to_tree(INT idx, DREAL alpha)
{
	// use lhs weights
	tries.set_weights_in_tree(true) ;
	
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;
	DREAL* phyl = &lhs_phyl_weights[seq_length*idx] ;
	
	if (use_normalization)
		alpha /=  sqrtdiag_lhs[idx] ;
	
	for (INT i=0; i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	if (opt_type==FASTBUTMEMHUNGRY)
	{
		//tries.set_use_compact_terminal_nodes(false) ;
		ASSERT(!tries.get_use_compact_terminal_nodes()) ;
	}
	
	ASSERT(len==seq_length) ;
	if (!weights_buffer)
		weights_buffer=new DREAL[seq_length*degree] ;
	for (INT i=0; i<seq_length; i++)
	{
		DREAL sum_phyl = 0.0 ;
		for (INT j=0; j<degree; j++)
		{
			sum_phyl += phyl[i] ;
			weights_buffer[j+i*degree] = weights[j]*sum_phyl / (j+1.0) ;
		}
	}
	
	for (INT i=0; i<len; i++)
    {
		INT max_s=-1;
		
		if (opt_type==SLOWBUTMEMEFFICIENT)
			max_s=0;
		else if (opt_type==FASTBUTMEMHUNGRY)
			max_s=shift[i];
		else {
#ifdef HAVE_PYTHON
			throw KernelException("unknown optimization type\n");
#else
			CIO::message(M_ERROR, "unknown optimization type\n");
#endif
		}
		
		for (INT s=max_s; s>=0; s--)
		{
			DREAL alpha_pw = (s==0) ? (alpha) : (alpha/(2.0*s)) ;
			tries.add_to_trie(i, s, vec, alpha_pw, weights_buffer, true) ;
			
			if ((s==0) || (i+s>=len))
				continue;
			
			tries.add_to_trie(i+s, -s, vec, alpha_pw, weights_buffer, true) ;
		}
    }
	
  ((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
  delete[] vec ;
  tree_initialized=true ;
}

void CWeightedDegreePositionPhylCharKernel::add_example_to_single_tree(INT idx, DREAL alpha, INT tree_num) 
{
	CIO::message(M_ERROR, "sorry not implemented") ;
}

DREAL CWeightedDegreePositionPhylCharKernel::compute_by_tree(INT idx)
{
	// use the rhs weights 
	tries.set_weights_in_tree(false) ;

	DREAL sum = 0 ;
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;
	DREAL* phyl = &rhs_phyl_weights[seq_length*idx] ;

	ASSERT(len==seq_length) ;
	if (!weights_buffer)
		weights_buffer=new DREAL[seq_length*degree] ;
	for (INT i=0; i<seq_length; i++)
	{
		DREAL sum_phyl = 0.0 ;
		for (INT j=0; j<degree; j++)
		{
			sum_phyl += phyl[i] ;
			weights_buffer[j+i*degree] = weights[j]*sum_phyl / (j+1.0) ;
		}
	}
	
	for (INT i=0; i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	for (INT i=0; i<len; i++)
		sum += tries.compute_by_tree_helper(vec, len, i, i, i, weights_buffer, true) ;
	
	if (opt_type==SLOWBUTMEMEFFICIENT)
    {
		for (INT i=0; i<len; i++)
		{
			for (INT s=1; (s<=shift[i]) && (i+s<len); s++)
			{
				sum+=tries.compute_by_tree_helper(vec, len, i, i+s, i, weights_buffer, true)/(2*s) ;
				sum+=tries.compute_by_tree_helper(vec, len, i+s, i, i, weights_buffer, true)/(2*s) ;
			}
		}
    }
	
	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	
	if (use_normalization)
		return sum/sqrtdiag_rhs[idx];
	else
		return sum;
}

void CWeightedDegreePositionPhylCharKernel::compute_by_tree(INT idx, DREAL* LevelContrib)
{
	CIO::message(M_ERROR, "not implemented") ;
}

bool CWeightedDegreePositionPhylCharKernel::set_weights(DREAL* ws, INT length, INT num_examples)
{
	if (seq_length!=length)
		CIO::message(M_ERROR, "lengths do not match: seq_length=%i length=%i\n", seq_length, length) ;
	if ((lhs->get_num_vectors()!=num_examples) && (rhs->get_num_vectors()!=num_examples))
		CIO::message(M_ERROR, "num_examples do not match: lhs->get_num_vectors()=%i rhs->get_num_vectors()=%i num_examples=%i\n", lhs->get_num_vectors(), rhs->get_num_vectors(), num_examples) ;
	
	if (lhs->get_num_vectors()==num_examples)
	{
		CIO::message(M_DEBUG, "setting lhs_phyl_weights\n") ;
		delete[] lhs_phyl_weights;
		lhs_phyl_weights=new DREAL[length*num_examples];
		ASSERT(lhs_phyl_weights) ;
		for (int i=0; i<length*num_examples; i++)
			lhs_phyl_weights[i]=ws[i];
	} ;

	if (rhs->get_num_vectors()==num_examples)
	{
		CIO::message(M_DEBUG, "setting rhs_phyl_weights\n") ;
		delete[] rhs_phyl_weights;
		rhs_phyl_weights=new DREAL[length*num_examples];
		ASSERT(rhs_phyl_weights) ;
		for (int i=0; i<length*num_examples; i++)
			rhs_phyl_weights[i]=ws[i];
	} ;

	CFeatures *orig_lhs=lhs, *orig_rhs=rhs ;
	lhs=NULL ;
	rhs=NULL ;
	return init(orig_lhs, orig_rhs, false) ;
}

void CWeightedDegreePositionPhylCharKernel::compute_batch(INT num_vec, INT* vec_idx, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor)
{
	CIO::message(M_ERROR, "not implemented") ;
}

DREAL* CWeightedDegreePositionPhylCharKernel::compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas)
{
	CIO::message(M_ERROR, "not implemented") ;
	return NULL ;
}




