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

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Trie.h"
#include "base/Parallel.h"

#include "kernel/WeightedDegreePositionCharKernel.h"
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
	CWeightedDegreePositionCharKernel* kernel;
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

//#define NEWSTUFF

#ifdef NEWSTUFF
#include <ctype.h>

const int ProtSimThresh = 0 ;
const char ProtSimStr[21]="ARNDCQEGHILKMFPSTWYV" ;
const int ProtSimMat[20][20]=
{{  100, 22, 26, 24, 21, 38, 26, 32, 21, 27, 27, 36, 41, 19, 19, 54, 44, 10, 21, 44},
 {22,  100, 27, 21,  9, 48, 24, 13, 27, 16, 17, 60, 27, 12, 13, 29, 29,  6, 14, 20},
 {26, 27,  100, 42, 12, 42, 28, 24, 27, 16, 16, 41, 25, 14, 12, 40, 39,  7, 16, 21},
 {24, 21, 42, 100, 10, 37, 40, 18, 20, 14, 13, 34, 21, 12, 14, 33, 31,  5, 13, 18},
 {21,  9, 12, 10, 100, 14,  8,  9,  9, 19, 19, 13, 24, 13,  7, 19, 22,  6, 11, 25},
 {38, 48, 42, 37, 14, 100, 62, 24, 39, 24, 25, 62, 42, 19, 19, 50, 42, 10, 23, 31},
 {26, 24, 28, 40,  8, 62, 100, 14, 27, 13, 13, 48, 26, 12, 13, 37, 30,  6, 14, 19},
 {32, 13, 24, 18,  9, 24, 14, 100, 13, 10, 13, 23, 18, 11, 10, 28, 22,  6, 11, 16},
 {21, 27, 27, 20,  9, 39, 27, 13, 100, 14, 17, 30, 25, 18, 10, 28, 24,  8, 34, 18},
 {27, 16, 16, 14, 19, 24, 13, 10, 14, 100, 60, 23, 70, 32, 10, 23, 35, 11, 23, 89},
 {27, 17, 16, 13, 19, 25, 13, 13, 17, 60, 100, 24, 89, 41, 12, 24, 31, 14, 25, 68},
 {36, 60, 41, 34, 13, 62, 48, 23, 30, 23, 24, 100, 36, 17, 21, 42, 40,  9, 19, 29},
 {41, 27, 25, 21, 24, 42, 26, 18, 25, 70, 89, 36, 100, 47, 17, 37, 45, 16, 31, 80},
 {19, 12, 14, 12, 13, 19, 12, 11, 18, 32, 41, 17, 47, 100,  9, 20, 24, 17, 58, 37},
 {19, 13, 12, 14,  7, 19, 13, 10, 10, 10, 12, 21, 17,  9, 100, 21, 17,  4,  8, 15},
 {54, 29, 40, 33, 19, 50, 37, 28, 28, 23, 24, 42, 37, 20, 21, 100, 56,  9, 22, 34},
 {44, 29, 39, 31, 22, 42, 30, 22, 24, 35, 31, 40, 45, 24, 17, 56, 100,  9, 23, 48},
 {10,  6,  7,  5,  6, 10,  6,  6,  8, 11, 14,  9, 16, 17,  4,  9,  9, 100, 22, 13},
 {21, 14, 16, 13, 11, 23, 14, 11, 34, 23, 25, 19, 31, 58,  8, 22, 23, 22, 100, 27},
 {44, 20, 21, 18, 25, 31, 19, 16, 18, 89, 68, 29, 80, 37, 15, 34, 48, 13, 27, 100}};

int ProtSim[128][128] ;

#endif

CWeightedDegreePositionCharKernel::CWeightedDegreePositionCharKernel(LONG size, DREAL* w, INT d, 
																			 INT max_mismatch_, INT * shift_, 
																			 INT shift_len_, bool use_norm,
																			 INT mkl_stepsize_)
    : CSimpleKernel<CHAR>(size), weights(NULL), position_weights(NULL), position_mask(NULL), counts(NULL),
      weights_buffer(NULL), mkl_stepsize(mkl_stepsize_), degree(d), length(0),
      max_mismatch(max_mismatch_), seq_length(0), max_shift_vec(NULL), 
      sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
      use_normalization(use_norm), tries(d), tree_initialized(false)
{
#ifdef NEWSTUFF
    fprintf(stderr, "initializing protein similarity table -- experimental\n") ;
    for (int i=0; i<128; i++)
		for (int j=0; j<128; j++)
			ProtSim[i][j]=0 ;
    for (int i=0; i<20; i++)
		for (int j=0; j<20; j++)
		{
			ProtSim[(int)ProtSimStr[i]][(int)ProtSimStr[j]]=ProtSimMat[i][j] ;
			ProtSim[(int)tolower(ProtSimStr[i])][(int)tolower(ProtSimStr[j])]=ProtSimMat[i][j] ;
			ProtSim[(int)tolower(ProtSimStr[i])][(int)ProtSimStr[j]]=ProtSimMat[i][j] ;
			ProtSim[(int)ProtSimStr[i]][(int)tolower(ProtSimStr[j])]=ProtSimMat[i][j] ;
		}
#endif 
	
    properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
    lhs=NULL;
    rhs=NULL;
	
    weights=new DREAL[d*(1+max_mismatch)];
    counts = new INT[d*(1+max_mismatch)];

    ASSERT(weights!=NULL);
    for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];

    shift_len = shift_len_ ;
    shift = new INT[shift_len] ;
    max_shift = 0 ;
	
    for (INT i=0; i<shift_len; i++)
    {
		shift[i] = shift_[i] ;
		if (shift[i]>max_shift)
			max_shift = shift[i] ;
    } ;
    ASSERT(max_shift>=0 && max_shift<=shift_len) ;

	max_shift_vec = new DREAL[max_shift+1] ;
}

CWeightedDegreePositionCharKernel::~CWeightedDegreePositionCharKernel() 
{
    cleanup();

    delete[] shift;
    shift = NULL;

    delete[] counts;
    counts = NULL;

    delete[] weights ;
    weights=NULL ;

    delete[] position_weights ;
    position_weights=NULL ;

    delete[] position_mask ;
    position_mask=NULL ;

    delete[] weights_buffer ;
    weights_buffer = NULL ;

	delete[] max_shift_vec ;
	max_shift_vec = NULL ;
}

void CWeightedDegreePositionCharKernel::remove_lhs() 
{ 
	CIO::message(M_DEBUG, "deleting CWeightedDegreePositionCharKernel optimization\n");
    delete_optimization();

#ifdef USE_SVMLIGHT
    if (lhs)
	cache_reset() ;
#endif //USE_SVMLIGHT

    if (sqrtdiag_lhs != sqrtdiag_rhs)
	delete[] sqrtdiag_rhs;
    delete[] sqrtdiag_lhs;

    lhs = NULL ; 
    rhs = NULL ; 
    initialized = false ;
    sqrtdiag_lhs = NULL ;
    sqrtdiag_rhs = NULL ;
	
    tries.destroy() ;
} ;

void CWeightedDegreePositionCharKernel::remove_rhs()
{
#ifdef USE_SVMLIGHT
    if (rhs)
	cache_reset() ;
#endif //USE_SVMLIGHT

    if (sqrtdiag_lhs != sqrtdiag_rhs)
	delete[] sqrtdiag_rhs;
    sqrtdiag_rhs = sqrtdiag_lhs ;
    rhs = lhs ;
}

  
bool CWeightedDegreePositionCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
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
         sg_error(sg_err_fun,"unknown optimization type\n");
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

void CWeightedDegreePositionCharKernel::cleanup()
{
	CIO::message(M_DEBUG, "deleting CWeightedDegreePositionCharKernel optimization\n");
    delete_optimization();

    if (sqrtdiag_lhs != sqrtdiag_rhs)
	delete[] sqrtdiag_rhs;
    sqrtdiag_rhs = NULL;

    delete[] sqrtdiag_lhs;
    sqrtdiag_lhs = NULL;

    tries.destroy() ;

    lhs = NULL;
    rhs = NULL;

    seq_length = 0;
    initialized = false;
    tree_initialized = false;
}

bool CWeightedDegreePositionCharKernel::load_init(FILE* src)
{
    return false;
}

bool CWeightedDegreePositionCharKernel::save_init(FILE* dest)
{
    return false;
}

bool CWeightedDegreePositionCharKernel::init_optimization(INT p_count, INT * IDX, DREAL * alphas, INT tree_num, INT upto_tree)
{
    if (upto_tree<0)
		upto_tree=tree_num;
	
    if (max_mismatch!=0)
    {
      sg_error(sg_err_fun,"CWeightedDegreePositionCharKernel optimization not implemented for mismatch!=0\n");
		return false ;
    }

    if (tree_num<0)
		CIO::message(M_DEBUG, "deleting CWeightedDegreePositionCharKernel optimization\n");
	delete_optimization();

    if (tree_num<0)
		CIO::message(M_DEBUG, "initializing CWeightedDegreePositionCharKernel optimization\n") ;

	int i=0;
	for (i=0; i<p_count; i++)
	{
		if (tree_num<0)
		{
			if ( (i % (p_count/10+1)) == 0)
				CIO::progress(i,0,p_count);
			add_example_to_tree(IDX[i], alphas[i]);
		}
		else
		{
			for (INT t=tree_num; t<=upto_tree; t++)
				add_example_to_single_tree(IDX[i], alphas[i], t);
		}
	}

    if (tree_num<0)
		CIO::message(M_DEBUG, "done.           \n");
	
    set_is_initialized(true) ;
    return true ;
}

bool CWeightedDegreePositionCharKernel::delete_optimization() 
{ 
	if ((opt_type==FASTBUTMEMHUNGRY) && (tries.get_use_compact_terminal_nodes())) 
	{
		tries.set_use_compact_terminal_nodes(false) ;
		CIO::message(M_DEBUG, "disabling compact trie nodes with FASTBUTMEMHUNGRY\n") ;
	}
	
	if (get_is_initialized())
	{
		if (opt_type==SLOWBUTMEMEFFICIENT)
			tries.delete_trees(true); 
		else if (opt_type==FASTBUTMEMHUNGRY)
			tries.delete_trees(false);  // still buggy
		else {
         sg_error(sg_err_fun,"unknown optimization type\n");
      }
		set_is_initialized(false);
		
		return true;
    }
	
	return false;
}

DREAL CWeightedDegreePositionCharKernel::compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
    DREAL sum0=0 ;
    for (INT i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;
	
    // no shift
    for (INT i=0; i<alen; i++)
    {
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		
		INT mismatches=0;
		DREAL sumi = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
			{
				mismatches++ ;
				if (mismatches>max_mismatch)
					break ;
			} ;
			sumi += weights[j+degree*mismatches];
		}
		if (position_weights!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
    } ;
	
    for (INT i=0; i<alen; i++)
    {
		for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;
			
			DREAL sumi1 = 0.0 ;
			// shift in sequence a
			INT mismatches=0;
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
				{
					mismatches++ ;
					if (mismatches>max_mismatch)
						break ;
				} ;
				sumi1 += weights[j+degree*mismatches];
			}
			DREAL sumi2 = 0.0 ;
			// shift in sequence b
			mismatches=0;
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
				{
					mismatches++ ;
					if (mismatches>max_mismatch)
						break ;
				} ;
				sumi2 += weights[j+degree*mismatches];
			}
			if (position_weights!=NULL)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
    }
	
    DREAL result = sum0 ;
    for (INT i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;
	
    return result ;
}

#ifdef NEWSTUFF

DREAL CWeightedDegreePositionCharKernel::compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
    DREAL sum0=0 ;
    for (INT i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;
	
    // no shift
    for (INT i=0; i<alen; i++)
    {
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		DREAL sumi = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			int sim = ProtSim[(int)avec[i+j]][(int)bvec[i+j]] ;
			if (sim<ProtSimThresh)
				break ;
			sumi += weights[j]*((double)sim) ;
		}
		sumi/=100.0 ;
		if (position_weights!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
    } ;
	
    for (INT i=0; i<alen; i++)
    {
	if ((position_weights!=NULL) && (position_weights[i]==0.0))
	    continue ;
	for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
	{
	    DREAL sumi = 0.0 ;
	    // shift in sequence a
	    for (INT j=0; (j<degree) && (i+j+k<alen); j++)
	    {
		int sim = ProtSim[(int)avec[i+j+k]][(int)bvec[i+j]] ;
		if (sim<ProtSimThresh)
		    break ;
		sumi += weights[j]*((double)sim) ;
	    }
	    // shift in sequence b
	    for (INT j=0; (j<degree) && (i+j+k<alen); j++)
	    {
		int sim = ProtSim[(int)avec[i+j]][(int)bvec[i+j+k]] ;
		if (sim<ProtSimThresh)
		    break ;
		sumi += weights[j]*((double)sim) ;
	    }
	    sumi/=100.0 ;
	    if (position_weights!=NULL)
		max_shift_vec[k-1] += position_weights[i]*sumi ;
	    else
		max_shift_vec[k-1] += sumi ;
	} ;
    }

    DREAL result = sum0 ;
    for (INT i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

    return result ;
}

#else

DREAL CWeightedDegreePositionCharKernel::compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
    DREAL sum0=0 ;
    for (INT i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;
	
    // no shift
    for (INT i=0; i<alen; i++)
    {
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;

		DREAL sumi = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
    } ;
	
    for (INT i=0; i<alen; i++)
    {
		for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;
			
			DREAL sumi1 = 0.0 ;
			// shift in sequence a
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j];
			}
			DREAL sumi2 = 0.0 ;
			// shift in sequence b
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[j];
			}
			if (position_weights!=NULL)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
    }
	
    DREAL result = sum0 ;
    for (INT i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;
	
    return result ;
}

#endif

DREAL CWeightedDegreePositionCharKernel::compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
    DREAL sum0=0 ;
    for (INT i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;
	
    if (!position_mask)
    {		
		position_mask = new bool[alen] ;
		for (INT i=0; i<alen; i++)
		{
			position_mask[i]=false ;
			
			for (INT j=0; j<degree; j++)
				if (weights[i*degree+j]!=0.0)
				{
					position_mask[i]=true ;
					break ;
				}
		}
    }
	
    // no shift
    for (INT i=0; i<alen; i++)
    {
		if (!position_mask[i])
			continue ;
		
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		DREAL sumi = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[i*degree+j];
		}
		if (position_weights!=NULL)
			sum0 += position_weights[i]*sumi ;
		else
			sum0 += sumi ;
    } ;
	
    for (INT i=0; i<alen; i++)
    {
		if (!position_mask[i])
			continue ;		
		for (INT k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;
			
			DREAL sumi1 = 0.0 ;
			// shift in sequence a
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[i*degree+j];
			}
			DREAL sumi2 = 0.0 ;
			// shift in sequence b
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[i*degree+j];
			}
			if (position_weights!=NULL)
				max_shift_vec[k-1] += position_weights[i]*sumi1 + position_weights[i+k]*sumi2 ;
			else
				max_shift_vec[k-1] += sumi1 + sumi2 ;
		} ;
    }
	
    DREAL result = sum0 ;
    for (INT i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;
	
    return result ;
}


DREAL CWeightedDegreePositionCharKernel::compute(INT idx_a, INT idx_b)
{
    INT alen, blen;
    bool afree, bfree;

    CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
    CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

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
    if (max_mismatch > 0)
	result = compute_with_mismatch(avec, alen, bvec, blen) ;
    else if (length==0)
	result = compute_without_mismatch(avec, alen, bvec, blen) ;
    else
	result = compute_without_mismatch_matrix(avec, alen, bvec, blen) ;
  
    ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
    ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
  
    result/=sqrt_both;
  
    return result ;
}

void CWeightedDegreePositionCharKernel::add_example_to_tree(INT idx, DREAL alpha)
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;
	
	if (use_normalization)
		alpha /=  sqrtdiag_lhs[idx] ;
	
	for (INT i=0; i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	if (opt_type==FASTBUTMEMHUNGRY)
	{
		//tries.set_use_compact_terminal_nodes(false) ;
		ASSERT(!tries.get_use_compact_terminal_nodes()) ;
	}
	
	for (INT i=0; i<len; i++)
    {
		INT max_s=-1;
		
		if (opt_type==SLOWBUTMEMEFFICIENT)
			max_s=0;
		else if (opt_type==FASTBUTMEMHUNGRY)
			max_s=shift[i];
		else {
         sg_error(sg_err_fun,"unknown optimization type\n");
      }
		
		for (INT s=max_s; s>=0; s--)
		{
			DREAL alpha_pw = (s==0) ? (alpha) : (alpha/(2.0*s)) ;
			tries.add_to_trie(i, s, vec, alpha_pw, weights, (length!=0)) ;
			//fprintf(stderr, "tree=%i, s=%i, example=%i, alpha_pw=%1.2f\n", i, s, idx, alpha_pw) ;
			
			if ((s==0) || (i+s>=len))
				continue;
			
			tries.add_to_trie(i+s, -s, vec, alpha_pw, weights, (length!=0)) ;
			//fprintf(stderr, "tree=%i, s=%i, example=%i, alpha_pw=%1.2f\n", i+s, -s, idx, alpha_pw) ;
		}
    }
	
  ((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
  delete[] vec ;
  tree_initialized=true ;
}

void CWeightedDegreePositionCharKernel::add_example_to_single_tree(INT idx, DREAL alpha, INT tree_num) 
{
    INT len ;
    bool free ;
    CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
    ASSERT(max_mismatch==0) ;
    INT *vec = new INT[len] ;
	
    if (use_normalization)
		alpha /=  sqrtdiag_lhs[idx] ;
	
    INT max_s=-1;
	
    if (opt_type==SLOWBUTMEMEFFICIENT)
		max_s=0;
    else if (opt_type==FASTBUTMEMHUNGRY)
	{
		ASSERT(!tries.get_use_compact_terminal_nodes()) ;
		max_s=shift[tree_num];
	}
    else {
       sg_error(sg_err_fun,"unknown optimization type\n");
    }
    for (INT i=CMath::max(0,tree_num-max_shift); i<CMath::min(len,tree_num+degree+max_shift); i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
    for (INT s=max_s; s>=0; s--)
    {
		DREAL alpha_pw = (s==0) ? (alpha) : (alpha/(2.0*s)) ;
		tries.add_to_trie(tree_num, s, vec, alpha_pw, weights, (length!=0)) ;
		//fprintf(stderr, "tree=%i, s=%i, example=%i, alpha_pw=%1.2f\n", tree_num, s, idx, alpha_pw) ;
	} 
	
    if (opt_type==FASTBUTMEMHUNGRY)
    {
		for (INT i=CMath::max(0,tree_num-max_shift); i<CMath::min(len,tree_num+max_shift+1); i++)
		{
			INT s=tree_num-i;
			if ((i+s<len) && (s>=1) && (s<=shift[i]))
			{
				DREAL alpha_pw = (s==0) ? (alpha) : (alpha/(2.0*s)) ;
				tries.add_to_trie(tree_num, -s, vec, alpha_pw, weights, (length!=0)) ; 
				//fprintf(stderr, "tree=%i, s=%i, example=%i, alpha_pw=%1.2f\n", tree_num, -s, idx, alpha_pw) ;
			}
		}
    }
	
    ((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
    delete[] vec ;
    tree_initialized=true ;
}

DREAL CWeightedDegreePositionCharKernel::compute_by_tree(INT idx)
{
	DREAL sum = 0 ;
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;
	
	for (INT i=0; i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	for (INT i=0; i<len; i++)
		sum += tries.compute_by_tree_helper(vec, len, i, i, i, weights, (length!=0)) ;
	
	if (opt_type==SLOWBUTMEMEFFICIENT)
    {
		for (INT i=0; i<len; i++)
		{
			for (INT s=1; (s<=shift[i]) && (i+s<len); s++)
			{
				sum+=tries.compute_by_tree_helper(vec, len, i, i+s, i, weights, (length!=0))/(2*s) ;
				sum+=tries.compute_by_tree_helper(vec, len, i+s, i, i+s, weights, (length!=0))/(2*s) ;
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

void CWeightedDegreePositionCharKernel::compute_by_tree(INT idx, DREAL* LevelContrib)
{
  INT len ;
  bool free ;
  CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);
  ASSERT(max_mismatch==0) ;
  INT *vec = new INT[len] ;
  
  for (INT i=0; i<len; i++)
    vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
  
  DREAL factor = 1.0 ;
  
  if (use_normalization)
    factor = 1.0/sqrtdiag_rhs[idx] ;
  
  for (INT i=0; i<len; i++)
    tries.compute_by_tree_helper(vec, len, i, i, i, LevelContrib, factor, mkl_stepsize, weights, (length!=0)) ;
  
  if (opt_type==SLOWBUTMEMEFFICIENT)
    {
      for (INT i=0; i<len; i++)
	for (INT k=1; (k<=shift[i]) && (i+k<len); k++)
	  {
	    tries.compute_by_tree_helper(vec, len, i, i+k, i, LevelContrib, factor/(2*k), mkl_stepsize, weights, (length!=0)) ;
	    tries.compute_by_tree_helper(vec, len, i+k, i, i+k, LevelContrib, factor/(2*k), mkl_stepsize, weights, (length!=0)) ;
	  }
    }
  
  ((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
  delete[] vec ;
}

DREAL *CWeightedDegreePositionCharKernel::compute_abs_weights(int &len) 
{
  return tries.compute_abs_weights(len) ;
}

bool CWeightedDegreePositionCharKernel::set_weights(DREAL* ws, INT d, INT len)
{
    CIO::message(M_DEBUG, "degree = %i  d=%i\n", degree, d) ;
    degree = d ;
    length=len;
	
    if (len <= 0)
		len=1;
	
    delete[] weights;
    weights=new DREAL[d*len];

    delete[] position_mask ;
    position_mask=NULL ;
	
    if (weights)
    {
		for (int i=0; i<degree*len; i++)
			weights[i]=ws[i];
		return true;
    }
    else
		return false;
}

bool CWeightedDegreePositionCharKernel::set_position_weights(DREAL* pws, INT len)
{
	fprintf(stderr, "len=%i\n", len) ;
	
    if (len==0)
    {
		delete[] position_weights ;
		position_weights = NULL ;
		tries.set_position_weights(position_weights) ;
		return true ;
    }
    if (seq_length==0)
		seq_length = len ;
	
    if (seq_length!=len) 
    {
		sg_error(sg_err_fun,"seq_length = %i, position_weights_length=%i\n", seq_length, len) ;
		return false ;
    }
    delete[] position_weights;
    position_weights=new DREAL[len];
	tries.set_position_weights(position_weights) ;
	
    if (position_weights)
    {
		for (int i=0; i<len; i++)
			position_weights[i]=pws[i];
		return true;
    }
    else
		return false;
}

void* CWeightedDegreePositionCharKernel::compute_batch_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;
	INT j=params->j;
	CWeightedDegreePositionCharKernel* wd=params->kernel;
	CTrie* tries=params->tries;
	DREAL* weights=params->weights;
	INT length=params->length;
	INT max_shift=params->max_shift;
	DREAL* sqrtdiag_rhs=params->sqrtdiag_rhs;
	INT* vec=params->vec;
	DREAL* result=params->result;
	DREAL factor=params->factor;
	INT* shift=params->shift;
	INT* vec_idx=params->vec_idx;

	for (INT i=params->start; i<params->end; i++)
	{
		INT len=0;
		bool freevec;
		CHAR* char_vec=((CCharFeatures*) wd->get_rhs())->get_feature_vector(vec_idx[i], len, freevec);
		for (INT k=CMath::max(0,j-max_shift); k<CMath::min(len,j+wd->get_degree()+max_shift); k++)
			vec[k]=((CCharFeatures*) wd->get_lhs())->get_alphabet()->remap_to_bin(char_vec[k]);

			DREAL norm_fac = 1.0 ;
			if (wd->get_use_normalization())
				norm_fac=1.0/sqrtdiag_rhs[vec_idx[i]] ;
			
			result[i] += factor*tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0))*norm_fac ;

		if (wd->get_optimization_type()==SLOWBUTMEMEFFICIENT)
		{
			for (INT q=CMath::max(0,j-max_shift); q<CMath::min(len,j+max_shift+1); q++)
			{
				INT s=j-q ;
				if ((s>=1) && (s<=shift[q]) && (q+s<len))
					result[i] += tries->compute_by_tree_helper(vec, len, q, q+s, q, weights, (length!=0))*norm_fac/(2.0*s) ;
			}
			for (INT s=1; (s<=shift[j]) && (j+s<len); s++)
				result[i] += tries->compute_by_tree_helper(vec, len, j+s, j, j+s, weights, (length!=0))*norm_fac/(2.0*s) ;
		}

		((CCharFeatures*) wd->get_rhs())->free_feature_vector(char_vec, vec_idx[i], freevec);
	}

	return NULL;
}

void CWeightedDegreePositionCharKernel::compute_batch(INT num_vec, INT* vec_idx, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor)
{
    ASSERT(get_rhs());
    ASSERT(num_vec<=get_rhs()->get_num_vectors());
    ASSERT(num_vec>0);
	ASSERT(vec_idx);
	ASSERT(result);

    INT num_feat=((CCharFeatures*) get_rhs())->get_num_features();
    ASSERT(num_feat>0);
	INT num_threads=CParallel::get_num_threads();
	ASSERT(num_threads>0);
	INT* vec= new INT[num_threads*num_feat];
	ASSERT(vec);
	
	if (num_threads < 2)
	{
#ifdef WIN32
		for (INT j=0; j<num_feat; j++)
#else
		for (INT j=0; j<num_feat && !CSignal::cancel_computations(); j++)
#endif
		{
			init_optimization(num_suppvec, IDX, alphas, j);
			S_THREAD_PARAM params;
			params.vec=vec;
			params.result=result;
			params.weights=weights;
			params.kernel=this;
			params.tries=&tries;
			params.factor=factor;
			params.j=j;
			params.start=0;
			params.end=num_vec;
			params.length=length;
			params.max_shift=max_shift;
			params.shift=shift;
			params.vec_idx=vec_idx;
			params.sqrtdiag_rhs=sqrtdiag_rhs;
			compute_batch_helper((void*) &params);

			CIO::progress(j,0,num_feat);
		}
	}
#ifndef WIN32
	else
	{
		for (INT j=0; j<num_feat && !CSignal::cancel_computations(); j++)
		{
			init_optimization(num_suppvec, IDX, alphas, j);
			pthread_t threads[num_threads-1];
			S_THREAD_PARAM params[num_threads];
			INT step= num_vec/num_threads;
			INT t;

			for (t=0; t<num_threads-1; t++)
			{
				params[t].vec=&vec[num_feat*t];
				params[t].result=result;
				params[t].weights=weights;
				params[t].kernel=this;
				params[t].tries=&tries;
				params[t].factor=factor;
				params[t].j=j;
				params[t].start = t*step;
				params[t].end = (t+1)*step;
				params[t].length=length;
				params[t].max_shift=max_shift;
				params[t].shift=shift;
				params[t].vec_idx=vec_idx;
				params[t].sqrtdiag_rhs=sqrtdiag_rhs;
				pthread_create(&threads[t], NULL, CWeightedDegreePositionCharKernel::compute_batch_helper, (void*)&params[t]);
			}

			params[t].vec=&vec[num_feat*t];
			params[t].result=result;
			params[t].weights=weights;
			params[t].kernel=this;
			params[t].tries=&tries;
			params[t].factor=factor;
			params[t].j=j;
			params[t].start=t*step;
			params[t].end=num_vec;
			params[t].length=length;
			params[t].max_shift=max_shift;
			params[t].shift=shift;
			params[t].vec_idx=vec_idx;
			params[t].sqrtdiag_rhs=sqrtdiag_rhs;
			compute_batch_helper((void*) &params[t]);

			for (t=0; t<num_threads-1; t++)
				pthread_join(threads[t], NULL);
			CIO::progress(j,0,num_feat);
		}
	}
#endif
    
    delete[] vec;
}

DREAL* CWeightedDegreePositionCharKernel::compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas)
{
    num_feat=((CCharFeatures*) get_rhs())->get_num_features();
    ASSERT(num_feat>0);
    ASSERT(((CCharFeatures*) get_rhs())->get_alphabet()->get_alphabet() == DNA);
    num_sym=4; //for now works only w/ DNA

    // === variables
    INT* nofsKmers = new INT[ max_degree ];
    DREAL** C = new DREAL*[ max_degree ];
    DREAL** L = new DREAL*[ max_degree ];
    DREAL** R = new DREAL*[ max_degree ];
    INT i;
    INT k;

    // --- return table
    INT bigtabSize = 0;
    for( k = 0; k < max_degree; ++k ) {
		nofsKmers[k] = (INT) pow( num_sym, k+1 );
        const INT tabSize = nofsKmers[k] * num_feat;
        bigtabSize += tabSize;
    }
    result= new DREAL[ bigtabSize ];
	
    // --- auxilliary tables
    INT tabOffs=0;
    for( k = 0; k < max_degree; ++k )
    {
		const INT tabSize = nofsKmers[k] * num_feat;
		C[k] = &result[tabOffs];
		L[k] = new DREAL[ tabSize ];
		R[k] = new DREAL[ tabSize ];
		tabOffs+=tabSize;
		for(i = 0; i < tabSize; i++ )
		{
			C[k][i] = 0.0;
			L[k][i] = 0.0;
			R[k][i] = 0.0;
		}
    }
	
    // --- tree parsing info
    DREAL* margFactors = new DREAL[ degree ];
    INT* x = new INT[ degree+1 ];
    INT* substrs = new INT[ degree+1 ];
    // - fill arrays
    margFactors[0] = 1.0;
    substrs[0] = 0;
    for( k=1; k < degree; ++k ) {
		margFactors[k] = 0.25 * margFactors[k-1];
		substrs[k] = -1;
    }
    substrs[degree] = -1;
    // - fill struct
    struct TreeParseInfo info;
    info.num_sym = num_sym;
    info.num_feat = num_feat;
    info.p = -1;
    info.k = -1;
    info.nofsKmers = nofsKmers;
    info.margFactors = margFactors;
    info.x = x;
    info.substrs = substrs;
    info.y0 = 0;
    info.C_k = NULL;
    info.L_k = NULL;
    info.R_k = NULL;
	
    // === main loop
    i = 0; // total progress
    for( k = 0; k < max_degree; ++k )
    {
		const INT nofKmers = nofsKmers[ k ];
		info.C_k = C[k];
		info.L_k = L[k];
		info.R_k = R[k];
		
		// --- run over all trees
		for(INT p = 0; p < num_feat; ++p )
		{
			init_optimization( num_suppvec, IDX, alphas, p );
			INT tree = p ;
			for(INT j = 0; j < degree+1; j++ ) {
				x[j] = -1;
			}
			tries.traverse( tree, p, info, 0, x, k );
			CIO::progress(i++,0,num_feat*max_degree);
	}
		
		// --- add partial overlap scores
		if( k > 0 ) {
			const INT j = k - 1;
			const INT nofJmers = (INT) pow( num_sym, j+1 );
			for(INT p = 0; p < num_feat; ++p ) {
				const INT offsetJ = nofJmers * p;
				const INT offsetJ1 = nofJmers * (p+1);
				const INT offsetK = nofKmers * p;
				INT y;
				INT sym;
				for( y = 0; y < nofJmers; ++y ) {
					for( sym = 0; sym < num_sym; ++sym ) {
						const INT y_sym = num_sym*y + sym;
						const INT sym_y = nofJmers*sym + y;
						ASSERT( 0 <= y_sym && y_sym < nofKmers );
						ASSERT( 0 <= sym_y && sym_y < nofKmers );
						C[k][ y_sym + offsetK ] += L[j][ y + offsetJ ];
						if( p < num_feat-1 ) {
							C[k][ sym_y + offsetK ] += R[j][ y + offsetJ1 ];
						}
					}
				}
			}
		}
        //   if( k > 1 )
        //     j = k-1
        //     for all positions p
        //       for all j-mers y
        //          for n in {A,C,G,T}
        //            C_k[ p, [y,n] ] += L_j[ p, y ]
        //            C_k[ p, [n,y] ] += R_j[ p+1, y ]
        //          end;
        //       end;
        //     end;
        //   end;
    }
	
    // === return a vector
    num_feat=1;
    num_sym = bigtabSize;
    // --- clean up
    delete[] nofsKmers;
    delete[] margFactors;
    delete[] substrs;
    delete[] x;
    delete[] C;
    for( k = 0; k < max_degree; ++k ) {
		delete[] L[k];
		delete[] R[k];
    }
    delete[] L;
    delete[] R;
    return result;
}




