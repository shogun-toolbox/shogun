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

#include "kernel/WeightedDegreeCharKernel.h"
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
	CWeightedDegreeCharKernel* kernel;
	CTrie* tries;
	DREAL factor;
	INT j;
	INT start;
	INT end;
	INT length;
	DREAL* sqrtdiag_rhs;
	INT* vec_idx;
};

CWeightedDegreeCharKernel::CWeightedDegreeCharKernel(INT size, EWDKernType typ, INT deg, INT max_mismatch_,
		bool use_norm, bool block, INT mkl_stepsize_, INT which_deg)
	: CSimpleKernel<CHAR>(size),weights(NULL),position_weights(NULL),
	  weights_buffer(NULL), mkl_stepsize(mkl_stepsize_),degree(deg), length(0), 
	  max_mismatch(max_mismatch_), seq_length(0), 
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), 
	  block_computation(block), use_normalization(use_norm), 
	  num_block_weights_external(0), block_weights_external(NULL), block_weights(NULL),
	  type(typ), which_degree(which_deg), tries(deg,max_mismatch_==0), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	lhs=NULL;
	rhs=NULL;

	// weights will be set using set_wd_weights
	if (typ != E_EXTERNAL)
		set_wd_weights_by_type(type);
}

CWeightedDegreeCharKernel::CWeightedDegreeCharKernel(INT size, double* w, INT d, INT max_mismatch_, 
		bool use_norm, bool block, INT mkl_stepsize_, INT which_deg)
	: CSimpleKernel<CHAR>(size),weights(NULL),position_weights(NULL),weights_buffer(NULL), mkl_stepsize(mkl_stepsize_), degree(d), length(0), 
	  max_mismatch(max_mismatch_), seq_length(0), 
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), 
	  block_computation(block), use_normalization(use_norm),
	  num_block_weights_external(0), block_weights_external(NULL), block_weights(NULL),
	  type(E_EXTERNAL), which_degree(which_deg), tries(d,max_mismatch_==0), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	lhs=NULL;
	rhs=NULL;

	weights=new DREAL[d*(1+max_mismatch)];
	ASSERT(weights!=NULL);
	for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];
}

CWeightedDegreeCharKernel::~CWeightedDegreeCharKernel() 
{
	cleanup();

	delete[] weights;
	weights=NULL;

	delete[] position_weights ;
	position_weights=NULL ;

	delete[] weights_buffer ;
	weights_buffer = NULL ;

}


void CWeightedDegreeCharKernel::remove_lhs() 
{ 
	SG_DEBUG( "deleting CWeightedDegreeCharKernel optimization\n");
	delete_optimization();

#ifdef SVMLIGHT
	if (lhs)
		cache_reset();
#endif

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;

	lhs = NULL; 
	rhs = NULL; 
	initialized = false;
	sqrtdiag_lhs = NULL;
	sqrtdiag_rhs = NULL;

	tries.destroy() ;
}

void CWeightedDegreeCharKernel::remove_rhs()
{
#ifdef SVMLIGHT
	if (rhs)
		cache_reset() ;
#endif

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs ;
	rhs = lhs ;
}

  
bool CWeightedDegreeCharKernel::init(CFeatures* l, CFeatures* r)
{
	INT lhs_changed = (lhs!=l) ;
	INT rhs_changed = (rhs!=r) ;

	SG_DEBUG( "lhs_changed: %i\n", lhs_changed);
	SG_DEBUG( "rhs_changed: %i\n", rhs_changed);

	ASSERT(l && (l->get_feature_type() == F_CHAR) && (l->get_feature_class() == C_SIMPLE)) ;
	ASSERT(r && (r->get_feature_type() == F_CHAR) && (r->get_feature_class() == C_SIMPLE)) ;
	
	ASSERT(l && ((((CCharFeatures*) l)->get_alphabet()->get_alphabet()==DNA) || 
				 (((CCharFeatures*) l)->get_alphabet()->get_alphabet()==RNA)));
	ASSERT(r && ((((CCharFeatures*) r)->get_alphabet()->get_alphabet()==DNA) || 
				 (((CCharFeatures*) r)->get_alphabet()->get_alphabet()==RNA)));
	
	if (lhs_changed) 
	{
		INT alen ;
		bool afree ;
		CHAR* avec=((CCharFeatures*) l)->get_feature_vector(0, alen, afree);
		seq_length = alen ;
		((CCharFeatures*) l)->free_feature_vector(avec, 0, afree);

		tries.destroy() ;
		tries.create(alen, max_mismatch==0) ;
	} 

	bool result=CSimpleKernel<CHAR>::init(l,r);
	initialized = false;
	INT i;

	init_block_weights();

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

void CWeightedDegreeCharKernel::cleanup()
{
	SG_DEBUG( "deleting CWeightedDegreeCharKernel optimization\n");
	delete_optimization();

	delete[] block_weights;
	block_weights=NULL;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs = NULL;

	tries.destroy() ;

	lhs = NULL;
	rhs = NULL;

	seq_length=0;
	initialized=false;
	tree_initialized = false;
}

bool CWeightedDegreeCharKernel::load_init(FILE* src)
{
	return false;
}

bool CWeightedDegreeCharKernel::save_init(FILE* dest)
{
	return false;
}
  

bool CWeightedDegreeCharKernel::init_optimization(INT count, INT* IDX, DREAL* alphas, INT tree_num)
{
	if (tree_num<0)
		SG_DEBUG( "deleting CWeightedDegreeCharKernel optimization\n");
	delete_optimization();

	if (tree_num<0)
		SG_DEBUG( "initializing CWeightedDegreeCharKernel optimization\n") ;

	int i=0;
	for (i=0; i<count; i++)
	{
		if (tree_num<0)
		{
			if ( (i % (count/10+1)) == 0)
				SG_PROGRESS(i, 0, count);
			
			if (max_mismatch==0)
				add_example_to_tree(IDX[i], alphas[i]) ;
			else
				add_example_to_tree_mismatch(IDX[i], alphas[i]) ;

			//SG_DEBUG( "number of used trie nodes: %i\n", tries.get_num_used_nodes()) ;
		}
		else
		{
			if (max_mismatch==0)
				add_example_to_single_tree(IDX[i], alphas[i], tree_num) ;
			else
				add_example_to_single_tree_mismatch(IDX[i], alphas[i], tree_num) ;
		}
	}
	
	if (tree_num<0)
		SG_PRINT( "done.           \n");

	//tries.compact_nodes(NO_CHILD, 0, weights) ;

	set_is_initialized(true) ;
	return true ;
}

bool CWeightedDegreeCharKernel::delete_optimization() 
{ 
	if (get_is_initialized())
	{
		tries.delete_trees(max_mismatch==0); 
		set_is_initialized(false);
		return true;
	}
	
	return false;
}


DREAL CWeightedDegreeCharKernel::compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	DREAL sum = 0.0 ;
	
	for (INT i=0; i<alen; i++)
	{
		DREAL sumi = 0.0 ;
		INT mismatches=0 ;
		
		for (INT j=0; (i+j<alen) && (j<degree); j++)
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
			sum+=position_weights[i]*sumi ;
		else
			sum+=sumi ;
	}
	return sum ;
}

DREAL CWeightedDegreeCharKernel::compute_using_block(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	DREAL sum=0;

	INT match_len=-1;

	ASSERT(alen==blen);

	for (INT i=0; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			match_len++;
		else
		{
			if (match_len>=0)
				sum+=block_weights[match_len];
			match_len=-1;
		}
	}

	if (match_len>=0)
		sum+=block_weights[match_len];

	return sum;
}

DREAL CWeightedDegreeCharKernel::compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	DREAL sum = 0.0 ;
	
	for (INT i=0; i<alen; i++)
	{
		DREAL sumi = 0.0 ;
		
		for (INT j=0; (i+j<alen) && (j<degree); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j];
		}
		if (position_weights!=NULL)
			sum+=position_weights[i]*sumi ;
		else
			sum+=sumi ;
	}
	return sum ;
}

DREAL CWeightedDegreeCharKernel::compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen)
{
	DREAL sum = 0.0 ;

	for (INT i=0; i<alen; i++)
	{
		DREAL sumi=0.0 ;
		for (INT j=0; (i+j<alen) && (j<degree); j++)
		{
			if (avec[i+j]!=bvec[i+j])
				break;
			sumi += weights[i*degree+j];
		}
		if (position_weights!=NULL)
			sum += position_weights[i]*sumi ;
		else
			sum += sumi ;
	}

	return sum ;
}


DREAL CWeightedDegreeCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  ASSERT(alen==blen);

  DREAL sqrt_a=1;
  DREAL sqrt_b=1;

  if (initialized && use_normalization)
  {
	  sqrt_a=sqrtdiag_lhs[idx_a];
	  sqrt_b=sqrtdiag_rhs[idx_b];
  }

  DREAL sqrt_both=sqrt_a*sqrt_b;

  double result=0;

  if (max_mismatch == 0 && length == 0 && block_computation)
	  result = compute_using_block(avec, alen, bvec, blen) ;
  else
  {
	  if (max_mismatch > 0)
		  result = compute_with_mismatch(avec, alen, bvec, blen) ;
	  else if (length==0)
		  result = compute_without_mismatch(avec, alen, bvec, blen) ;
	  else
		  result = compute_without_mismatch_matrix(avec, alen, bvec, blen) ;
  }
  
  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (double) result/sqrt_both;
}


void CWeightedDegreeCharKernel::add_example_to_tree(INT idx, DREAL alpha) 
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
	
	if (length == 0 || max_mismatch > 0)
	{
		for (INT i=0; i<len; i++)
		{
			DREAL alpha_pw = alpha ;
			/*if (position_weights!=NULL)
			  alpha_pw *= position_weights[i] ;*/
			if (alpha_pw==0.0)
				continue ;
			tries.add_to_trie(i, 0, vec, alpha_pw, weights, (length!=0)) ;
		}
	}
	else
	{
		for (INT i=0; i<len; i++)
		{
			DREAL alpha_pw = alpha ;
			/*if (position_weights!=NULL) 
			  alpha_pw = alpha*position_weights[i] ;*/
			if (alpha_pw==0.0)
				continue ;
			tries.add_to_trie(i, 0, vec, alpha_pw, weights, (length!=0)) ;		
		}
	}
	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreeCharKernel::add_example_to_single_tree(INT idx, DREAL alpha, INT tree_num) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	if (use_normalization)
		alpha /=  sqrtdiag_lhs[idx] ;
	
	for (INT i=tree_num; i<tree_num+degree && i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	if (length == 0 || max_mismatch > 0)
	{
		DREAL alpha_pw = alpha ;
		/*if (position_weights!=NULL)
		  alpha_pw = alpha*position_weights[tree_num] ;*/
		if (alpha_pw!=0.0)
			tries.add_to_trie(tree_num, 0, vec, alpha_pw, weights, (length!=0)) ;
	}
	else
	{
		DREAL alpha_pw = alpha ;
		/*if (position_weights!=NULL) 
		  alpha_pw = alpha*position_weights[tree_num] ;*/
		if (alpha_pw!=0.0)
			tries.add_to_trie(tree_num, 0, vec, alpha_pw, weights, (length!=0)) ;
	}
	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreeCharKernel::add_example_to_tree_mismatch(INT idx, DREAL alpha) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);
	
	INT *vec = new INT[len] ;
	
	if (use_normalization)
		alpha /=  sqrtdiag_lhs[idx] ;
	
	for (INT i=0; i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	/*for (INT q=0; q<40; q++)
	  fprintf(stderr, "w[%i]=%f\n", q,weights[q]) ;*/
	
	for (INT i=0; i<len; i++)
	{
		DREAL alpha_pw = alpha ;
		/*if (position_weights!=NULL)
		  alpha_pw = alpha*position_weights[i] ;*/
		if (alpha_pw==0.0)
			continue ;
		tries.add_example_to_tree_mismatch_recursion(NO_CHILD, i, alpha_pw, &vec[i], len-i, 0, 0, max_mismatch, weights) ;
	}
	
	
	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreeCharKernel::add_example_to_single_tree_mismatch(INT idx, DREAL alpha, INT tree_num) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) lhs)->get_feature_vector(idx, len, free);

	INT *vec = new INT[len] ;

	if (use_normalization)
		alpha /=  sqrtdiag_lhs[idx] ;

	for (INT i=tree_num; i<len && i<tree_num+degree; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

	DREAL alpha_pw = alpha ;
	/*if (position_weights!=NULL)
	  alpha_pw = alpha*position_weights[tree_num] ;*/
	if (alpha_pw!=0.0)
		tries.add_example_to_tree_mismatch_recursion(NO_CHILD, tree_num, alpha_pw, &vec[tree_num], len-tree_num, 0, 0, max_mismatch, weights) ;

	((CCharFeatures*) lhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	tree_initialized=true ;
}


DREAL CWeightedDegreeCharKernel::compute_by_tree(INT idx) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);
	ASSERT(char_vec && len>0);
	INT *vec = new INT[len] ;
	
	for (INT i=0; i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

	DREAL sum=0 ;
	for (INT i=0; i<len; i++)
		sum += tries.compute_by_tree_helper(vec, len, i, i, i, weights, (length!=0));
	
	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
	
	if (use_normalization)
		return sum/sqrtdiag_rhs[idx];
	else
		return sum;
}

void CWeightedDegreeCharKernel::compute_by_tree(INT idx, DREAL* LevelContrib) 
{
	INT len ;
	bool free ;
	CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(idx, len, free);

	INT *vec = new INT[len] ;

	for (INT i=0; i<len; i++)
		vec[i]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

	DREAL factor = 1.0 ;
	if (use_normalization)
		factor = 1.0/sqrtdiag_rhs[idx] ;

	for (INT i=0; i<len; i++)
	  tries.compute_by_tree_helper(vec, len, i, i, i, LevelContrib, factor, mkl_stepsize, weights, (length!=0));

	((CCharFeatures*) rhs)->free_feature_vector(char_vec, idx, free);
	delete[] vec ;
}

DREAL *CWeightedDegreeCharKernel::compute_abs_weights(int &len) 
{
	return tries.compute_abs_weights(len) ;
}

bool CWeightedDegreeCharKernel::set_wd_weights_by_type(EWDKernType p_type)
{
	ASSERT(degree>0);
	ASSERT(p_type==E_WD); /// if we know a better weighting later on do a switch

	delete[] weights;
	weights=new DREAL[degree];
	if (weights)
	{
		INT i;
		DREAL sum=0;
		for (i=0; i<degree; i++)
		{
			weights[i]=degree-i;
			sum+=weights[i];
		}
		for (i=0; i<degree; i++)
			weights[i]/=sum;

		for (i=0; i<degree; i++)
		{
			for (INT j=1; j<=max_mismatch; j++)
			{
				if (j<i+1)
				{
					INT nk=CMath::nchoosek(i+1, j);
					weights[i+j*degree]=weights[i]/(nk*pow(3,j));
				}
				else
					weights[i+j*degree]= 0;
			}
		}

		if (which_degree>=0)
		{
			ASSERT(which_degree<degree);
			for (i=0; i<degree; i++)
			{
				if (i!=which_degree)
					weights[i]=0;
				else
					weights[i]=1;
			}
		}
		return true;
	}
	else
		return false;
}

bool CWeightedDegreeCharKernel::set_weights(DREAL* ws, INT d, INT len)
{
	SG_DEBUG( "degree = %i  d=%i\n", degree, d) ;
	degree = d ;
	tries.set_degree(d);
	length=len;
	
	if (len == 0)
		len=1;
	
	delete[] weights;
	weights=new DREAL[d*len];
	
	if (weights)
	{
		for (int i=0; i<degree*len; i++)
			weights[i]=ws[i];
		return true;
	}
	else
		return false;
}

bool CWeightedDegreeCharKernel::set_position_weights(DREAL* pws, INT len)
{
	if (len==0)
	{
		delete[] position_weights ;
		position_weights = NULL ;
		tries.set_position_weights(position_weights) ;
	}
	
    if (seq_length!=len) 
	{
      SG_ERROR( "seq_length = %i, position_weights_length=%i\n", seq_length, len) ;
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

bool CWeightedDegreeCharKernel::init_block_weights_from_wd()
{
	delete[] block_weights;
	block_weights=new DREAL[CMath::max(seq_length,degree)];

	if (block_weights)
	{
		double deg=degree;
		INT k;
		for (k=0; k<degree ; k++)
			block_weights[k]=(-pow(k,3) + (3*deg-3)*pow(k,2) + (9*deg-2)*k + 6*deg) / (3*deg*(deg+1));
		for (k=degree; k<seq_length ; k++)
			block_weights[k]=(-deg+3*k+4)/3;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_from_wd_external()
{
	ASSERT(weights);
	delete[] block_weights;
	block_weights=new DREAL[CMath::max(seq_length,degree)];

	if (block_weights)
	{
		INT i=0;
		block_weights[0]=weights[0];
		for (i=1; i<CMath::max(seq_length,degree); i++)
			block_weights[i]=0;

		for (i=1; i<CMath::max(seq_length,degree); i++)
		{
			block_weights[i]=block_weights[i-1];

			DREAL contrib=0;
			for (INT j=0; j<CMath::min(degree,i+1); j++)
				contrib+=weights[j];

			block_weights[i]+=contrib;
		}
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_const()
{
	delete[] block_weights;
	block_weights=new DREAL[seq_length];

	if (block_weights)
	{
		for (int i=1; i<seq_length+1 ; i++)
			block_weights[i-1]=1.0/seq_length;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_linear()
{
	delete[] block_weights;
	block_weights=new DREAL[seq_length];

	if (block_weights)
	{
		for (int i=1; i<seq_length+1 ; i++)
			block_weights[i-1]=degree*i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_sqpoly()
{
	delete[] block_weights;
	block_weights=new DREAL[seq_length];

	if (block_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			block_weights[i-1]=((double) i)*i;

		for (int i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_cubicpoly()
{
	delete[] block_weights;
	block_weights=new DREAL[seq_length];

	if (block_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			block_weights[i-1]=((double) i)*i*i;

		for (int i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_exp()
{
	delete[] block_weights;
	block_weights=new DREAL[seq_length];

	if (block_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			block_weights[i-1]=exp(((double) i/10.0));

		for (int i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_log()
{
	delete[] block_weights;
	block_weights=new DREAL[seq_length];

	if (block_weights)
	{
		for (int i=1; i<degree+1 ; i++)
			block_weights[i-1]=pow(log(i),2);

		for (int i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i-degree+1+pow(log(degree+1),2);
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights_external()
{
	if (block_weights_external && (seq_length == num_block_weights_external) )
	{
		delete[] block_weights;
		block_weights=new DREAL[seq_length];

		if (block_weights)
		{
			for (int i=0; i<seq_length; i++)
				block_weights[i]=block_weights_external[i];
		}
	}
	else {
      SG_ERROR( "sequence longer then weights (seqlen:%d, wlen:%d)\n", seq_length, block_weights_external);
   }
	return (block_weights!=NULL);
}

bool CWeightedDegreeCharKernel::init_block_weights()
{
	switch (type)
	{
		case E_WD:
			return init_block_weights_from_wd();
		case E_EXTERNAL:
			return init_block_weights_from_wd_external();
		case E_BLOCK_CONST:
			return init_block_weights_const();
		case E_BLOCK_LINEAR:
			return init_block_weights_linear();
		case E_BLOCK_SQPOLY:
			return init_block_weights_sqpoly();
		case E_BLOCK_CUBICPOLY:
			return init_block_weights_cubicpoly();
		case E_BLOCK_EXP:
			return init_block_weights_exp();
		case E_BLOCK_LOG:
			return init_block_weights_log();
		case E_BLOCK_EXTERNAL:
			return init_block_weights_external();
		default:
			return false;
	};
}


void* CWeightedDegreeCharKernel::compute_batch_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;
	INT j=params->j;
	CWeightedDegreeCharKernel* wd=params->kernel;
	CTrie* tries=params->tries;
	DREAL* weights=params->weights;
	INT length=params->length;
	DREAL* sqrtdiag_rhs=params->sqrtdiag_rhs;
	INT* vec=params->vec;
	DREAL* result=params->result;
	DREAL factor=params->factor;
	INT* vec_idx=params->vec_idx;

	for (INT i=params->start; i<params->end; i++)
	{
		INT len=0;
		bool freevec;
		CHAR* char_vec=((CCharFeatures*) wd->get_rhs())->get_feature_vector(vec_idx[i], len, freevec);
		for (INT k=j; k<CMath::min(len,j+wd->get_degree()); k++)
			vec[k]=((CCharFeatures*) wd->get_lhs())->get_alphabet()->remap_to_bin(char_vec[k]);

		if (wd->get_use_normalization())
			result[i] += factor*tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0))/sqrtdiag_rhs[vec_idx[i]];
		else
			result[i] += factor*tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0));

		((CCharFeatures*) wd->get_rhs())->free_feature_vector(char_vec, i, freevec);
	}

	return NULL;
}

void CWeightedDegreeCharKernel::compute_batch(INT num_vec, INT* vec_idx, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor)
{
	ASSERT(get_rhs());
    ASSERT(num_vec<=get_rhs()->get_num_vectors());
	ASSERT(num_vec>0);
	ASSERT(vec_idx);
	ASSERT(result);

	INT num_feat=((CCharFeatures*) get_rhs())->get_num_features();
	ASSERT(num_feat>0);
	INT num_threads=parallel.get_num_threads();
	ASSERT(num_threads>0);
	INT* vec= new INT[num_threads*num_feat];
	ASSERT(vec);

	if (num_threads < 2)
	{
#ifdef CYGWIN
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
			params.sqrtdiag_rhs=sqrtdiag_rhs;
			params.vec_idx=vec_idx;
			compute_batch_helper((void*) &params);

			SG_PROGRESS(j,0,num_feat);
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
				params[t].vec_idx=vec_idx;
				params[t].sqrtdiag_rhs=sqrtdiag_rhs;
				pthread_create(&threads[t], NULL, CWeightedDegreeCharKernel::compute_batch_helper, (void*)&params[t]);
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
			params[t].vec_idx=vec_idx;
			params[t].sqrtdiag_rhs=sqrtdiag_rhs;
			compute_batch_helper((void*) &params[t]);

			for (t=0; t<num_threads-1; t++)
				pthread_join(threads[t], NULL);
			SG_PROGRESS(j,0,num_feat);
		}
	}
#endif

	delete[] vec;
}



DREAL* CWeightedDegreeCharKernel::compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas)
{
    num_feat=((CCharFeatures*) get_rhs())->get_num_features();
    ASSERT(num_feat>0);
    ASSERT(((CCharFeatures*) get_rhs())->get_alphabet()->get_alphabet() == DNA);
    num_sym=4; //for now works only w/ DNA

    // variables
    INT* nofsKmers = new INT[ max_degree ];
    DREAL** C = new DREAL*[ max_degree ];
    DREAL** L = new DREAL*[ max_degree ];
    DREAL** R = new DREAL*[ max_degree ];
    INT i;
    INT k;

    // return table
    INT bigtabSize = 0;
    for( k = 0; k < max_degree; ++k ) {
		nofsKmers[k] = (INT) pow( num_sym, k+1 );
        const INT tabSize = nofsKmers[k] * num_feat;
        bigtabSize += tabSize;
    }
    result= new DREAL[ bigtabSize ];
	
    // auxilliary tables
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
	
    // tree parsing info
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
	
	bool orig_use_compact_terminal_nodes = tries.get_use_compact_terminal_nodes() ;
	tries.set_use_compact_terminal_nodes(false) ;
	
    // main loop
    i = 0; // total progress
    for( k = 0; k < max_degree; ++k )
    {
		const INT nofKmers = nofsKmers[ k ];
		info.C_k = C[k];
		info.L_k = L[k];
		info.R_k = R[k];
		
		// run over all trees
		for(INT p = 0; p < num_feat; ++p )
		{
			init_optimization( num_suppvec, IDX, alphas, p );
			INT tree = p ;
			for(INT j = 0; j < degree+1; j++ ) {
				x[j] = -1;
			}
			tries.traverse( tree, p, info, 0, x, k );
			SG_PROGRESS(i++,0,num_feat*max_degree);
	}
		
		// add partial overlap scores
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

	tries.set_use_compact_terminal_nodes(orig_use_compact_terminal_nodes) ;
	
    // return a vector
    num_feat=1;
    num_sym = bigtabSize;
    // clean up
    delete[] nofsKmers;
    delete[] margFactors;
    delete[] substrs;
    delete[] x;
    delete[] C;
    for( k = 0; k < max_degree; ++k ) {
		delete L[k];
		delete R[k];
    }
    delete[] L;
    delete[] R;
    return result;
}


/*


DREAL* CWeightedDegreeCharKernel::compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas)
{
	num_feat=((CCharFeatures*) get_rhs())->get_num_features();
	ASSERT(num_feat>0);
	ASSERT(((CCharFeatures*) get_rhs())->get_alphabet()->get_alphabet() == DNA);
	num_sym=4; //for now works only w/ DNA
	INT sym_offset=(INT) pow(num_sym,max_degree);

	if (!result)
	{
		INT buflen=(INT) num_feat*sym_offset;
		result= new DREAL[buflen];
		ASSERT(result);
		memset(result, 0, sizeof(DREAL)*buflen);
	}

	for (INT i=0; i<num_feat; i++)
	{
		//init_optimization(num_suppvec, IDX, alphas, i, CMath::min(num_feat-1,i+1));
		init_optimization(num_suppvec, IDX, alphas, i);

		tries.compute_scoring_helper(NO_CHILD, i, 0, 0.0, 0, max_degree, num_feat, num_sym, sym_offset, 0, result);
		SG_PROGRESS(i,0,num_feat);
	}
	num_sym=sym_offset;

	return result;
}
*/
