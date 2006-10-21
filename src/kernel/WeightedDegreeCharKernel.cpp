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
#include "lib/Trie.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"

CWeightedDegreeCharKernel::CWeightedDegreeCharKernel(INT size, INT max_mismatch_, bool use_norm, bool block, INT mkl_stepsize_)
	: CSimpleKernel<CHAR>(size),weights(NULL),position_weights(NULL),
	  weights_buffer(NULL), mkl_stepsize(mkl_stepsize_),degree(0), length(0), 
	  max_mismatch(max_mismatch_), seq_length(0), 
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), 
	  block_computation(block), use_normalization(use_norm), tries(0), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	lhs=NULL;
	rhs=NULL;
	matching_weights=NULL;
}

CWeightedDegreeCharKernel::CWeightedDegreeCharKernel(INT size, double* w, INT d, INT max_mismatch_, bool use_norm, bool block, INT mkl_stepsize_)
	: CSimpleKernel<CHAR>(size),weights(NULL),position_weights(NULL),weights_buffer(NULL), mkl_stepsize(mkl_stepsize_), degree(d), length(0), 
	  max_mismatch(max_mismatch_), seq_length(0), 
	  sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), 
	  block_computation(block), use_normalization(use_norm), tries(d)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	lhs=NULL;
	rhs=NULL;
	matching_weights=NULL;

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

  
bool CWeightedDegreeCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	INT lhs_changed = (lhs!=l) ;
	INT rhs_changed = (rhs!=r) ;

	CIO::message(M_DEBUG, "lhs_changed: %i\n", lhs_changed);
	CIO::message(M_DEBUG, "rhs_changed: %i\n", rhs_changed);

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
		tries.create(alen) ;
	} 

	bool result=CSimpleKernel<CHAR>::init(l,r,do_init);
	initialized = false ;
	INT i;

	init_matching_weights_wd();

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
	delete_optimization();

	delete[] matching_weights;
	matching_weights=NULL;

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
	delete_optimization();
	
	CIO::message(M_DEBUG, "initializing CWeightedDegreeCharKernel optimization\n") ;
	int i=0;
	for (i=0; i<count; i++)
	{
		if (tree_num<0)
		{
			if ( (i % (count/10+1)) == 0)
				CIO::progress(i, 0, count);
			
			if (max_mismatch==0)
				add_example_to_tree(IDX[i], alphas[i]) ;
			else
				add_example_to_tree_mismatch(IDX[i], alphas[i]) ;

			CIO::message(M_DEBUG, "number of used trie nodes: %i\n", tries.get_num_used_nodes()) ;
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
		CIO::message(M_MESSAGEONLY, "done.           \n");
	
	set_is_initialized(true) ;
	return true ;
}

bool CWeightedDegreeCharKernel::delete_optimization() 
{ 
	CIO::message(M_DEBUG, "deleting CWeightedDegreeCharKernel optimization\n");

	if (get_is_initialized())
	{
		tries.delete_trees(); 
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

	for (INT i=0; i<alen; i++)
	{
		if (avec[i]==bvec[i])
			match_len++;
		else
		{
			if (match_len>=0)
				sum+=matching_weights[match_len];
			match_len=-1;
		}
	}

	if (match_len>=0)
		sum+=matching_weights[match_len];

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
			if (position_weights!=NULL)
				alpha_pw *= position_weights[i] ;
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
			if (position_weights!=NULL) 
				alpha_pw = alpha*position_weights[i] ;
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
		if (position_weights!=NULL)
			alpha_pw = alpha*position_weights[tree_num] ;
		if (alpha_pw!=0.0)
			tries.add_to_trie(tree_num, 0, vec, alpha_pw, weights, (length!=0)) ;
	}
	else
	{
		DREAL alpha_pw = alpha ;
		if (position_weights!=NULL) 
			alpha_pw = alpha*position_weights[tree_num] ;
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

	for (INT i=0; i<len; i++)
	{
		DREAL alpha_pw = alpha ;
		if (position_weights!=NULL)
			alpha_pw = alpha*position_weights[i] ;
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
	if (position_weights!=NULL)
		alpha_pw = alpha*position_weights[tree_num] ;
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

bool CWeightedDegreeCharKernel::set_weights(DREAL* ws, INT d, INT len)
{
	CIO::message(M_DEBUG, "degree = %i  d=%i\n", degree, d) ;
	degree = d ;
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
	}
	
    if (seq_length!=len) 
	{
		CIO::message(M_ERROR, "seq_length = %i, position_weights_length=%i\n", seq_length, len) ;
		return false ;
	}
	delete[] position_weights;
	position_weights=new DREAL[len];
	
	if (position_weights)
	{
		for (int i=0; i<len; i++)
			position_weights[i]=pws[i];
		return true;
	}
	else
		return false;
}

bool CWeightedDegreeCharKernel::init_matching_weights_wd()
{
	delete[] matching_weights;
	matching_weights=new DREAL[seq_length];

	if (matching_weights)
	{
		double deg=degree;
		INT k;
		for (k=0; k<degree ; k++)
			matching_weights[k]=(-pow(k,3) + (3*deg-3)*pow(k,2) + (9*deg-2)*k + 6*deg) / (3*deg*(deg+1));
		for (k=degree; k<seq_length ; k++)
			matching_weights[k]=(-deg+3*k+4)/3;
	}

	return (matching_weights!=NULL);
}


DREAL* CWeightedDegreeCharKernel::compute_batch(INT& num_vec, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor)
{
	ASSERT(get_rhs());
	num_vec=get_rhs()->get_num_vectors();
	ASSERT(num_vec>0);
	INT num_feat=((CCharFeatures*) get_rhs())->get_num_features();
	ASSERT(num_feat>0);

	if (!result)
	{
		result= new DREAL[num_vec];
		ASSERT(result);
		memset(result, 0, sizeof(DREAL)*num_vec);
	}

	INT* vec= new INT[num_feat];

	for (INT j=0; j<num_feat; j++)
	{
		init_optimization(num_suppvec, IDX, alphas, j);
		
		for (INT i=0; i<num_vec; i++)
		{
			INT len=0;
			bool freevec;
			CHAR* char_vec=((CCharFeatures*) rhs)->get_feature_vector(i, len, freevec);
			for (INT k=j; k<CMath::min(len,j+degree); k++)
				vec[k]=((CCharFeatures*) lhs)->get_alphabet()->remap_to_bin(char_vec[k]);

			if (use_normalization)
			  result[i] += factor*tries.compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0))/sqrtdiag_rhs[i];
			else
			  result[i] += factor*tries.compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0));

			((CCharFeatures*) rhs)->free_feature_vector(char_vec, i, freevec);
		}
		CIO::progress(j,0,num_feat);
	}

	delete[] vec;

	return result;
}




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
		CIO::progress(i,0,num_feat);
	}
	num_sym=sym_offset;

	return result;
}
