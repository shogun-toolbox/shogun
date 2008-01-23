/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Signal.h"
#include "lib/Trie.h"
#include "base/Parallel.h"

#include "kernel/WeightedDegreeStringKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"

#ifndef WIN32
#include <pthread.h>
#endif

struct S_THREAD_PARAM
{
	INT* vec;
	DREAL* result;
	DREAL* weights;
	CWeightedDegreeStringKernel* kernel;
	CTrie<DNATrie>* tries;
	DREAL factor;
	INT j;
	INT start;
	INT end;
	INT length;
	INT* vec_idx;
};

CWeightedDegreeStringKernel::CWeightedDegreeStringKernel (
	INT degree_, EWDKernType type_)
: CStringKernel<CHAR>(10),weights(NULL),position_weights(NULL),
	weights_buffer(NULL), mkl_stepsize(1),degree(degree_), length(0),
	max_mismatch(0), seq_length(0), initialized(false),
	block_computation(true), use_normalization(true),
	normalization_const(1.0), num_block_weights_external(0),
	block_weights_external(NULL), block_weights(NULL), type(type_),
	which_degree(-1), tries(NULL), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	lhs=NULL;
	rhs=NULL;

	if (type!=E_EXTERNAL)
		set_wd_weights_by_type(type);
}

CWeightedDegreeStringKernel::CWeightedDegreeStringKernel (
	DREAL *weights_, INT degree_)
: CStringKernel<CHAR>(10), weights(NULL), position_weights(NULL),
	weights_buffer(NULL), mkl_stepsize(1), degree(degree_), length(0),
	max_mismatch(0), seq_length(0), initialized(false),
	block_computation(true), use_normalization(true),
	normalization_const(1.0), num_block_weights_external(0),
	block_weights_external(NULL), block_weights(NULL), type(E_EXTERNAL),
	which_degree(-1), tries(NULL), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	lhs=NULL;
	rhs=NULL;

	weights=new DREAL[degree*(1+max_mismatch)];
	ASSERT(weights!=NULL);
	for (INT i=0; i<degree*(1+max_mismatch); i++)
		weights[i]=weights_[i];
}

CWeightedDegreeStringKernel::CWeightedDegreeStringKernel(
	CStringFeatures<CHAR>* l, CStringFeatures<CHAR>* r, INT degree_)
: CStringKernel<CHAR>(10), weights(NULL), position_weights(NULL),
	weights_buffer(NULL), mkl_stepsize(1), degree(degree_), length(0),
	max_mismatch(0), seq_length(0), initialized(false),
	block_computation(true), use_normalization(true),
	normalization_const(1.0), num_block_weights_external(0),
	block_weights_external(NULL), block_weights(NULL), type(E_WD),
	which_degree(-1), tries(NULL), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;

	set_wd_weights_by_type(type);
	init(l, r);
}

CWeightedDegreeStringKernel::~CWeightedDegreeStringKernel()
{
	cleanup();

	delete[] weights;
	weights=NULL;

	delete[] position_weights ;
	position_weights=NULL ;

	delete[] weights_buffer ;
	weights_buffer = NULL ;
}


void CWeightedDegreeStringKernel::remove_lhs()
{ 
	SG_DEBUG( "deleting CWeightedDegreeStringKernel optimization\n");
	delete_optimization();

#ifdef SVMLIGHT
	if (lhs)
		cache_reset();
#endif

	lhs = NULL ; 
	rhs = NULL ; 
	initialized = false ;

	tries->destroy() ;
}

void CWeightedDegreeStringKernel::remove_rhs()
{
#ifdef SVMLIGHT
	if (rhs)
		cache_reset() ;
#endif
	rhs = lhs ;
}

void CWeightedDegreeStringKernel::create_empty_tries()
{
	seq_length=((CStringFeatures<CHAR>*) lhs)->get_max_vector_length();

	tries->destroy() ;
	tries->create(seq_length, max_mismatch==0) ;
}
  
bool CWeightedDegreeStringKernel::init(CFeatures* l, CFeatures* r)
{
	INT lhs_changed = (lhs!=l) ;
	INT rhs_changed = (rhs!=r) ;

	bool result=CStringKernel<CHAR>::init(l,r);
	initialized=false;

	SG_DEBUG("lhs_changed: %i\n", lhs_changed);
	SG_DEBUG("rhs_changed: %i\n", rhs_changed);

	ASSERT(((((CStringFeatures<CHAR>*) l)->get_alphabet()->get_alphabet()==DNA) || 
				 (((CStringFeatures<CHAR>*) l)->get_alphabet()->get_alphabet()==RNA)));
	ASSERT(((((CStringFeatures<CHAR>*) r)->get_alphabet()->get_alphabet()==DNA) || 
				 (((CStringFeatures<CHAR>*) r)->get_alphabet()->get_alphabet()==RNA)));

	if (tries!=NULL) {
		tries->delete_trees(max_mismatch==0);
		delete tries;
	}
	tries=new CTrie<DNATrie>(degree, max_mismatch==0);
	ASSERT(tries);
	create_empty_tries();

	init_block_weights();

	if (use_normalization)
		normalization_const=block_weights[seq_length-1];
	else
		normalization_const=1.0;
	
	this->lhs=(CStringFeatures<CHAR>*) l;
	this->rhs=(CStringFeatures<CHAR>*) r;

	initialized=true;
	return result;
}

void CWeightedDegreeStringKernel::cleanup()
{
	SG_DEBUG("deleting CWeightedDegreeStringKernel optimization\n");
	delete_optimization();

	delete[] block_weights;
	block_weights=NULL;

	tries->destroy();
	delete tries;
	tries=NULL;

	lhs=NULL;
	rhs=NULL;

	seq_length=0;
	initialized=false;
	tree_initialized = false;
}

bool CWeightedDegreeStringKernel::load_init(FILE* src)
{
	return false;
}

bool CWeightedDegreeStringKernel::save_init(FILE* dest)
{
	return false;
}
  

bool CWeightedDegreeStringKernel::init_optimization(INT count, INT* IDX, DREAL* alphas, INT tree_num)
{
	if (tree_num<0)
		SG_DEBUG( "deleting CWeightedDegreeStringKernel optimization\n");
	delete_optimization();

	if (tree_num<0)
		SG_DEBUG( "initializing CWeightedDegreeStringKernel optimization\n") ;

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

bool CWeightedDegreeStringKernel::delete_optimization()
{ 
	if (get_is_initialized())
	{
		tries->delete_trees(max_mismatch==0); 
		set_is_initialized(false);
		return true;
	}
	
	return false;
}


DREAL CWeightedDegreeStringKernel::compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen)
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

DREAL CWeightedDegreeStringKernel::compute_using_block(CHAR* avec, INT alen, CHAR* bvec, INT blen)
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

DREAL CWeightedDegreeStringKernel::compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen)
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

DREAL CWeightedDegreeStringKernel::compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen)
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


DREAL CWeightedDegreeStringKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;

  CHAR* avec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
  CHAR* bvec=((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

  // can only deal with strings of same length
  ASSERT(alen==blen);

  DREAL result=0;

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
  
  return result/normalization_const;
}


void CWeightedDegreeStringKernel::add_example_to_tree(INT idx, DREAL alpha)
{
	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, len);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	for (INT i=0; i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	if (length == 0 || max_mismatch > 0)
	{
		for (INT i=0; i<len; i++)
		{
			DREAL alpha_pw = alpha ;
			/*if (position_weights!=NULL)
			  alpha_pw *= position_weights[i] ;*/
			if (alpha_pw==0.0)
				continue ;
			tries->add_to_trie(i, 0, vec, alpha_pw, weights, (length!=0)) ;
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
			tries->add_to_trie(i, 0, vec, alpha_pw, weights, (length!=0)) ;		
		}
	}
	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreeStringKernel::add_example_to_single_tree(INT idx, DREAL alpha, INT tree_num) 
{
	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, len);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	for (INT i=tree_num; i<tree_num+degree && i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	if (length == 0 || max_mismatch > 0)
	{
		DREAL alpha_pw = alpha ;
		/*if (position_weights!=NULL)
		  alpha_pw = alpha*position_weights[tree_num] ;*/
		if (alpha_pw!=0.0)
			tries->add_to_trie(tree_num, 0, vec, alpha_pw, weights, (length!=0)) ;
	}
	else
	{
		DREAL alpha_pw = alpha ;
		/*if (position_weights!=NULL) 
		  alpha_pw = alpha*position_weights[tree_num] ;*/
		if (alpha_pw!=0.0)
			tries->add_to_trie(tree_num, 0, vec, alpha_pw, weights, (length!=0)) ;
	}
	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreeStringKernel::add_example_to_tree_mismatch(INT idx, DREAL alpha)
{
	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, len);
	
	INT *vec = new INT[len] ;
	
	for (INT i=0; i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);
	
	/*for (INT q=0; q<40; q++)
	  fprintf(stderr, "w[%i]=%f\n", q,weights[q]) ;*/
	
	for (INT i=0; i<len; i++)
	{
		DREAL alpha_pw = alpha ;
		/*if (position_weights!=NULL)
		  alpha_pw = alpha*position_weights[i] ;*/
		if (alpha_pw==0.0)
			continue ;
		tries->add_example_to_tree_mismatch_recursion(NO_CHILD, i, alpha_pw, &vec[i], len-i, 0, 0, max_mismatch, weights) ;
	}
	
	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreeStringKernel::add_example_to_single_tree_mismatch(INT idx, DREAL alpha, INT tree_num)
{
	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, len);

	INT *vec = new INT[len] ;

	for (INT i=tree_num; i<len && i<tree_num+degree; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

	DREAL alpha_pw = alpha ;
	/*if (position_weights!=NULL)
	  alpha_pw = alpha*position_weights[tree_num] ;*/
	if (alpha_pw!=0.0)
		tries->add_example_to_tree_mismatch_recursion(NO_CHILD, tree_num, alpha_pw, &vec[tree_num], len-tree_num, 0, 0, max_mismatch, weights) ;

	delete[] vec ;
	tree_initialized=true ;
}


DREAL CWeightedDegreeStringKernel::compute_by_tree(INT idx)
{
	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx, len);
	ASSERT(char_vec && len>0);
	INT *vec = new INT[len] ;
	
	for (INT i=0; i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

	DREAL sum=0 ;
	for (INT i=0; i<len; i++)
		sum += tries->compute_by_tree_helper(vec, len, i, i, i, weights, (length!=0));
	
	delete[] vec ;
	
	return sum/normalization_const;
}

void CWeightedDegreeStringKernel::compute_by_tree(INT idx, DREAL* LevelContrib)
{
	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx, len);

	INT *vec = new INT[len] ;

	for (INT i=0; i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

	for (INT i=0; i<len; i++)
	  tries->compute_by_tree_helper(vec, len, i, i, i, LevelContrib, 1.0/normalization_const, mkl_stepsize, weights, (length!=0));

	delete[] vec ;
}

DREAL *CWeightedDegreeStringKernel::compute_abs_weights(int &len)
{
	return tries->compute_abs_weights(len) ;
}

bool CWeightedDegreeStringKernel::set_wd_weights_by_type(EWDKernType p_type)
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

bool CWeightedDegreeStringKernel::set_weights(DREAL* ws, INT d, INT len)
{
	SG_DEBUG("degree = %i  d=%i\n", degree, d);
	degree=d;
	tries->set_degree(degree);
	length=len;
	
	if (length==0) length=1;
	INT num_weights=degree*(length+max_mismatch);
	delete[] weights;
	weights=new DREAL[num_weights];
	if (weights)
	{
		for (INT i=0; i<num_weights; i++) {
			if (ws[i]) // len(ws) might be != num_weights?
				weights[i]=ws[i];
		}
		return true;
	}
	else
		return false;
}

bool CWeightedDegreeStringKernel::set_position_weights(DREAL* pws, INT len)
{
	if (len==0)
	{
		delete[] position_weights ;
		position_weights = NULL ;
		tries->set_position_weights(position_weights) ;
	}
	
    if (seq_length!=len) 
	{
      SG_ERROR( "seq_length = %i, position_weights_length=%i\n", seq_length, len) ;
		return false ;
	}
	delete[] position_weights;
	position_weights=new DREAL[len];
	tries->set_position_weights(position_weights) ;
	
	if (position_weights)
	{
		for (int i=0; i<len; i++)
			position_weights[i]=pws[i];
		return true;
	}
	else
		return false;
}

bool CWeightedDegreeStringKernel::init_block_weights_from_wd()
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

bool CWeightedDegreeStringKernel::init_block_weights_from_wd_external()
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

bool CWeightedDegreeStringKernel::init_block_weights_const()
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

bool CWeightedDegreeStringKernel::init_block_weights_linear()
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

bool CWeightedDegreeStringKernel::init_block_weights_sqpoly()
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

bool CWeightedDegreeStringKernel::init_block_weights_cubicpoly()
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

bool CWeightedDegreeStringKernel::init_block_weights_exp()
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

bool CWeightedDegreeStringKernel::init_block_weights_log()
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

bool CWeightedDegreeStringKernel::init_block_weights_external()
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

bool CWeightedDegreeStringKernel::init_block_weights()
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


void* CWeightedDegreeStringKernel::compute_batch_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;
	INT j=params->j;
	CWeightedDegreeStringKernel* wd=params->kernel;
	CTrie<DNATrie>* tries=params->tries;
	DREAL* weights=params->weights;
	INT length=params->length;
	INT* vec=params->vec;
	DREAL* result=params->result;
	DREAL factor=params->factor;
	INT* vec_idx=params->vec_idx;

	for (INT i=params->start; i<params->end; i++)
	{
		INT len=0;
		CHAR* char_vec=((CStringFeatures<CHAR>*) wd->get_rhs())->get_feature_vector(vec_idx[i], len);
		for (INT k=j; k<CMath::min(len,j+wd->get_degree()); k++)
			vec[k]=((CStringFeatures<CHAR>*) wd->get_lhs())->get_alphabet()->remap_to_bin(char_vec[k]);

		result[i] += factor*tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0))/wd->get_normalization_const();
	}

	return NULL;
}

void CWeightedDegreeStringKernel::compute_batch(INT num_vec, INT* vec_idx, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor)
{
	ASSERT(get_rhs());
    ASSERT(num_vec<=get_rhs()->get_num_vectors());
	ASSERT(num_vec>0);
	ASSERT(vec_idx);
	ASSERT(result);

	INT num_feat=((CStringFeatures<CHAR>*) get_rhs())->get_max_vector_length();
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
			params.tries=tries;
			params.factor=factor;
			params.j=j;
			params.start=0;
			params.end=num_vec;
			params.length=length;
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
				params[t].tries=tries;
				params[t].factor=factor;
				params[t].j=j;
				params[t].start = t*step;
				params[t].end = (t+1)*step;
				params[t].length=length;
				params[t].vec_idx=vec_idx;
				pthread_create(&threads[t], NULL, CWeightedDegreeStringKernel::compute_batch_helper, (void*)&params[t]);
			}
			params[t].vec=&vec[num_feat*t];
			params[t].result=result;
			params[t].weights=weights;
			params[t].kernel=this;
			params[t].tries=tries;
			params[t].factor=factor;
			params[t].j=j;
			params[t].start=t*step;
			params[t].end=num_vec;
			params[t].length=length;
			params[t].vec_idx=vec_idx;
			compute_batch_helper((void*) &params[t]);

			for (t=0; t<num_threads-1; t++)
				pthread_join(threads[t], NULL);
			SG_PROGRESS(j,0,num_feat);
		}
	}
#endif

	delete[] vec;

	//really also free memory as this can be huge on testing especially when
	//using the combined kernel
	create_empty_tries();
}

bool CWeightedDegreeStringKernel::set_max_mismatch(INT max)
{
	if (type==E_EXTERNAL && max!=0) {
		return false;
	}

	max_mismatch=max;

	if (lhs!=NULL && rhs!=NULL)
		return init(lhs, rhs);
	else
		return true;
}

