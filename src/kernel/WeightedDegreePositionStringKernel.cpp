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

#include "kernel/WeightedDegreePositionStringKernel.h"
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
	CWeightedDegreePositionStringKernel* kernel;
	CTrie* tries;
	DREAL factor;
	INT j;
	INT start;
	INT end;
	INT length;
	INT max_shift;
	INT* shift;
	INT* vec_idx;
};

CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel(INT size, INT d, 
		INT max_mismatch_, bool use_norm, INT mkl_stepsize_)
	: CStringKernel<CHAR>(size), weights(NULL), position_weights(NULL), 
	  position_weights_lhs(NULL), position_weights_rhs(NULL),
	  weights_buffer(NULL), mkl_stepsize(mkl_stepsize_), degree(d), length(0),
	  max_mismatch(max_mismatch_), seq_length(0), shift(NULL), shift_len(0),
	  initialized(false), use_normalization(use_norm),
	  normalization_const(1.0), num_block_weights_external(0),
	  block_weights_external(NULL), block_weights(NULL), type(E_EXTERNAL),
	  tries(d), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;
	set_wd_weights();
	ASSERT(weights);

}

CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel(INT size, DREAL* w, INT d, 
		INT max_mismatch_, INT * shift_, 
		INT shift_len_, bool use_norm,
		INT mkl_stepsize_)
	: CStringKernel<CHAR>(size), weights(NULL), position_weights(NULL), 
	  position_weights_lhs(NULL), position_weights_rhs(NULL),
	  weights_buffer(NULL), mkl_stepsize(mkl_stepsize_), degree(d), length(0),
	  max_mismatch(max_mismatch_), seq_length(0), shift(NULL),
	  initialized(false), use_normalization(use_norm),
	  normalization_const(1.0), num_block_weights_external(0),
	  block_weights_external(NULL), block_weights(NULL), type(E_EXTERNAL),
	  tries(d), tree_initialized(false)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;

	weights=new DREAL[d*(1+max_mismatch)];

	ASSERT(weights);
	for (INT i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];

	set_shifts(shift_, shift_len_);
}

CWeightedDegreePositionStringKernel::~CWeightedDegreePositionStringKernel() 
{
	cleanup();

	delete[] shift;
	shift = NULL;

	delete[] weights ;
	weights=NULL ;

	delete[] position_weights ;
	position_weights=NULL ;

	delete[] position_weights_lhs ;
	position_weights_lhs=NULL ;

	delete[] position_weights_rhs ;
	position_weights_rhs=NULL ;

	delete[] weights_buffer ;
	weights_buffer = NULL ;
}

void CWeightedDegreePositionStringKernel::remove_lhs() 
{ 
	SG_DEBUG( "deleting CWeightedDegreePositionStringKernel optimization\n");
	delete_optimization();

#ifdef USE_SVMLIGHT
	if (lhs)
		cache_reset() ;
#endif //USE_SVMLIGHT

	lhs = NULL ; 
	rhs = NULL ; 
	initialized = false ;

	tries.destroy() ;
} ;

void CWeightedDegreePositionStringKernel::remove_rhs()
{
#ifdef USE_SVMLIGHT
	if (rhs)
		cache_reset() ;
#endif //USE_SVMLIGHT

	rhs = lhs ;
}


bool CWeightedDegreePositionStringKernel::init(CFeatures* l, CFeatures* r)
{
	INT lhs_changed = (lhs!=l) ;
	INT rhs_changed = (rhs!=r) ;

	bool result=CStringKernel<CHAR>::init(l,r);

	SG_DEBUG( "lhs_changed: %i\n", lhs_changed) ;
	SG_DEBUG( "rhs_changed: %i\n", rhs_changed) ;

	ASSERT(((((CStringFeatures<CHAR>*) l)->get_alphabet()->get_alphabet()==DNA) || 
				(((CStringFeatures<CHAR>*) l)->get_alphabet()->get_alphabet()==RNA)));
	ASSERT(((((CStringFeatures<CHAR>*) r)->get_alphabet()->get_alphabet()==DNA) || 
				(((CStringFeatures<CHAR>*) r)->get_alphabet()->get_alphabet()==RNA)));

	if (lhs_changed) 
	{
		seq_length = ((CStringFeatures<CHAR>*) l)->get_max_vector_length();

		tries.destroy() ;
		if (opt_type==SLOWBUTMEMEFFICIENT)
			tries.create(seq_length, true); 
		else if (opt_type==FASTBUTMEMHUNGRY)
			tries.create(seq_length, false);  // still buggy
		else {
			SG_ERROR( "unknown optimization type\n");
		}

		init_block_weights();

		normalization_const=1.0;
		if (use_normalization)
			normalization_const=compute(0,0);
	} 

	SG_DEBUG( "use normalization:%d (const:%f)\n", (use_normalization) ? 1 : 0,
			normalization_const);


	initialized = true ;
	return result;
}

void CWeightedDegreePositionStringKernel::cleanup()
{
	SG_DEBUG( "deleting CWeightedDegreePositionStringKernel optimization\n");
	delete_optimization();

	tries.destroy() ;

	lhs = NULL;
	rhs = NULL;

	seq_length = 0;
	initialized = false;
	tree_initialized = false;
}

bool CWeightedDegreePositionStringKernel::load_init(FILE* src)
{
	return false;
}

bool CWeightedDegreePositionStringKernel::save_init(FILE* dest)
{
	return false;
}

bool CWeightedDegreePositionStringKernel::init_optimization(INT p_count, INT * IDX, DREAL * alphas, INT tree_num, INT upto_tree)
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;
	
	if (upto_tree<0)
		upto_tree=tree_num;

	if (max_mismatch!=0)
	{
		SG_ERROR( "CWeightedDegreePositionStringKernel optimization not implemented for mismatch!=0\n");
		return false ;
	}

	if (tree_num<0)
		SG_DEBUG( "deleting CWeightedDegreePositionStringKernel optimization\n");
	delete_optimization();

	if (tree_num<0)
		SG_DEBUG( "initializing CWeightedDegreePositionStringKernel optimization\n") ;

	int i=0;
	for (i=0; i<p_count; i++)
	{
		if (tree_num<0)
		{
			if ( (i % (p_count/10+1)) == 0)
				SG_PROGRESS(i,0,p_count);
			add_example_to_tree(IDX[i], alphas[i]);
		}
		else
		{
			for (INT t=tree_num; t<=upto_tree; t++)
				add_example_to_single_tree(IDX[i], alphas[i], t);
		}
	}

	if (tree_num<0)
		SG_DEBUG( "done.           \n");

	set_is_initialized(true) ;
	return true ;
}

bool CWeightedDegreePositionStringKernel::delete_optimization() 
{ 
	if ((opt_type==FASTBUTMEMHUNGRY) && (tries.get_use_compact_terminal_nodes())) 
	{
		tries.set_use_compact_terminal_nodes(false) ;
		SG_DEBUG( "disabling compact trie nodes with FASTBUTMEMHUNGRY\n") ;
	}

	if (get_is_initialized())
	{
		if (opt_type==SLOWBUTMEMEFFICIENT)
			tries.delete_trees(true); 
		else if (opt_type==FASTBUTMEMHUNGRY)
			tries.delete_trees(false);  // still buggy
		else {
			SG_ERROR( "unknown optimization type\n");
		}
		set_is_initialized(false);

		return true;
	}

	return false;
}

DREAL CWeightedDegreePositionStringKernel::compute_with_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
	DREAL max_shift_vec[max_shift];
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

DREAL CWeightedDegreePositionStringKernel::compute_without_mismatch(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
	DREAL max_shift_vec[max_shift];
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

DREAL CWeightedDegreePositionStringKernel::compute_without_mismatch_matrix(CHAR* avec, INT alen, CHAR* bvec, INT blen) 
{
	DREAL max_shift_vec[max_shift];
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
			sumi += weights[i*degree+j];
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

DREAL CWeightedDegreePositionStringKernel::compute_without_mismatch_position_weights(CHAR* avec, DREAL* pos_weights_lhs, INT alen, CHAR* bvec, DREAL* pos_weights_rhs, INT blen) 
{
	DREAL max_shift_vec[max_shift];
	DREAL sum0=0 ;
	for (INT i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;

	// no shift
	for (INT i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;

		DREAL sumi = 0.0 ;
	    DREAL posweight_lhs = 0.0 ;
	    DREAL posweight_rhs = 0.0 ;
		for (INT j=0; (j<degree) && (i+j<alen); j++)
		{
			posweight_lhs += pos_weights_lhs[i+j] ;
			posweight_rhs += pos_weights_rhs[i+j] ;
			
			if (avec[i+j]!=bvec[i+j])
				break ;
			sumi += weights[j]*(posweight_lhs/(j+1))*(posweight_rhs/(j+1)) ;
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

			// shift in sequence a	
			DREAL sumi1 = 0.0 ;
			DREAL posweight_lhs = 0.0 ;
			DREAL posweight_rhs = 0.0 ;
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				posweight_lhs += pos_weights_lhs[i+j+k] ;
				posweight_rhs += pos_weights_rhs[i+j] ;
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j]*(posweight_lhs/(j+1))*(posweight_rhs/(j+1)) ;
			}
			// shift in sequence b
			DREAL sumi2 = 0.0 ;
			posweight_lhs = 0.0 ;
			posweight_rhs = 0.0 ;
			for (INT j=0; (j<degree) && (i+j+k<alen); j++)
			{
				posweight_lhs += pos_weights_lhs[i+j] ;
				posweight_rhs += pos_weights_rhs[i+j+k] ;
				if (avec[i+j]!=bvec[i+j+k])
					break ;
				sumi2 += weights[j]*(posweight_lhs/(j+1))*(posweight_rhs/(j+1)) ;
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


DREAL CWeightedDegreePositionStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	CHAR* avec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR* bvec=((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	// can only deal with strings of same length
	ASSERT(alen == blen);
	ASSERT(shift_len == alen) ;

	DREAL result = 0 ;
	if (position_weights_lhs!=NULL || position_weights_rhs!=NULL)
	{
		ASSERT(max_mismatch==0) ;
		result = compute_without_mismatch_position_weights(avec, &position_weights_lhs[idx_a*alen], alen, bvec, &position_weights_rhs[idx_b*blen], blen) ;
	}
	else if (max_mismatch > 0)
		result = compute_with_mismatch(avec, alen, bvec, blen) ;
	else if (length==0)
		result = compute_without_mismatch(avec, alen, bvec, blen) ;
	else
		result = compute_without_mismatch_matrix(avec, alen, bvec, blen) ;

	result/=normalization_const;

	return result ;
}

void CWeightedDegreePositionStringKernel::add_example_to_tree(INT idx, DREAL alpha)
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;

	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, len);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	for (INT i=0; i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

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
			SG_ERROR( "unknown optimization type\n");
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

	delete[] vec ;
	tree_initialized=true ;
}

void CWeightedDegreePositionStringKernel::add_example_to_single_tree(INT idx, DREAL alpha, INT tree_num) 
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;

	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, len);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	INT max_s=-1;

	if (opt_type==SLOWBUTMEMEFFICIENT)
		max_s=0;
	else if (opt_type==FASTBUTMEMHUNGRY)
	{
		ASSERT(!tries.get_use_compact_terminal_nodes()) ;
		max_s=shift[tree_num];
	}
	else {
		SG_ERROR( "unknown optimization type\n");
	}
	for (INT i=CMath::max(0,tree_num-max_shift); i<CMath::min(len,tree_num+degree+max_shift); i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

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
	delete[] vec ;
	tree_initialized=true ;
}

DREAL CWeightedDegreePositionStringKernel::compute_by_tree(INT idx)
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;

	DREAL sum = 0 ;
	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx, len);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	for (INT i=0; i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

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

	delete[] vec ;

	return sum/normalization_const;
}

void CWeightedDegreePositionStringKernel::compute_by_tree(INT idx, DREAL* LevelContrib)
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;

	INT len ;
	CHAR* char_vec=((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx, len);
	ASSERT(max_mismatch==0) ;
	INT *vec = new INT[len] ;

	for (INT i=0; i<len; i++)
		vec[i]=((CStringFeatures<CHAR>*) lhs)->get_alphabet()->remap_to_bin(char_vec[i]);

	for (INT i=0; i<len; i++)
		tries.compute_by_tree_helper(vec, len, i, i, i, LevelContrib, 1.0/normalization_const, mkl_stepsize, weights, (length!=0)) ;

	if (opt_type==SLOWBUTMEMEFFICIENT)
	{
		for (INT i=0; i<len; i++)
			for (INT k=1; (k<=shift[i]) && (i+k<len); k++)
			{
				tries.compute_by_tree_helper(vec, len, i, i+k, i, LevelContrib, 1.0/(2*k*normalization_const), mkl_stepsize, weights, (length!=0)) ;
				tries.compute_by_tree_helper(vec, len, i+k, i, i+k, LevelContrib, 1.0/(2*k*normalization_const), mkl_stepsize, weights, (length!=0)) ;
			}
	}

	delete[] vec ;
}

DREAL *CWeightedDegreePositionStringKernel::compute_abs_weights(int &len) 
{
	return tries.compute_abs_weights(len) ;
}

bool CWeightedDegreePositionStringKernel::set_shifts(INT* shift_, INT shift_len_)
{
	delete[] shift;

	shift_len = shift_len_ ;
	shift = new INT[shift_len] ;

	if (shift)
	{
		max_shift = 0 ;

		for (INT i=0; i<shift_len; i++)
		{
			shift[i] = shift_[i] ;
			if (shift[i]>max_shift)
				max_shift = shift[i] ;
		}

		ASSERT(max_shift>=0 && max_shift<=shift_len) ;
	}
	
	return false;
}

bool CWeightedDegreePositionStringKernel::set_wd_weights()
{
	ASSERT(degree>0);

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

		return true;
	}
	else
		return false;
}

bool CWeightedDegreePositionStringKernel::set_weights(DREAL* ws, INT d, INT len)
{
	SG_DEBUG( "degree = %i  d=%i\n", degree, d) ;
	degree = d ;
	length=len;

	if (len <= 0)
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

bool CWeightedDegreePositionStringKernel::set_position_weights(DREAL* pws, INT len)
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

bool CWeightedDegreePositionStringKernel::set_position_weights_lhs(DREAL* pws, INT len, INT num)
{
	fprintf(stderr, "lhs %i %i %i\n", len, num, seq_length) ;
	
	if (position_weights_rhs==position_weights_lhs)
		position_weights_rhs=NULL ;
	else
		delete_position_weights_rhs() ;

	if (len==0)
	{
		return delete_position_weights_lhs() ;
	}
	
	if (seq_length!=len) 
	{
		SG_ERROR( "seq_length = %i, position_weights_length=%i\n", seq_length, len) ;
		return false ;
	}
	if (!get_lhs())
	{
		SG_ERROR("get_lhs()=NULL\n") ;
		return false ;
	}
	if (get_lhs()->get_num_vectors()!=num)
	{
		SG_ERROR("get_lhs()->get_num_vectors()=%i, num=%i\n", get_lhs()->get_num_vectors(), num) ;
		return false ;
	}
	
	delete[] position_weights_lhs;
	position_weights_lhs=new DREAL[len*num];
	if (position_weights_lhs)
	{
		for (int i=0; i<len*num; i++)
			position_weights_lhs[i]=pws[i];
		return true;
	}
	else
		return false;
}

bool CWeightedDegreePositionStringKernel::set_position_weights_rhs(DREAL* pws, INT len, INT num)
{
	fprintf(stderr, "rhs %i %i %i\n", len, num, seq_length) ;
	if (len==0)
	{
		if (position_weights_rhs==position_weights_lhs)
		{
			position_weights_rhs=NULL ;
			return true ;
		}
		return delete_position_weights_rhs() ;
	}

	if (seq_length!=len) 
	{
		SG_ERROR( "seq_length = %i, position_weights_length=%i\n", seq_length, len) ;
		return false ;
	}
	if (!get_rhs())
	{
		if (!get_lhs())
		{
			SG_ERROR("get_rhs()==0 and get_lhs()=NULL\n") ;
			return false ;
		}
		if (get_lhs()->get_num_vectors()!=num)
		{
			SG_ERROR("get_lhs()->get_num_vectors()=%i, num=%i\n", get_lhs()->get_num_vectors(), num) ;
			return false ;
		}
	} else
		if (get_rhs()->get_num_vectors()!=num)
		{
			SG_ERROR("get_rhs()->get_num_vectors()=%i, num=%i\n", get_rhs()->get_num_vectors(), num) ;
			return false ;
		}


	delete[] position_weights_rhs;
	position_weights_rhs=new DREAL[len*num];
	if (position_weights_rhs)
	{
		for (int i=0; i<len*num; i++)
			position_weights_rhs[i]=pws[i];
		return true;
	}
	else
		return false;
}

bool CWeightedDegreePositionStringKernel::init_block_weights_from_wd()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_from_wd_external()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_const()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_linear()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_sqpoly()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_cubicpoly()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_exp()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_log()
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

bool CWeightedDegreePositionStringKernel::init_block_weights_external()
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

bool CWeightedDegreePositionStringKernel::init_block_weights()
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



void* CWeightedDegreePositionStringKernel::compute_batch_helper(void* p)
{
	S_THREAD_PARAM* params = (S_THREAD_PARAM*) p;
	INT j=params->j;
	CWeightedDegreePositionStringKernel* wd=params->kernel;
	CTrie* tries=params->tries;
	DREAL* weights=params->weights;
	INT length=params->length;
	INT max_shift=params->max_shift;
	INT* vec=params->vec;
	DREAL* result=params->result;
	DREAL factor=params->factor;
	INT* shift=params->shift;
	INT* vec_idx=params->vec_idx;

	for (INT i=params->start; i<params->end; i++)
	{
		INT len=0;
		CHAR* char_vec=((CStringFeatures<CHAR>*) wd->get_rhs())->get_feature_vector(vec_idx[i], len);
		for (INT k=CMath::max(0,j-max_shift); k<CMath::min(len,j+wd->get_degree()+max_shift); k++)
			vec[k]=((CStringFeatures<CHAR>*) wd->get_lhs())->get_alphabet()->remap_to_bin(char_vec[k]);

		result[i] += factor*tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0))/wd->normalization_const ;

		if (wd->get_optimization_type()==SLOWBUTMEMEFFICIENT)
		{
			for (INT q=CMath::max(0,j-max_shift); q<CMath::min(len,j+max_shift+1); q++)
			{
				INT s=j-q ;
				if ((s>=1) && (s<=shift[q]) && (q+s<len))
					result[i] += tries->compute_by_tree_helper(vec, len, q, q+s, q, weights, (length!=0))/(2.0*s*wd->normalization_const) ;
			}
			for (INT s=1; (s<=shift[j]) && (j+s<len); s++)
				result[i] += tries->compute_by_tree_helper(vec, len, j+s, j, j+s, weights, (length!=0))/(2.0*s*wd->normalization_const) ;
		}
	}

	return NULL;
}

void CWeightedDegreePositionStringKernel::compute_batch(INT num_vec, INT* vec_idx, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas, DREAL factor)
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;

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
				params[t].max_shift=max_shift;
				params[t].shift=shift;
				params[t].vec_idx=vec_idx;
				pthread_create(&threads[t], NULL, CWeightedDegreePositionStringKernel::compute_batch_helper, (void*)&params[t]);
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
			compute_batch_helper((void*) &params[t]);

			for (t=0; t<num_threads-1; t++)
				pthread_join(threads[t], NULL);
			SG_PROGRESS(j,0,num_feat);
		}
	}
#endif

	delete[] vec;
}

DREAL* CWeightedDegreePositionStringKernel::compute_scoring(INT max_degree, INT& num_feat, INT& num_sym, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas)
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;

	num_feat=((CStringFeatures<CHAR>*) get_rhs())->get_max_vector_length();
	ASSERT(num_feat>0);
	ASSERT(((CStringFeatures<CHAR>*) get_rhs())->get_alphabet()->get_alphabet() == DNA);
	num_sym=4; //for now works only w/ DNA

	ASSERT(max_degree>0);

	// === variables
	INT* nofsKmers = new INT[ max_degree ];
	DREAL** C = new DREAL*[ max_degree ];
	DREAL** L = new DREAL*[ max_degree ];
	DREAL** R = new DREAL*[ max_degree ];

	ASSERT(nofsKmers);
	ASSERT(C);
	ASSERT(L);
	ASSERT(R);

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
	ASSERT(result);

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
	ASSERT(margFactors);

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
			SG_PROGRESS(i++,0,num_feat*max_degree);
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

CHAR* CWeightedDegreePositionStringKernel::compute_consensus(INT &num_feat, INT num_suppvec, INT* IDX, DREAL* alphas)
{
	ASSERT(position_weights_lhs==NULL) ;
	ASSERT(position_weights_rhs==NULL) ;

	//only works for order <= 32
	ASSERT(degree<=32);
	ASSERT(!tries.get_use_compact_terminal_nodes());
	num_feat=((CStringFeatures<CHAR>*) get_rhs())->get_max_vector_length();
	ASSERT(num_feat>0);
	ASSERT(((CStringFeatures<CHAR>*) get_rhs())->get_alphabet()->get_alphabet() == DNA);

	//consensus
	CHAR* result= new CHAR[num_feat];
	ASSERT(result);

	//backtracking and scoring table
	INT num_tables=CMath::max(1,num_feat-degree+1);
	CDynamicArray<ConsensusEntry>** table = new CDynamicArray<ConsensusEntry>*[num_tables];
	ASSERT(table);
	
	for (INT i=0; i<num_tables; i++)
	{
		table[i] = new CDynamicArray<ConsensusEntry>(num_suppvec/10);
		ASSERT(table[i]);
	}

	//compute consensus via dynamic programming
	for (INT i=0; i<num_tables; i++)
	{
		bool cumulative=false;

		if (i<num_tables-1)
			init_optimization(num_suppvec, IDX, alphas, i);
		else
		{
			init_optimization(num_suppvec, IDX, alphas, i, num_feat-1);
			cumulative=true;
		}

		if (i==0)
			tries.fill_backtracking_table(i, NULL, table[i], cumulative, weights);
		else
			tries.fill_backtracking_table(i, table[i-1], table[i], cumulative, weights);

		SG_PROGRESS(i,0,num_feat);
	}


	//INT n=table[0]->get_num_elements();

	//for (INT i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[0]->get_element(i);
	//	SG_PRINT("first: str:0%0llx sc:%f bt:%d\n",e.string,e.score,e.bt);
	//}

	//n=table[num_tables-1]->get_num_elements();
	//for (INT i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[num_tables-1]->get_element(i);
	//	SG_PRINT("last: str:0%0llx sc:%f bt:%d\n",e.string,e.score,e.bt);
	//}
	//n=table[num_tables-2]->get_num_elements();
	//for (INT i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[num_tables-2]->get_element(i);
	//	SG_PRINT("second last: str:0%0llx sc:%f bt:%d\n",e.string,e.score,e.bt);
	//}

	const CHAR* acgt="ACGT";

	//backtracking start
	INT max_idx=-1;
	SHORTREAL max_score=0;
	INT num_elements=table[num_tables-1]->get_num_elements();

	for (INT i=0; i<num_elements; i++)
	{
		DREAL sc=table[num_tables-1]->get_element(i).score;
		if (sc>max_score || max_idx==-1)
		{
			max_idx=i;
			max_score=sc;
		}
	}
	ULONG endstr=table[num_tables-1]->get_element(max_idx).string;

	SG_INFO("max_idx:%d num_el:%d num_feat:%d num_tables:%d max_score:%f\n", max_idx, num_elements, num_feat, num_tables, max_score);

	for (INT i=0; i<degree; i++)
		result[num_feat-1-i]=acgt[(endstr >> (2*i)) & 3];

	if (num_tables>1)
	{
		for (INT i=num_tables-1; i>=0; i--)
		{
			//SG_PRINT("max_idx: %d, i:%d\n", max_idx, i);
			result[i]=acgt[table[i]->get_element(max_idx).string >> (2*(degree-1)) & 3];
			max_idx=table[i]->get_element(max_idx).bt;
		}
	}

	//for (INT t=0; t<num_tables; t++)
	//{
	//	n=table[t]->get_num_elements();
	//	for (INT i=0; i<n; i++)
	//	{
	//		ConsensusEntry e= table[t]->get_element(i);
	//		SG_PRINT("table[%d,%d]: str:0%0llx sc:%+f bt:%d\n",t,i, e.string,e.score,e.bt);
	//	}
	//}

	for (INT i=0; i<num_tables; i++)
		delete table[i];

	delete[] table;
	return result;
}


#ifdef TRIE_FOR_POIMS

DREAL* CWeightedDegreePositionStringKernel::compute_POIM( INT max_degree, INT& num_feat, INT& num_sym, DREAL* result, INT num_suppvec, INT* IDX, DREAL* alphas )
{
  // === check
  ASSERT( position_weights_lhs == NULL );
  ASSERT( position_weights_rhs == NULL );
  num_feat=((CStringFeatures<CHAR>*) get_rhs())->get_max_vector_length();
  ASSERT( num_feat > 0 );
  ASSERT( ((CStringFeatures<CHAR>*) get_rhs())->get_alphabet()->get_alphabet() == DNA );
  ASSERT( max_degree > 0 );

  // === general variables
  static const INT NUM_SYM = Trie::NUM_SYMS;
  const INT seqLen = num_feat;
  DREAL** subs;
  INT i;
  INT k;
  //INT y;

  // === init tables "subs" for substring scores / POIMs
  // --- compute table sizes
  INT* offsets;
  INT offset;
  offsets = new INT[ max_degree ];
  offset = 0;
  for( k = 0; k < max_degree; ++k ) {
    offsets[k] = offset;
    const INT nofsKmers = (INT) pow( NUM_SYM, k+1 );
    const INT tabSize = nofsKmers * seqLen;
    offset += tabSize;
  }
  // --- allocate memory
  const INT bigTabSize = offset;
  result = new DREAL[ bigTabSize ];
  ASSERT( result != NULL );
  for( i = 0; i < bigTabSize; ++i ) {
    result[i] = 0;
  }
  // --- set pointers for tables
  subs = new DREAL*[ max_degree ];
  ASSERT( subs != NULL );
  for( k = 0; k < max_degree; ++k ) {
    subs[k] = &result[ offsets[k] ];
  }
  delete[] offsets;

  // === init trees; precalc S, L and R
  //for( i = 0; i < seqLen; ++i ) {
  //init_optimization( num_suppvec, IDX, alphas, i );
  //const INT treeIdx = i;
  //}
  init_optimization(num_suppvec, IDX, alphas, 0, seqLen-1);
  tries.POIMs_precalc_SLR();

  // === compute substring scores / POIMs
  tries.POIMs_extract_W( subs, max_degree );
  //for( k = 0; k < max_degree-1; ++k ) {
  //  const INT nofsKmers = (INT) pow( NUM_SYM, k+1 );
  //  DREAL* const subs_kn = ( k > 0 ) ? subs[ k-1 ] : NULL;
  //  DREAL* const subs_ko = subs[ k ];
  //  DREAL* const subs_kp = subs[ k+1 ];
  //  for( y = 0; y < nofsKmers; ++y ) {
  //    
  //  }
  //}

  // === clean; return "subs" as vector
  delete[] subs;
  num_feat = 1;
  num_sym = bigTabSize;
  return result;
}

#endif

