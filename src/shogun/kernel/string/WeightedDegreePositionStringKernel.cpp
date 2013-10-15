/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Trie.h>
#include <shogun/base/Parallel.h>

#include <shogun/kernel/string/WeightedDegreePositionStringKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>

#include <shogun/classifier/svm/SVM.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

#define TRIES(X) ((use_poim_tries) ? (poim_tries.X) : (tries.X))

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <class Trie> struct S_THREAD_PARAM_WDS
{
	int32_t* vec;
	float64_t* result;
	float64_t* weights;
	CWeightedDegreePositionStringKernel* kernel;
	CTrie<Trie>* tries;
	float64_t factor;
	int32_t j;
	int32_t start;
	int32_t end;
	int32_t length;
	int32_t max_shift;
	int32_t* shift;
	int32_t* vec_idx;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel(
	void)
: CStringKernel<char>()
{
	init();
}

CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel(
	int32_t size, int32_t d, int32_t mm, int32_t mkls)
: CStringKernel<char>(size)
{
	init();

	mkl_stepsize=mkls;
	degree=d;
	max_mismatch=mm;

	tries=CTrie<DNATrie>(d);
	poim_tries=CTrie<POIMTrie>(d);

	set_wd_weights();
	ASSERT(weights)
}

CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel(
	int32_t size, SGVector<float64_t> w, int32_t d, int32_t mm, SGVector<int32_t> s,
	int32_t mkls)
: CStringKernel<char>(size)
{
	init();

	mkl_stepsize=mkls;
	degree=d;
	max_mismatch=mm;

	tries=CTrie<DNATrie>(d);
	poim_tries=CTrie<POIMTrie>(d);

	weights=SG_MALLOC(float64_t, d*(1+max_mismatch));
	weights_degree=degree;
	weights_length=(1+max_mismatch);

	for (int32_t i=0; i<d*(1+max_mismatch); i++)
		weights[i]=w[i];

	set_shifts(s);
}

CWeightedDegreePositionStringKernel::CWeightedDegreePositionStringKernel(
	CStringFeatures<char>* l, CStringFeatures<char>* r, int32_t d)
: CStringKernel<char>()
{
	init();

	mkl_stepsize=1;
	degree=d;

	tries=CTrie<DNATrie>(d);
	poim_tries=CTrie<POIMTrie>(d);

	set_wd_weights();
	ASSERT(weights)

	init(l, r);
}


CWeightedDegreePositionStringKernel::~CWeightedDegreePositionStringKernel()
{
	cleanup();
	cleanup_POIM2();

	SG_FREE(shift);
	shift=NULL;

	SG_FREE(weights);
	weights=NULL;
	weights_degree=0;
	weights_length=0;

	SG_FREE(block_weights);
	block_weights=NULL;

	SG_FREE(position_weights);
	position_weights=NULL;

	SG_FREE(position_weights_lhs);
	position_weights_lhs=NULL;

	SG_FREE(position_weights_rhs);
	position_weights_rhs=NULL;

	SG_FREE(weights_buffer);
	weights_buffer=NULL;
}

void CWeightedDegreePositionStringKernel::remove_lhs()
{
	SG_DEBUG("deleting CWeightedDegreePositionStringKernel optimization\n")
	delete_optimization();

	tries.destroy();
	poim_tries.destroy();

	CKernel::remove_lhs();
}

void CWeightedDegreePositionStringKernel::create_empty_tries()
{
	ASSERT(lhs)
	seq_length = ((CStringFeatures<char>*) lhs)->get_max_vector_length();

	if (opt_type==SLOWBUTMEMEFFICIENT)
	{
		tries.create(seq_length, true);
		poim_tries.create(seq_length, true);
	}
	else if (opt_type==FASTBUTMEMHUNGRY)
	{
		tries.create(seq_length, false);  // still buggy
		poim_tries.create(seq_length, false);  // still buggy
	}
	else
		SG_ERROR("unknown optimization type\n")
}

bool CWeightedDegreePositionStringKernel::init(CFeatures* l, CFeatures* r)
{
	int32_t lhs_changed = (lhs!=l) ;
	int32_t rhs_changed = (rhs!=r) ;

	CStringKernel<char>::init(l,r);

	SG_DEBUG("lhs_changed: %i\n", lhs_changed)
	SG_DEBUG("rhs_changed: %i\n", rhs_changed)

	CStringFeatures<char>* sf_l=(CStringFeatures<char>*) l;
	CStringFeatures<char>* sf_r=(CStringFeatures<char>*) r;

	/* set shift */
	if (shift_len==0) {
		shift_len=sf_l->get_vector_length(0);
		int32_t *shifts=SG_MALLOC(int32_t, shift_len);
		for (int32_t i=0; i<shift_len; i++) {
			shifts[i]=1;
		}
		set_shifts(SGVector<int32_t>(shifts, shift_len, false));
		SG_FREE(shifts);
	}


	int32_t len=sf_l->get_max_vector_length();
	if (lhs_changed && !sf_l->have_same_length(len))
		SG_ERROR("All strings in WD kernel must have same length (lhs wrong)!\n")

	if (rhs_changed && !sf_r->have_same_length(len))
		SG_ERROR("All strings in WD kernel must have same length (rhs wrong)!\n")

	SG_UNREF(alphabet);
	alphabet= sf_l->get_alphabet();
	CAlphabet* ralphabet=sf_r->get_alphabet();

	if (!((alphabet->get_alphabet()==DNA) || (alphabet->get_alphabet()==RNA)))
		properties &= ((uint64_t) (-1)) ^ (KP_LINADD | KP_BATCHEVALUATION);

	ASSERT(ralphabet->get_alphabet()==alphabet->get_alphabet())
	SG_UNREF(ralphabet);

	//whenever init is called also init tries and block weights
	create_empty_tries();
	init_block_weights();

	return init_normalizer();
}

void CWeightedDegreePositionStringKernel::cleanup()
{
	SG_DEBUG("deleting CWeightedDegreePositionStringKernel optimization\n")
	delete_optimization();

	SG_FREE(block_weights);
	block_weights=NULL;

	tries.destroy();
	poim_tries.destroy();

	seq_length = 0;
	tree_initialized = false;

	SG_UNREF(alphabet);
	alphabet=NULL;

	CKernel::cleanup();
}

bool CWeightedDegreePositionStringKernel::init_optimization(
	int32_t p_count, int32_t * IDX, float64_t * alphas, int32_t tree_num,
	int32_t upto_tree)
{
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)

	if (upto_tree<0)
		upto_tree=tree_num;

	if (max_mismatch!=0)
	{
		SG_ERROR("CWeightedDegreePositionStringKernel optimization not implemented for mismatch!=0\n")
		return false ;
	}

	if (tree_num<0)
		SG_DEBUG("deleting CWeightedDegreePositionStringKernel optimization\n")

	delete_optimization();

	if (tree_num<0)
		SG_DEBUG("initializing CWeightedDegreePositionStringKernel optimization\n")

	for (int32_t i=0; i<p_count; i++)
	{
		if (tree_num<0)
		{
			if ( (i % (p_count/10+1)) == 0)
				SG_PROGRESS(i,0,p_count)
			add_example_to_tree(IDX[i], alphas[i]);
		}
		else
		{
			for (int32_t t=tree_num; t<=upto_tree; t++)
				add_example_to_single_tree(IDX[i], alphas[i], t);
		}
	}

	if (tree_num<0)
		SG_DONE()

	set_is_initialized(true) ;
	return true ;
}

bool CWeightedDegreePositionStringKernel::delete_optimization()
{
	if ((opt_type==FASTBUTMEMHUNGRY) && (tries.get_use_compact_terminal_nodes()))
	{
		tries.set_use_compact_terminal_nodes(false) ;
		SG_DEBUG("disabling compact trie nodes with FASTBUTMEMHUNGRY\n")
	}

	if (get_is_initialized())
	{
		if (opt_type==SLOWBUTMEMEFFICIENT)
			tries.delete_trees(true);
		else if (opt_type==FASTBUTMEMHUNGRY)
			tries.delete_trees(false);  // still buggy
		else {
			SG_ERROR("unknown optimization type\n")
		}
		set_is_initialized(false);

		return true;
	}

	return false;
}

float64_t CWeightedDegreePositionStringKernel::compute_with_mismatch(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	float64_t* max_shift_vec= SG_MALLOC(float64_t, max_shift);
    float64_t sum0=0 ;
    for (int32_t i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;

    // no shift
    for (int32_t i=0; i<alen; i++)
    {
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;

		int32_t mismatches=0;
		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
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

    for (int32_t i=0; i<alen; i++)
    {
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			int32_t mismatches=0;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
				{
					mismatches++ ;
					if (mismatches>max_mismatch)
						break ;
				} ;
				sumi1 += weights[j+degree*mismatches];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			mismatches=0;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
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

    float64_t result = sum0 ;
    for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	SG_FREE(max_shift_vec);
    return result ;
}

float64_t CWeightedDegreePositionStringKernel::compute_without_mismatch(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	float64_t* max_shift_vec = SG_MALLOC(float64_t, max_shift);
	float64_t sum0=0 ;
	for (int32_t i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;

	// no shift
	for (int32_t i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;

		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
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

	for (int32_t i=0; i<alen; i++)
	{
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
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

	float64_t result = sum0 ;
	for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	SG_FREE(max_shift_vec);

	return result ;
}

float64_t CWeightedDegreePositionStringKernel::compute_without_mismatch_matrix(
	char* avec, int32_t alen, char* bvec, int32_t blen)
{
	float64_t* max_shift_vec = SG_MALLOC(float64_t, max_shift);
	float64_t sum0=0 ;
	for (int32_t i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;

	// no shift
	for (int32_t i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;
		float64_t sumi = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
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

	for (int32_t i=0; i<alen; i++)
	{
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			float64_t sumi1 = 0.0 ;
			// shift in sequence a
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[i*degree+j];
			}
			float64_t sumi2 = 0.0 ;
			// shift in sequence b
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
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

	float64_t result = sum0 ;
	for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	SG_FREE(max_shift_vec);
	return result ;
}

float64_t CWeightedDegreePositionStringKernel::compute_without_mismatch_position_weights(
	char* avec, float64_t* pos_weights_lhs, int32_t alen, char* bvec,
	float64_t* pos_weights_rhs, int32_t blen)
{
	float64_t* max_shift_vec = SG_MALLOC(float64_t, max_shift);
	float64_t sum0=0 ;
	for (int32_t i=0; i<max_shift; i++)
		max_shift_vec[i]=0 ;

	// no shift
	for (int32_t i=0; i<alen; i++)
	{
		if ((position_weights!=NULL) && (position_weights[i]==0.0))
			continue ;

		float64_t sumi = 0.0 ;
	    float64_t posweight_lhs = 0.0 ;
	    float64_t posweight_rhs = 0.0 ;
		for (int32_t j=0; (j<degree) && (i+j<alen); j++)
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

	for (int32_t i=0; i<alen; i++)
	{
		for (int32_t k=1; (k<=shift[i]) && (i+k<alen); k++)
		{
			if ((position_weights!=NULL) && (position_weights[i]==0.0) && (position_weights[i+k]==0.0))
				continue ;

			// shift in sequence a
			float64_t sumi1 = 0.0 ;
			float64_t posweight_lhs = 0.0 ;
			float64_t posweight_rhs = 0.0 ;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
			{
				posweight_lhs += pos_weights_lhs[i+j+k] ;
				posweight_rhs += pos_weights_rhs[i+j] ;
				if (avec[i+j+k]!=bvec[i+j])
					break ;
				sumi1 += weights[j]*(posweight_lhs/(j+1))*(posweight_rhs/(j+1)) ;
			}
			// shift in sequence b
			float64_t sumi2 = 0.0 ;
			posweight_lhs = 0.0 ;
			posweight_rhs = 0.0 ;
			for (int32_t j=0; (j<degree) && (i+j+k<alen); j++)
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

	float64_t result = sum0 ;
	for (int32_t i=0; i<max_shift; i++)
		result += max_shift_vec[i]/(2*(i+1)) ;

	SG_FREE(max_shift_vec);
	return result ;
}


float64_t CWeightedDegreePositionStringKernel::compute(
	int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool free_avec, free_bvec;

	char* avec=((CStringFeatures<char>*) lhs)->get_feature_vector(idx_a, alen, free_avec);
	char* bvec=((CStringFeatures<char>*) rhs)->get_feature_vector(idx_b, blen, free_bvec);
	// can only deal with strings of same length
	ASSERT(alen==blen)
	ASSERT(shift_len==alen)

	float64_t result = 0 ;
	if (position_weights_lhs!=NULL || position_weights_rhs!=NULL)
	{
		ASSERT(max_mismatch==0)
		float64_t* position_weights_rhs_ = position_weights_rhs ;
		if (lhs==rhs)
			position_weights_rhs_ = position_weights_lhs ;

		result = compute_without_mismatch_position_weights(avec, &position_weights_lhs[idx_a*alen], alen, bvec, &position_weights_rhs_[idx_b*blen], blen) ;
	}
	else if (max_mismatch > 0)
		result = compute_with_mismatch(avec, alen, bvec, blen) ;
	else if (length==0)
		result = compute_without_mismatch(avec, alen, bvec, blen) ;
	else
		result = compute_without_mismatch_matrix(avec, alen, bvec, blen) ;

	((CStringFeatures<char>*) lhs)->free_feature_vector(avec, idx_a, free_avec);
	((CStringFeatures<char>*) rhs)->free_feature_vector(bvec, idx_b, free_bvec);

	return result ;
}


void CWeightedDegreePositionStringKernel::add_example_to_tree(
	int32_t idx, float64_t alpha)
{
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=((CStringFeatures<char>*) lhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec = SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	((CStringFeatures<char>*) lhs)->free_feature_vector(char_vec, idx, free_vec);

	if (opt_type==FASTBUTMEMHUNGRY)
	{
		//TRIES(set_use_compact_terminal_nodes(false)) ;
		ASSERT(!TRIES(get_use_compact_terminal_nodes()))
	}

	for (int32_t i=0; i<len; i++)
	{
		int32_t max_s=-1;

		if (opt_type==SLOWBUTMEMEFFICIENT)
			max_s=0;
		else if (opt_type==FASTBUTMEMHUNGRY)
			max_s=shift[i];
		else {
			SG_ERROR("unknown optimization type\n")
		}

		for (int32_t s=max_s; s>=0; s--)
		{
			float64_t alpha_pw = normalizer->normalize_lhs((s==0) ? (alpha) : (alpha/(2.0*s)), idx);
			TRIES(add_to_trie(i, s, vec, alpha_pw, weights, (length!=0))) ;
			if ((s==0) || (i+s>=len))
				continue;

			TRIES(add_to_trie(i+s, -s, vec, alpha_pw, weights, (length!=0))) ;
		}
	}

	SG_FREE(vec);
	tree_initialized=true ;
}

void CWeightedDegreePositionStringKernel::add_example_to_single_tree(
	int32_t idx, float64_t alpha, int32_t tree_num)
{
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=((CStringFeatures<char>*) lhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec=SG_MALLOC(int32_t, len);
	int32_t max_s=-1;

	if (opt_type==SLOWBUTMEMEFFICIENT)
		max_s=0;
	else if (opt_type==FASTBUTMEMHUNGRY)
	{
		ASSERT(!tries.get_use_compact_terminal_nodes())
		max_s=shift[tree_num];
	}
	else {
		SG_ERROR("unknown optimization type\n")
	}
	for (int32_t i=CMath::max(0,tree_num-max_shift);
			i<CMath::min(len,tree_num+degree+max_shift); i++)
	{
		vec[i]=alphabet->remap_to_bin(char_vec[i]);
	}
	((CStringFeatures<char>*) lhs)->free_feature_vector(char_vec, idx, free_vec);

	for (int32_t s=max_s; s>=0; s--)
	{
		float64_t alpha_pw = normalizer->normalize_lhs((s==0) ? (alpha) : (alpha/(2.0*s)), idx);
		tries.add_to_trie(tree_num, s, vec, alpha_pw, weights, (length!=0)) ;
	}

	if (opt_type==FASTBUTMEMHUNGRY)
	{
		for (int32_t i=CMath::max(0,tree_num-max_shift); i<CMath::min(len,tree_num+max_shift+1); i++)
		{
			int32_t s=tree_num-i;
			if ((i+s<len) && (s>=1) && (s<=shift[i]))
			{
				float64_t alpha_pw = normalizer->normalize_lhs((s==0) ? (alpha) : (alpha/(2.0*s)), idx);
				tries.add_to_trie(tree_num, -s, vec, alpha_pw, weights, (length!=0)) ;
			}
		}
	}
	SG_FREE(vec);
	tree_initialized=true ;
}

float64_t CWeightedDegreePositionStringKernel::compute_by_tree(int32_t idx)
{
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	float64_t sum=0;
	int32_t len=0;
	bool free_vec;
	char* char_vec=((CStringFeatures<char>*) rhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec=SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);

	((CStringFeatures<char>*) rhs)->free_feature_vector(char_vec, idx, free_vec);

	for (int32_t i=0; i<len; i++)
		sum += tries.compute_by_tree_helper(vec, len, i, i, i, weights, (length!=0)) ;

	if (opt_type==SLOWBUTMEMEFFICIENT)
	{
		for (int32_t i=0; i<len; i++)
		{
			for (int32_t s=1; (s<=shift[i]) && (i+s<len); s++)
			{
				sum+=tries.compute_by_tree_helper(vec, len, i, i+s, i, weights, (length!=0))/(2*s) ;
				sum+=tries.compute_by_tree_helper(vec, len, i+s, i, i+s, weights, (length!=0))/(2*s) ;
			}
		}
	}

	SG_FREE(vec);

	return normalizer->normalize_rhs(sum, idx);
}

void CWeightedDegreePositionStringKernel::compute_by_tree(
	int32_t idx, float64_t* LevelContrib)
{
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	int32_t len=0;
	bool free_vec;
	char* char_vec=((CStringFeatures<char>*) rhs)->get_feature_vector(idx, len, free_vec);
	ASSERT(max_mismatch==0)
	int32_t *vec=SG_MALLOC(int32_t, len);

	for (int32_t i=0; i<len; i++)
		vec[i]=alphabet->remap_to_bin(char_vec[i]);

	((CStringFeatures<char>*) rhs)->free_feature_vector(char_vec, idx, free_vec);

	for (int32_t i=0; i<len; i++)
	{
		tries.compute_by_tree_helper(vec, len, i, i, i, LevelContrib,
				normalizer->normalize_rhs(1.0, idx), mkl_stepsize, weights,
				(length!=0));
	}

	if (opt_type==SLOWBUTMEMEFFICIENT)
	{
		for (int32_t i=0; i<len; i++)
			for (int32_t k=1; (k<=shift[i]) && (i+k<len); k++)
			{
				tries.compute_by_tree_helper(vec, len, i, i+k, i, LevelContrib,
						normalizer->normalize_rhs(1.0/(2*k), idx), mkl_stepsize,
						weights, (length!=0)) ;
				tries.compute_by_tree_helper(vec, len, i+k, i, i+k,
						LevelContrib, normalizer->normalize_rhs(1.0/(2*k), idx),
						mkl_stepsize, weights, (length!=0)) ;
			}
	}

	SG_FREE(vec);
}

float64_t* CWeightedDegreePositionStringKernel::compute_abs_weights(
	int32_t &len)
{
	return tries.compute_abs_weights(len);
}

void CWeightedDegreePositionStringKernel::set_shifts(SGVector<int32_t> shifts)
{
	SG_FREE(shift);

	shift_len = shifts.vlen;
	shift = SG_MALLOC(int32_t, shift_len);

	if (shift)
	{
		max_shift = 0 ;

		for (int32_t i=0; i<shift_len; i++)
		{
			shift[i] = shifts.vector[i] ;
			max_shift = CMath::max(shift[i], max_shift);
		}

		ASSERT(max_shift>=0 && max_shift<=shift_len)
	}
}

bool CWeightedDegreePositionStringKernel::set_wd_weights()
{
	ASSERT(degree>0)

	SG_FREE(weights);
	weights=SG_MALLOC(float64_t, degree);
	weights_degree=degree;
	weights_length=1;

	if (weights)
	{
		int32_t i;
		float64_t sum=0;
		for (i=0; i<degree; i++)
		{
			weights[i]=degree-i;
			sum+=weights[i];
		}
		for (i=0; i<degree; i++)
			weights[i]/=sum;

		for (i=0; i<degree; i++)
		{
			for (int32_t j=1; j<=max_mismatch; j++)
			{
				if (j<i+1)
				{
					int32_t nk=CMath::nchoosek(i+1, j);
					weights[i+j*degree]=weights[i]/(nk*CMath::pow(3.0,j));
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

bool CWeightedDegreePositionStringKernel::set_weights(SGMatrix<float64_t> new_weights)
{
	float64_t* ws=new_weights.matrix;
	int32_t d=new_weights.num_rows;
	int32_t len=new_weights.num_cols;

	if (d!=degree || len<0)
		SG_ERROR("WD: Dimension mismatch (should be (seq_length | 1) x degree) got (%d x %d)\n", len, degree)

	degree=d;
	length=len;

	if (len <= 0)
		len=1;

	weights_degree=degree;
	weights_length=len+max_mismatch;

	SG_DEBUG("Creating weights of size %dx%d\n", weights_degree, weights_length)
	int32_t num_weights=weights_degree*weights_length;
	SG_FREE(weights);
	weights=SG_MALLOC(float64_t, num_weights);

	for (int32_t i=0; i<degree*len; i++)
		weights[i]=ws[i];

	return true;
}

void CWeightedDegreePositionStringKernel::set_position_weights(SGVector<float64_t> pws)
{
	if (seq_length==0)
		seq_length=pws.vlen;

	if (seq_length!=pws.vlen)
		SG_ERROR("seq_length = %i, position_weights_length=%i\n", seq_length, pws.vlen)

	SG_FREE(position_weights);
	position_weights=SG_MALLOC(float64_t, pws.vlen);
	position_weights_len=pws.vlen;
	tries.set_position_weights(position_weights);

	for (int32_t i=0; i<pws.vlen; i++)
		position_weights[i]=pws.vector[i];
}

bool CWeightedDegreePositionStringKernel::set_position_weights_lhs(float64_t* pws, int32_t len, int32_t num)
{
	if (position_weights_rhs==position_weights_lhs)
		position_weights_rhs=NULL;
	else
		delete_position_weights_rhs();

	if (len==0)
	{
		return delete_position_weights_lhs();
	}

	if (seq_length!=len)
	{
		SG_ERROR("seq_length = %i, position_weights_length=%i\n", seq_length, len)
		return false;
	}

	SG_FREE(position_weights_lhs);
	position_weights_lhs=SG_MALLOC(float64_t, len*num);
	position_weights_lhs_len=len*num;

	for (int32_t i=0; i<len*num; i++)
		position_weights_lhs[i]=pws[i];

	return true;
}

bool CWeightedDegreePositionStringKernel::set_position_weights_rhs(
	float64_t* pws, int32_t len, int32_t num)
{
	if (len==0)
	{
		if (position_weights_rhs==position_weights_lhs)
		{
			position_weights_rhs=NULL;
			return true;
		}
		return delete_position_weights_rhs();
	}

	if (seq_length!=len)
	{
		SG_ERROR("seq_length = %i, position_weights_length=%i\n", seq_length, len)
		return false;
	}

	SG_FREE(position_weights_rhs);
	position_weights_rhs=SG_MALLOC(float64_t, len*num);
	position_weights_rhs_len=len*num;

	for (int32_t i=0; i<len*num; i++)
		position_weights_rhs[i]=pws[i];

	return true;
}

bool CWeightedDegreePositionStringKernel::init_block_weights_from_wd()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, CMath::max(seq_length,degree));

	if (block_weights)
	{
		int32_t k;
		float64_t d=degree; // use float to evade rounding errors below

		for (k=0; k<degree; k++)
			block_weights[k]=
				(-CMath::pow(k, 3)+(3*d-3)*CMath::pow(k, 2)+(9*d-2)*k+6*d)/(3*d*(d+1));
		for (k=degree; k<seq_length; k++)
			block_weights[k]=(-d+3*k+4)/3;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreePositionStringKernel::init_block_weights_from_wd_external()
{
	ASSERT(weights)
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, CMath::max(seq_length,degree));

	if (block_weights)
	{
		int32_t i=0;
		block_weights[0]=weights[0];
		for (i=1; i<CMath::max(seq_length,degree); i++)
			block_weights[i]=0;

		for (i=1; i<CMath::max(seq_length,degree); i++)
		{
			block_weights[i]=block_weights[i-1];

			float64_t contrib=0;
			for (int32_t j=0; j<CMath::min(degree,i+1); j++)
				contrib+=weights[j];

			block_weights[i]+=contrib;
		}
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreePositionStringKernel::init_block_weights_const()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	if (block_weights)
	{
		for (int32_t i=1; i<seq_length+1 ; i++)
			block_weights[i-1]=1.0/seq_length;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreePositionStringKernel::init_block_weights_linear()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	if (block_weights)
	{
		for (int32_t i=1; i<seq_length+1 ; i++)
			block_weights[i-1]=degree*i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreePositionStringKernel::init_block_weights_sqpoly()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	if (block_weights)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i-1]=((float64_t) i)*i;

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreePositionStringKernel::init_block_weights_cubicpoly()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	if (block_weights)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i-1]=((float64_t) i)*i*i;

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreePositionStringKernel::init_block_weights_exp()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	if (block_weights)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i-1]=exp(((float64_t) i/10.0));

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i;
	}

	return (block_weights!=NULL);
}

bool CWeightedDegreePositionStringKernel::init_block_weights_log()
{
	SG_FREE(block_weights);
	block_weights=SG_MALLOC(float64_t, seq_length);

	if (block_weights)
	{
		for (int32_t i=1; i<degree+1 ; i++)
			block_weights[i-1]=CMath::pow(CMath::log((float64_t) i),2);

		for (int32_t i=degree+1; i<seq_length+1 ; i++)
			block_weights[i-1]=i-degree+1+CMath::pow(CMath::log(degree+1.0),2);
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
	};
	return false;
}



void* CWeightedDegreePositionStringKernel::compute_batch_helper(void* p)
{
	S_THREAD_PARAM_WDS<DNATrie>* params = (S_THREAD_PARAM_WDS<DNATrie>*) p;
	int32_t j=params->j;
	CWeightedDegreePositionStringKernel* wd=params->kernel;
	CTrie<DNATrie>* tries=params->tries;
	float64_t* weights=params->weights;
	int32_t length=params->length;
	int32_t max_shift=params->max_shift;
	int32_t* vec=params->vec;
	float64_t* result=params->result;
	float64_t factor=params->factor;
	int32_t* shift=params->shift;
	int32_t* vec_idx=params->vec_idx;

	for (int32_t i=params->start; i<params->end; i++)
	{
		int32_t len=0;
		CStringFeatures<char>* rhs_feat=((CStringFeatures<char>*) wd->get_rhs());
		CAlphabet* alpha=wd->alphabet;

		bool free_vec;
		char* char_vec=rhs_feat->get_feature_vector(vec_idx[i], len, free_vec);
		for (int32_t k=CMath::max(0,j-max_shift); k<CMath::min(len,j+wd->get_degree()+max_shift); k++)
			vec[k]=alpha->remap_to_bin(char_vec[k]);
		rhs_feat->free_feature_vector(char_vec, vec_idx[i], free_vec);

		SG_UNREF(rhs_feat);

		result[i] += factor*wd->normalizer->normalize_rhs(tries->compute_by_tree_helper(vec, len, j, j, j, weights, (length!=0)), vec_idx[i]);

		if (wd->get_optimization_type()==SLOWBUTMEMEFFICIENT)
		{
			for (int32_t q=CMath::max(0,j-max_shift); q<CMath::min(len,j+max_shift+1); q++)
			{
				int32_t s=j-q ;
				if ((s>=1) && (s<=shift[q]) && (q+s<len))
				{
					result[i] +=
						wd->normalizer->normalize_rhs(tries->compute_by_tree_helper(vec,
								len, q, q+s, q, weights, (length!=0)),
								vec_idx[i])/(2.0*s);
				}
			}

			for (int32_t s=1; (s<=shift[j]) && (j+s<len); s++)
			{
				result[i] +=
					wd->normalizer->normalize_rhs(tries->compute_by_tree_helper(vec,
								len, j+s, j, j+s, weights, (length!=0)),
								vec_idx[i])/(2.0*s);
			}
		}
	}

	return NULL;
}

void CWeightedDegreePositionStringKernel::compute_batch(
	int32_t num_vec, int32_t* vec_idx, float64_t* result, int32_t num_suppvec,
	int32_t* IDX, float64_t* alphas, float64_t factor)
{
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)
	ASSERT(rhs)
	ASSERT(num_vec<=rhs->get_num_vectors())
	ASSERT(num_vec>0)
	ASSERT(vec_idx)
	ASSERT(result)
	create_empty_tries();

	int32_t num_feat=((CStringFeatures<char>*) rhs)->get_max_vector_length();
	ASSERT(num_feat>0)
	int32_t num_threads=parallel->get_num_threads();
	ASSERT(num_threads>0)
	int32_t* vec=SG_MALLOC(int32_t, num_threads*num_feat);

	if (num_threads < 2)
	{
#ifdef WIN32
	   for (int32_t j=0; j<num_feat; j++)
#else
       CSignal::clear_cancel();
	   for (int32_t j=0; j<num_feat && !CSignal::cancel_computations(); j++)
#endif
			{
				init_optimization(num_suppvec, IDX, alphas, j);
				S_THREAD_PARAM_WDS<DNATrie> params;
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

				SG_PROGRESS(j,0,num_feat)
			}
	}
#ifdef HAVE_PTHREAD
	else
	{

		CSignal::clear_cancel();
		for (int32_t j=0; j<num_feat && !CSignal::cancel_computations(); j++)
		{
			init_optimization(num_suppvec, IDX, alphas, j);
			pthread_t* threads = SG_MALLOC(pthread_t, num_threads-1);
			S_THREAD_PARAM_WDS<DNATrie>* params = SG_MALLOC(S_THREAD_PARAM_WDS<DNATrie>, num_threads);
			int32_t step= num_vec/num_threads;
			int32_t t;

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
			SG_PROGRESS(j,0,num_feat)

			SG_FREE(params);
			SG_FREE(threads);
		}
	}
#endif

	SG_FREE(vec);

	//really also free memory as this can be huge on testing especially when
	//using the combined kernel
	create_empty_tries();
}

float64_t* CWeightedDegreePositionStringKernel::compute_scoring(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym, float64_t* result,
	int32_t num_suppvec, int32_t* IDX, float64_t* alphas)
{
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)

	num_feat=((CStringFeatures<char>*) rhs)->get_max_vector_length();
	ASSERT(num_feat>0)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	num_sym=4; //for now works only w/ DNA

	ASSERT(max_degree>0)

	// === variables
	int32_t* nofsKmers=SG_MALLOC(int32_t, max_degree);
	float64_t** C=SG_MALLOC(float64_t*, max_degree);
	float64_t** L=SG_MALLOC(float64_t*, max_degree);
	float64_t** R=SG_MALLOC(float64_t*, max_degree);

	int32_t i;
	int32_t k;

	// --- return table
	int32_t bigtabSize=0;
	for (k=0; k<max_degree; ++k )
	{
		nofsKmers[k]=(int32_t) CMath::pow(num_sym, k+1);
		const int32_t tabSize=nofsKmers[k]*num_feat;
		bigtabSize+=tabSize;
	}
	result=SG_MALLOC(float64_t, bigtabSize);

	// --- auxilliary tables
	int32_t tabOffs=0;
	for( k = 0; k < max_degree; ++k )
	{
		const int32_t tabSize = nofsKmers[k] * num_feat;
		C[k] = &result[tabOffs];
		L[k] = SG_MALLOC(float64_t,  tabSize );
		R[k] = SG_MALLOC(float64_t,  tabSize );
		tabOffs+=tabSize;
		for(i = 0; i < tabSize; i++ )
		{
			C[k][i] = 0.0;
			L[k][i] = 0.0;
			R[k][i] = 0.0;
		}
	}

	// --- tree parsing info
	float64_t* margFactors=SG_MALLOC(float64_t, degree);

	int32_t* x = SG_MALLOC(int32_t,  degree+1 );
	int32_t* substrs = SG_MALLOC(int32_t,  degree+1 );
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
		const int32_t nofKmers = nofsKmers[ k ];
		info.C_k = C[k];
		info.L_k = L[k];
		info.R_k = R[k];

		// --- run over all trees
		for(int32_t p = 0; p < num_feat; ++p )
		{
			init_optimization( num_suppvec, IDX, alphas, p );
			int32_t tree = p ;
			for(int32_t j = 0; j < degree+1; j++ ) {
				x[j] = -1;
			}
			tries.traverse( tree, p, info, 0, x, k );
			SG_PROGRESS(i++,0,num_feat*max_degree)
		}

		// --- add partial overlap scores
		if( k > 0 ) {
			const int32_t j = k - 1;
			const int32_t nofJmers = (int32_t) CMath::pow( num_sym, j+1 );
			for(int32_t p = 0; p < num_feat; ++p ) {
				const int32_t offsetJ = nofJmers * p;
				const int32_t offsetJ1 = nofJmers * (p+1);
				const int32_t offsetK = nofKmers * p;
				int32_t y;
				int32_t sym;
				for( y = 0; y < nofJmers; ++y ) {
					for( sym = 0; sym < num_sym; ++sym ) {
						const int32_t y_sym = num_sym*y + sym;
						const int32_t sym_y = nofJmers*sym + y;
						ASSERT(0<=y_sym && y_sym<nofKmers)
						ASSERT(0<=sym_y && sym_y<nofKmers)
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
	SG_FREE(nofsKmers);
	SG_FREE(margFactors);
	SG_FREE(substrs);
	SG_FREE(x);
	SG_FREE(C);
	for( k = 0; k < max_degree; ++k ) {
		SG_FREE(L[k]);
		SG_FREE(R[k]);
	}
	SG_FREE(L);
	SG_FREE(R);
	return result;
}

char* CWeightedDegreePositionStringKernel::compute_consensus(
	int32_t &num_feat, int32_t num_suppvec, int32_t* IDX, float64_t* alphas)
{
	ASSERT(position_weights_lhs==NULL)
	ASSERT(position_weights_rhs==NULL)
	//only works for order <= 32
	ASSERT(degree<=32)
	ASSERT(!tries.get_use_compact_terminal_nodes())
	num_feat=((CStringFeatures<char>*) rhs)->get_max_vector_length();
	ASSERT(num_feat>0)
	ASSERT(alphabet)
	ASSERT(alphabet->get_alphabet()==DNA || alphabet->get_alphabet()==RNA)

	//consensus
	char* result=SG_MALLOC(char, num_feat);

	//backtracking and scoring table
	int32_t num_tables=CMath::max(1,num_feat-degree+1);
	DynArray<ConsensusEntry>** table=SG_MALLOC(DynArray<ConsensusEntry>*, num_tables);

	for (int32_t i=0; i<num_tables; i++)
		table[i]=new DynArray<ConsensusEntry>(num_suppvec/10);

	//compute consensus via dynamic programming
	for (int32_t i=0; i<num_tables; i++)
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

		SG_PROGRESS(i,0,num_feat)
	}


	//int32_t n=table[0]->get_num_elements();

	//for (int32_t i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[0]->get_element(i);
	//	SG_PRint32_t("first: str:0%0llx sc:%f bt:%d\n",e.string,e.score,e.bt);
	//}

	//n=table[num_tables-1]->get_num_elements();
	//for (int32_t i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[num_tables-1]->get_element(i);
	//	SG_PRint32_t("last: str:0%0llx sc:%f bt:%d\n",e.string,e.score,e.bt);
	//}
	//n=table[num_tables-2]->get_num_elements();
	//for (int32_t i=0; i<n; i++)
	//{
	//	ConsensusEntry e= table[num_tables-2]->get_element(i);
	//	SG_PRINT("second last: str:0%0llx sc:%f bt:%d\n",e.string,e.score,e.bt)
	//}

	const char* acgt="ACGT";

	//backtracking start
	int32_t max_idx=-1;
	float32_t max_score=0;
	int32_t num_elements=table[num_tables-1]->get_num_elements();

	for (int32_t i=0; i<num_elements; i++)
	{
		float64_t sc=table[num_tables-1]->get_element(i).score;
		if (sc>max_score || max_idx==-1)
		{
			max_idx=i;
			max_score=sc;
		}
	}
	uint64_t endstr=table[num_tables-1]->get_element(max_idx).string;

	SG_INFO("max_idx:%d num_el:%d num_feat:%d num_tables:%d max_score:%f\n", max_idx, num_elements, num_feat, num_tables, max_score)

	for (int32_t i=0; i<degree; i++)
		result[num_feat-1-i]=acgt[(endstr >> (2*i)) & 3];

	if (num_tables>1)
	{
		for (int32_t i=num_tables-1; i>=0; i--)
		{
			//SG_PRINT("max_idx: %d, i:%d\n", max_idx, i)
			result[i]=acgt[table[i]->get_element(max_idx).string >> (2*(degree-1)) & 3];
			max_idx=table[i]->get_element(max_idx).bt;
		}
	}

	//for (int32_t t=0; t<num_tables; t++)
	//{
	//	n=table[t]->get_num_elements();
	//	for (int32_t i=0; i<n; i++)
	//	{
	//		ConsensusEntry e= table[t]->get_element(i);
	//		SG_PRINT("table[%d,%d]: str:0%0llx sc:%+f bt:%d\n",t,i, e.string,e.score,e.bt)
	//	}
	//}

	for (int32_t i=0; i<num_tables; i++)
		delete table[i];

	SG_FREE(table);
	return result;
}


float64_t* CWeightedDegreePositionStringKernel::extract_w(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
	float64_t* w_result, int32_t num_suppvec, int32_t* IDX, float64_t* alphas)
{
  delete_optimization();
  use_poim_tries=true;
  poim_tries.delete_trees(false);

  // === check
  ASSERT(position_weights_lhs==NULL)
  ASSERT(position_weights_rhs==NULL)
  num_feat=((CStringFeatures<char>*) rhs)->get_max_vector_length();
  ASSERT(num_feat>0)
  ASSERT(alphabet->get_alphabet()==DNA)
  ASSERT(max_degree>0)

  // === general variables
  static const int32_t NUM_SYMS = poim_tries.NUM_SYMS;
  const int32_t seqLen = num_feat;
  float64_t** subs;
  int32_t i;
  int32_t k;
  //int32_t y;

  // === init tables "subs" for substring scores / POIMs
  // --- compute table sizes
  int32_t* offsets;
  int32_t offset;
  offsets = SG_MALLOC(int32_t,  max_degree );
  offset = 0;
  for( k = 0; k < max_degree; ++k ) {
    offsets[k] = offset;
    const int32_t nofsKmers = (int32_t) CMath::pow( NUM_SYMS, k+1 );
    const int32_t tabSize = nofsKmers * seqLen;
    offset += tabSize;
  }
  // --- allocate memory
  const int32_t bigTabSize = offset;
  w_result=SG_MALLOC(float64_t, bigTabSize);
  for (i=0; i<bigTabSize; ++i)
    w_result[i]=0;

  // --- set pointers for tables
  subs = SG_MALLOC(float64_t*,  max_degree );
  ASSERT( subs != NULL )
  for( k = 0; k < max_degree; ++k ) {
    subs[k] = &w_result[ offsets[k] ];
  }
  SG_FREE(offsets);

  // === init trees; extract "w"
  init_optimization( num_suppvec, IDX, alphas, -1);
  poim_tries.POIMs_extract_W( subs, max_degree );

  // === clean; return "subs" as vector
  SG_FREE(subs);
  num_feat = 1;
  num_sym = bigTabSize;
  use_poim_tries=false;
  poim_tries.delete_trees(false);
  return w_result;
}

float64_t* CWeightedDegreePositionStringKernel::compute_POIM(
	int32_t max_degree, int32_t& num_feat, int32_t& num_sym,
	float64_t* poim_result, int32_t num_suppvec, int32_t* IDX,
	float64_t* alphas, float64_t* distrib )
{
  delete_optimization();
  use_poim_tries=true;
  poim_tries.delete_trees(false);

  // === check
  ASSERT(position_weights_lhs==NULL)
  ASSERT(position_weights_rhs==NULL)
  num_feat=((CStringFeatures<char>*) rhs)->get_max_vector_length();
  ASSERT(num_feat>0)
  ASSERT(alphabet->get_alphabet()==DNA)
  ASSERT(max_degree!=0)
  ASSERT(distrib)

  // === general variables
  static const int32_t NUM_SYMS = poim_tries.NUM_SYMS;
  const int32_t seqLen = num_feat;
  float64_t** subs;
  int32_t i;
  int32_t k;

  // === DEBUGGING mode
  //
  // Activated if "max_degree" < 0.
  // Allows to output selected partial score.
  //
  // |max_degree| mod 4
  //   0: substring
  //   1: superstring
  //   2: left overlap
  //   3: right overlap
  //
  const int32_t debug = ( max_degree < 0 ) ? ( abs(max_degree) % 4 + 1 ) : 0;
  if( debug ) {
    max_degree = abs(max_degree) / 4;
    switch( debug ) {
    case 1: {
      printf( "POIM DEBUGGING: substring only (max order=%d)\n", max_degree );
      break;
    }
    case 2: {
      printf( "POIM DEBUGGING: superstring only (max order=%d)\n", max_degree );
      break;
    }
    case 3: {
      printf( "POIM DEBUGGING: left overlap only (max order=%d)\n", max_degree );
      break;
    }
    case 4: {
      printf( "POIM DEBUGGING: right overlap only (max order=%d)\n", max_degree );
      break;
    }
    default: {
      printf( "POIM DEBUGGING: something is wrong (max order=%d)\n", max_degree );
      ASSERT(0)
      break;
    }
    }
  }

  // --- compute table sizes
  int32_t* offsets;
  int32_t offset;
  offsets = SG_MALLOC(int32_t,  max_degree );
  offset = 0;
  for( k = 0; k < max_degree; ++k ) {
    offsets[k] = offset;
    const int32_t nofsKmers = (int32_t) CMath::pow( NUM_SYMS, k+1 );
    const int32_t tabSize = nofsKmers * seqLen;
    offset += tabSize;
  }
  // --- allocate memory
  const int32_t bigTabSize=offset;
  poim_result=SG_MALLOC(float64_t, bigTabSize);
  for (i=0; i<bigTabSize; ++i )
    poim_result[i]=0;

  // --- set pointers for tables
  subs=SG_MALLOC(float64_t*, max_degree);
  for (k=0; k<max_degree; ++k)
    subs[k]=&poim_result[offsets[k]];

  SG_FREE(offsets);

  // === init trees; precalc S, L and R
  init_optimization( num_suppvec, IDX, alphas, -1);
  poim_tries.POIMs_precalc_SLR( distrib );

  // === compute substring scores
  if( debug==0 || debug==1 ) {
    poim_tries.POIMs_extract_W( subs, max_degree );
    for( k = 1; k < max_degree; ++k ) {
      const int32_t nofKmers2 = ( k > 1 ) ? (int32_t) CMath::pow(NUM_SYMS,k-1) : 0;
      const int32_t nofKmers1 = (int32_t) CMath::pow( NUM_SYMS, k );
      const int32_t nofKmers0 = nofKmers1 * NUM_SYMS;
      for( i = 0; i < seqLen; ++i ) {
	float64_t* const subs_k2i1 = ( k>1 && i<seqLen-1 ) ? &subs[k-2][(i+1)*nofKmers2] : NULL;
	float64_t* const subs_k1i1 = ( i < seqLen-1 ) ? &subs[k-1][(i+1)*nofKmers1] : NULL;
	float64_t* const subs_k1i0 = & subs[ k-1 ][ i*nofKmers1 ];
	float64_t* const subs_k0i  = & subs[ k-0 ][ i*nofKmers0 ];
	int32_t y0;
	for( y0 = 0; y0 < nofKmers0; ++y0 ) {
	  const int32_t y1l = y0 / NUM_SYMS;
	  const int32_t y1r = y0 % nofKmers1;
	  const int32_t y2 = y1r / NUM_SYMS;
	  subs_k0i[ y0 ] += subs_k1i0[ y1l ];
	  if( i < seqLen-1 ) {
	    subs_k0i[ y0 ] += subs_k1i1[ y1r ];
	    if( k > 1 ) {
	      subs_k0i[ y0 ] -= subs_k2i1[ y2 ];
	    }
	  }
	}
      }
    }
  }

  // === compute POIMs
  poim_tries.POIMs_add_SLR( subs, max_degree, debug );

  // === clean; return "subs" as vector
  SG_FREE(subs);
  num_feat = 1;
  num_sym = bigTabSize;

  use_poim_tries=false;
  poim_tries.delete_trees(false);

  return poim_result;
}


void CWeightedDegreePositionStringKernel::prepare_POIM2(SGMatrix<float64_t> distrib)
{
	SG_FREE(m_poim_distrib);
	int32_t num_sym=distrib.num_cols;
	int32_t num_feat=distrib.num_rows;
	m_poim_distrib=SG_MALLOC(float64_t, num_sym*num_feat);
	memcpy(m_poim_distrib, distrib.matrix, num_sym*num_feat*sizeof(float64_t));
	m_poim_num_sym=num_sym;
	m_poim_num_feat=num_feat;
}

void CWeightedDegreePositionStringKernel::compute_POIM2(
	int32_t max_degree, CSVM* svm)
{
	ASSERT(svm)
	int32_t num_suppvec=svm->get_num_support_vectors();
	int32_t* sv_idx=SG_MALLOC(int32_t, num_suppvec);
	float64_t* sv_weight=SG_MALLOC(float64_t, num_suppvec);

	for (int32_t i=0; i<num_suppvec; i++)
	{
		sv_idx[i]=svm->get_support_vector(i);
		sv_weight[i]=svm->get_alpha(i);
	}

	if ((max_degree < 1) || (max_degree > 12))
	{
		//SG_WARNING("max_degree out of range 1..12 (%d).\n", max_degree)
		SG_WARNING("max_degree out of range 1..12 (%d). setting to 1.\n", max_degree)
		max_degree=1;
	}

	int32_t num_feat = m_poim_num_feat;
	int32_t num_sym = m_poim_num_sym;
	SG_FREE(m_poim);

	m_poim = compute_POIM(max_degree, num_feat, num_sym, NULL,	num_suppvec, sv_idx,
						  sv_weight, m_poim_distrib);

	ASSERT(num_feat==1)
	m_poim_result_len=num_sym;

	SG_FREE(sv_weight);
	SG_FREE(sv_idx);
}

SGVector<float64_t> CWeightedDegreePositionStringKernel::get_POIM2()
{
	SGVector<float64_t> poim(m_poim, m_poim_result_len, false);
	return poim;
}

void CWeightedDegreePositionStringKernel::cleanup_POIM2()
{
	SG_FREE(m_poim) ;
	m_poim=NULL ;
	SG_FREE(m_poim_distrib) ;
	m_poim_distrib=NULL ;
	m_poim_num_sym=0 ;
	m_poim_num_sym=0 ;
	m_poim_result_len=0 ;
}

void CWeightedDegreePositionStringKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();

	tries=CTrie<DNATrie>(degree);
	poim_tries=CTrie<POIMTrie>(degree);

	if (weights)
		init_block_weights();
}

void CWeightedDegreePositionStringKernel::init()
{
	weights=NULL;
	weights_length=0;
	weights_degree=0;
	position_weights=NULL;
	position_weights_len=0;

	position_weights_lhs=NULL;
	position_weights_lhs_len=0;
	position_weights_rhs=NULL;
	position_weights_rhs_len=0;

	weights_buffer=NULL;
	mkl_stepsize=1;
	degree=1;
	length=0;

	max_shift=0;
	max_mismatch=0;
	seq_length=0;
	shift=NULL;
	shift_len=0;

	block_weights=NULL;
	block_computation=true;
	type=E_EXTERNAL;
	which_degree=-1;
	tries=CTrie<DNATrie>(1);
	poim_tries=CTrie<POIMTrie>(1);

	tree_initialized=false;
	use_poim_tries=false;
	m_poim_distrib=NULL;

	m_poim=NULL;
	m_poim_num_sym=0;
	m_poim_num_feat=0;
	m_poim_result_len=0;

	alphabet=NULL;

	properties |= KP_LINADD | KP_KERNCOMBINATION | KP_BATCHEVALUATION;

	set_normalizer(new CSqrtDiagKernelNormalizer());

	m_parameters->add_matrix(&weights, &weights_degree, &weights_length,
			"weights", "WD Kernel weights.");
	m_parameters->add_vector(&position_weights, &position_weights_len,
			"position_weights",
			"Weights per position.");
	m_parameters->add_vector(&position_weights_lhs, &position_weights_lhs_len,
			"position_weights_lhs",
			"Weights per position left hand side.");
	m_parameters->add_vector(&position_weights_rhs, &position_weights_rhs_len,
			"position_weights_rhs",
			"Weights per position right hand side.");
	m_parameters->add_vector(&shift, &shift_len,
			"shift",
			"Shift Vector.");
	SG_ADD(&max_shift, "max_shift", "Maximal shift.", MS_AVAILABLE);
	SG_ADD(&mkl_stepsize, "mkl_stepsize", "MKL step size.", MS_AVAILABLE);
	SG_ADD(&degree, "degree", "Order of WD kernel.", MS_AVAILABLE);
	SG_ADD(&max_mismatch, "max_mismatch",
			"Number of allowed mismatches.", MS_AVAILABLE);
	SG_ADD(&block_computation, "block_computation",
			"If block computation shall be used.", MS_NOT_AVAILABLE);
	SG_ADD((machine_int_t*) &type, "type",
			"WeightedDegree kernel type.", MS_AVAILABLE);
	SG_ADD(&which_degree, "which_degree",
			"The selected degree. All degrees are used by default (for value -1).",
			MS_AVAILABLE);
	SG_ADD((CSGObject**) &alphabet, "alphabet",
			"Alphabet of Features.", MS_NOT_AVAILABLE);
}
