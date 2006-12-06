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

// HMM.cpp: implementation of the CDynProg class.
// $Id$
//////////////////////////////////////////////////////////////////////

#include "structure/DynProg.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "lib/config.h"
#include "features/StringFeatures.h"
#include "features/CharFeatures.h"
#include "features/Alphabet.h"
#include "structure/Plif.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"
#include "lib/fibheap.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>

#ifdef SUNOS
extern "C" int	finite(double);
#endif

#define USEHEAP 0
#define USEORIGINALLIST 0
#define USEFIXEDLENLIST 2
//#define USE_TMP_ARRAYCLASS

static INT word_degree_default[4]={3,4,5,6} ;
static INT cum_num_words_default[5]={0,64,320,1344,5440} ;
static INT num_words_default[4]={64,256,1024,4096} ;

CDynProg::CDynProg()
	: m_seq(1,1), m_pos(1), m_orf_info(1,2), m_segment_sum_weights(1,1), m_plif_list(1), 
	  m_PEN(1,1), m_PEN_state_signals(1,2), 
	  m_genestr(1), m_dict_weights(1,1), m_segment_loss(2, 1,1), m_segment_ids_mask(1,1),
	  m_scores(1), m_states(1,1), m_positions(1,1),
	  transition_matrix_a_id(1,1), transition_matrix_a(1,1), transition_matrix_a_deriv(1,1), 
	  initial_state_distribution_p(1), initial_state_distribution_p_deriv(1), 
	  end_state_distribution_q(1), end_state_distribution_q_deriv(1), 
	  dict_weights(1,1), dict_weights_array(dict_weights.get_array()),

	  // multi svm
	  num_degrees(4), 
	  num_svms(8), 
	  word_degree(word_degree_default, num_degrees, true, true),
	  cum_num_words(cum_num_words_default, num_degrees+1, true, true),
	  cum_num_words_array(cum_num_words.get_array()),
	  num_words(num_words_default, num_degrees, true, true),
	  num_words_array(num_words.get_array()),
	  word_used(num_degrees, num_words[num_degrees-1]),
	  word_used_array(word_used.get_array()),
	  svm_values_unnormalized(num_degrees, num_svms),
	  svm_pos_start(num_degrees),
	  num_unique_words(num_degrees),
	  svm_arrays_clean(true),

	  // single svm
	  num_svms_single(1),
	  word_degree_single(1),
	  num_words_single(4), 
	  word_used_single(num_words_single),
	  svm_value_unnormalized_single(num_svms_single),
	  num_unique_words_single(0),

	  max_a_id(0)
{
	trans_list_forward = NULL ;
	trans_list_forward_cnt = NULL ;
	trans_list_forward_val = NULL ;
	trans_list_forward_id = NULL ;
	trans_list_len = 0 ;

	mem_initialized = true ;

	this->N=1;
	m_step=0 ;

#ifdef ARRAY_STATISTICS
	word_degree.set_name("word_degree") ;
#endif

}

CDynProg::~CDynProg()
{
	if (trans_list_forward_cnt)
	  delete[] trans_list_forward_cnt ;
	if (trans_list_forward)
	{
	    for (INT i=0; i<trans_list_len; i++)
			if (trans_list_forward[i])
				delete[] trans_list_forward[i] ;
	    delete[] trans_list_forward ;
	}
	if (trans_list_forward_val)
	{
	    for (INT i=0; i<trans_list_len; i++)
			if (trans_list_forward_val[i])
				delete[] trans_list_forward_val[i] ;
	    delete[] trans_list_forward_val ;
	}
	if (trans_list_forward_id)
	{
	    for (INT i=0; i<trans_list_len; i++)
			if (trans_list_forward_id[i])
				delete[] trans_list_forward_id[i] ;
	    delete[] trans_list_forward_id ;
	}
}

////////////////////////////////////////////////////////////////////////////////

void CDynProg::set_N(INT p_N)
{
	N=p_N ;

	transition_matrix_a_id.resize_array(N,N) ;
	transition_matrix_a.resize_array(N,N) ;
	transition_matrix_a_deriv.resize_array(N,N) ;
	initial_state_distribution_p.resize_array(N) ;
	initial_state_distribution_p_deriv.resize_array(N) ;
	end_state_distribution_q.resize_array(N);
	end_state_distribution_q_deriv.resize_array(N) ;

	m_orf_info.resize_array(N,2) ;
	m_PEN.resize_array(N,N) ;
	m_PEN_state_signals.resize_array(N,2) ;
}

void CDynProg::set_p(DREAL *p, INT p_N) 
{
	ASSERT(p_N==N) ;
	//m_orf_info.resize_array(p_N,2) ;
	//m_PEN.resize_array(p_N,p_N) ;

	initial_state_distribution_p.set_array(p, p_N, true, true) ;
}

void CDynProg::set_q(DREAL *q, INT p_N) 
{
	ASSERT(p_N==N) ;
	end_state_distribution_q.set_array(q, p_N, true, true) ;
}

void CDynProg::set_a(DREAL *a, INT p_M, INT p_N) 
{
	ASSERT(p_N==N) ;
	ASSERT(p_M==p_N) ;
	transition_matrix_a.set_array(a, p_N, p_N, true, true) ;
	transition_matrix_a_deriv.resize_array(p_N, p_N) ;
}

void CDynProg::set_a_id(INT *a, INT p_M, INT p_N) 
{
	ASSERT(p_N==N) ;
	ASSERT(p_M==p_N) ;
	transition_matrix_a_id.set_array(a, p_N, p_N, true, true) ;
	max_a_id = 0 ;
	for (INT i=0; i<p_N; i++)
		for (INT j=0; j<p_N; j++)
			max_a_id = CMath::max(max_a_id, transition_matrix_a_id.element(i,j)) ;
}

void CDynProg::set_a_trans(DREAL *a_trans, INT num_trans, INT p_N) 
{
	ASSERT((p_N==3) || (p_N==4)) ;

	delete[] trans_list_forward ;
	delete[] trans_list_forward_cnt ;
	delete[] trans_list_forward_val ;
	delete[] trans_list_forward_id ;

	trans_list_forward = NULL ;
	trans_list_forward_cnt = NULL ;
	trans_list_forward_val = NULL ;
	trans_list_len = 0 ;

	transition_matrix_a.zero() ;
	transition_matrix_a_id.zero() ;

	mem_initialized = true ;

	trans_list_forward_cnt=NULL ;
	trans_list_len = N ;
	trans_list_forward = new T_STATES*[N] ;
	trans_list_forward_cnt = new T_STATES[N] ;
	trans_list_forward_val = new DREAL*[N] ;
	trans_list_forward_id = new INT*[N] ;
	
	INT start_idx=0;
	for (INT j=0; j<N; j++)
	{
		INT old_start_idx=start_idx;

		while (start_idx<num_trans && a_trans[start_idx+num_trans]==j)
		{
			start_idx++;
			
			if (start_idx>1 && start_idx<num_trans)
				ASSERT(a_trans[start_idx+num_trans-1] <= a_trans[start_idx+num_trans]);
		}
		
		if (start_idx>1 && start_idx<num_trans)
			ASSERT(a_trans[start_idx+num_trans-1] <= a_trans[start_idx+num_trans]);
		
		INT len=start_idx-old_start_idx;
		ASSERT(len>=0);
		
		trans_list_forward_cnt[j] = 0 ;
		
		if (len>0)
		{
			trans_list_forward[j]     = new T_STATES[len] ;
			trans_list_forward_val[j] = new DREAL[len] ;
			trans_list_forward_id[j] = new INT[len] ;
		}
		else
		{
			trans_list_forward[j]     = NULL;
			trans_list_forward_val[j] = NULL;
			trans_list_forward_id[j]  = NULL;
		}
	}
	
	for (INT i=0; i<num_trans; i++)
	{
		INT to   = (INT)a_trans[i] ;
		INT from = (INT)a_trans[i+num_trans] ;
		DREAL val = a_trans[i+num_trans*2] ;
		INT id = 0 ;
		if (p_N==4)
			id = (INT)a_trans[i+num_trans*3] ;
		
		ASSERT(from>=0 && from<N) ;
		ASSERT(to>=0 && to<N) ;
		
		trans_list_forward[from][trans_list_forward_cnt[from]]=to ;
		trans_list_forward_val[from][trans_list_forward_cnt[from]]=val ;
		trans_list_forward_id[from][trans_list_forward_cnt[from]]=id ;
		trans_list_forward_cnt[from]++ ;
		transition_matrix_a.element(from, to) = val ;
		transition_matrix_a_id.element(from, to) = id ;
	} ;

	max_a_id = 0 ;
	for (INT i=0; i<p_N; i++)
		for (INT j=0; j<p_N; j++)
			max_a_id = CMath::max(max_a_id, transition_matrix_a_id.element(i,j)) ;
	max_a_id = 8 ;
}

void CDynProg::init_svm_arrays(INT p_num_degrees, INT p_num_svms)
{
	svm_arrays_clean=false ;

	word_degree.resize_array(num_degrees) ;

	cum_num_words.resize_array(num_degrees+1) ;
	cum_num_words_array=cum_num_words.get_array() ;

	num_words.resize_array(num_degrees) ;
	num_words_array=num_words.get_array() ;
	
	svm_values_unnormalized.resize_array(num_degrees, num_svms) ;
	svm_pos_start.resize_array(num_degrees) ;
	num_unique_words.resize_array(num_degrees) ;
} 


void CDynProg::init_word_degree_array(INT * p_word_degree_array, INT num_elem)
{
	svm_arrays_clean=false ;

	word_degree.resize_array(num_degrees) ;
	ASSERT(num_degrees==num_elem) ;

	for (INT i=0; i<num_degrees; i++)
		word_degree[i]=p_word_degree_array[i] ;

} 

void CDynProg::init_cum_num_words_array(INT * p_cum_num_words_array, INT num_elem)
{
	svm_arrays_clean=false ;

	cum_num_words.resize_array(num_degrees+1) ;
	cum_num_words_array=cum_num_words.get_array() ;
	ASSERT(num_degrees+1==num_elem) ;

	for (INT i=0; i<num_degrees+1; i++)
		cum_num_words[i]=p_cum_num_words_array[i] ;
} 

void CDynProg::init_num_words_array(INT * p_num_words_array, INT num_elem)
{
	svm_arrays_clean=false ;

	num_words.resize_array(num_degrees) ;
	num_words_array=num_words.get_array() ;
	ASSERT(num_degrees==num_elem) ;

	for (INT i=0; i<num_degrees; i++)
		num_words[i]=p_num_words_array[i] ;

	word_used.resize_array(num_degrees, num_words[num_degrees-1]) ;
	word_used_array=word_used.get_array() ;
} 

bool CDynProg::check_svm_arrays()
{
	if ((word_degree.get_dim1()==num_degrees) &&
		(cum_num_words.get_dim1()==num_degrees+1) &&
		(num_words.get_dim1()==num_degrees) &&
		(word_used.get_dim1()==num_degrees) &&
		(word_used.get_dim2()==num_words[num_degrees-1]) &&
		(svm_values_unnormalized.get_dim1()==num_degrees) &&
		(svm_values_unnormalized.get_dim2()==num_svms) &&
		(svm_pos_start.get_dim1()==num_degrees) &&
		(num_unique_words.get_dim1()==num_degrees))
	{
		svm_arrays_clean=true ;
		return true ;
	}
	else
	{
		svm_arrays_clean=false ;
		return false ;	
	}
}

void CDynProg::best_path_set_seq(DREAL *seq, INT p_N, INT seq_len) 
{
	if (!svm_arrays_clean)
	{
		CIO::message(M_ERROR, "SVM arrays not clean") ;
		return ;
	} ;

	ASSERT(p_N==N) ;
	ASSERT(initial_state_distribution_p.get_dim1()==N) ;
	ASSERT(end_state_distribution_q.get_dim1()==N) ;	
	
	m_seq.set_array(seq, N, seq_len, true, true) ;
	this->N=N ;

	m_call=3 ;
	m_step=2 ;
}

void CDynProg::best_path_set_pos(INT *pos, INT seq_len)  
{
	if (m_step!=2)
		CIO::message(M_ERROR, "please call best_path_set_seq first\n") ;
	
	if (seq_len!=m_seq.get_dim2())
		CIO::message(M_ERROR, "pos size does not match previous info %i!=%i\n", seq_len, m_seq.get_dim2()) ;

	m_pos.set_array(pos, seq_len, true, true) ;

	m_step=3 ;
}

void CDynProg::best_path_set_orf_info(INT *orf_info, INT m, INT n) 
{
	if (m_step!=3)
		CIO::message(M_ERROR, "please call best_path_set_pos first\n") ;
		
	if (m!=N)
		CIO::message(M_ERROR, "orf_info size does not match previous info %i!=%i\n", m, N) ;
	if (n!=2)
		CIO::message(M_ERROR, "orf_info size incorrect %i!=2\n", n) ;
	m_orf_info.set_array(orf_info, m, n, true, true) ;
	
	m_call=1 ;
	m_step=4 ;
}

void CDynProg::best_path_set_segment_sum_weights(DREAL *segment_sum_weights, INT num_states, INT seq_len) 
{
	if (m_step!=3)
		CIO::message(M_ERROR, "please call best_path_set_pos first\n") ;
		
	if (num_states!=N)
		CIO::message(M_ERROR, "segment_sum_weights size does not match previous info %i!=%i\n", num_states, N) ;
	if (seq_len!=m_pos.get_dim1())
		CIO::message(M_ERROR, "segment_sum_weights size incorrect %i!=%i\n", seq_len, m_pos.get_dim1()) ;

	m_segment_sum_weights.set_array(segment_sum_weights, num_states, seq_len, true, true) ;
	
	m_call=2 ;
	m_step=4 ;
}

void CDynProg::best_path_set_plif_list(CPlif **plif_list, INT num_plif) 
{
	if (m_step!=4)
		CIO::message(M_ERROR, "please call best_path_set_orf_info or best_path_segment_sum_weights first\n") ;

	m_plif_list.set_array(plif_list, num_plif, true, true) ;

	m_step=5 ;
}

void CDynProg::best_path_set_plif_id_matrix(INT *plif_id_matrix, INT m, INT n) 
{
	if (m_step!=5)
		CIO::message(M_ERROR, "please call best_path_set_plif_list first\n") ;

	if ((m!=N) || (n!=N))
		CIO::message(M_ERROR, "plif_id_matrix size does not match previous info %i!=%i or %i!=%i\n", m, N, n, N) ;

	CArray2<INT> id_matrix(plif_id_matrix, N, N, false, false) ;
	m_PEN.resize_array(N, N) ;
	for (INT i=0; i<N; i++)
		for (INT j=0; j<N; j++)
			if (id_matrix.element(i,j)>=0)
				m_PEN.element(i,j)=m_plif_list[id_matrix.element(i,j)] ;
			else
				m_PEN.element(i,j)=NULL ;

	m_step=6 ;
}

void CDynProg::best_path_set_plif_state_signal_matrix(INT *plif_id_matrix, INT m, INT n) 
{
	if (m_step!=6)
		CIO::message(M_ERROR, "please call best_path_set_plif_id_matrix first\n") ;
	
	if ((m!=N) || (n!=2))
		CIO::message(M_ERROR, "plif_state_signal_matrix size does not match previous info %i!=%i or %i!=%i\n", m, N, n, 2) ;

	CArray2<INT> id_matrix(plif_id_matrix, N, 2, false, false) ;
	m_PEN_state_signals.resize_array(N,2) ;
	for (INT i=0; i<N; i++)
		for (INT j=0; j<2; j++)
			if (id_matrix.element(i,j)>=0)
				m_PEN_state_signals.element(i,j)=m_plif_list[id_matrix.element(i,j)] ;
			else
				m_PEN_state_signals.element(i,j)=NULL ;

	m_step=6 ;
}

void CDynProg::best_path_set_genestr(CHAR* genestr, INT genestr_len)
{
	if (m_step!=6)
		CIO::message(M_ERROR, "please call best_path_set_plif_id_matrix first\n") ;

	m_genestr.set_array(genestr, genestr_len, true, true) ;

	m_step=7 ;
}

void CDynProg::best_path_set_dict_weights(DREAL* dictionary_weights, INT dict_len, INT n) 
{
	if (m_step!=7)
		CIO::message(M_ERROR, "please call best_path_set_genestr first\n") ;

	if (num_svms!=n)
		CIO::message(M_ERROR, "dict_weights array does not match num_svms=%i!=%i\n", num_svms, n) ;

	m_dict_weights.set_array(dictionary_weights, dict_len, num_svms, true, true) ;

	// initialize, so it does not bother when not used
	m_segment_loss.resize_array(2, max_a_id+1, max_a_id+1) ;
	m_segment_loss.zero() ;
	m_segment_ids_mask.resize_array(2, m_seq.get_dim2()) ;
	m_segment_ids_mask.zero() ;

	m_step=8 ;
}

void CDynProg::best_path_set_segment_loss(DREAL* segment_loss, INT m, INT n) 
{
	// here we need two matrices. Store it in one: 2N x N
	if (m!=2*n)
		CIO::message(M_ERROR, "segment_loss should be 2 x quadratic matrix: %i!=%i\n", 2*m, n) ;

	//if (n!=max_a_id+1)
	//CIO::message(M_ERROR, "segment_loss size should match max_a_id: %i!=%i\n", n, max_a_id+1) ;

	m_segment_loss.set_array(segment_loss, 2, m/2, n, true, true) ;
}

void CDynProg::best_path_set_segment_ids_mask(INT* segment_ids_mask, INT m, INT n) 
{
	if (m!=2)// || n!=m_seq.get_dim2())
		CIO::message(M_ERROR, "segment_ids_mask should be a 2 x seq_len matrix: %i!=2 and %i!=%i\n", m, m_seq.get_dim2(), n) ;

	m_segment_ids_mask.set_array(segment_ids_mask, m, n, true, true) ;
}


void CDynProg::best_path_call(INT nbest, bool use_orf) 
{
	if (m_step!=8)
		CIO::message(M_ERROR, "please call best_path_set_dict_weights first\n") ;
	if (m_call!=1)
		CIO::message(M_ERROR, "please call best_path_set_orf_info first\n") ;
	ASSERT(N==m_seq.get_dim1()) ;
	ASSERT(m_seq.get_dim2()==m_pos.get_dim1()) ;

	m_scores.resize_array(nbest) ;
	m_states.resize_array(nbest, m_seq.get_dim2()) ;
	m_positions.resize_array(nbest, m_seq.get_dim2()) ;

	DREAL* PEN_values = NULL ;
	DREAL * PEN_input_values = NULL ;
	INT num_PEN_id = 0 ;

	m_call=1 ;

	best_path_trans(m_seq.get_array(), m_seq.get_dim2(), m_pos.get_array(), m_orf_info.get_array(),
					m_PEN.get_array(), m_PEN_state_signals.get_array(), 
					m_genestr.get_array(), m_genestr.get_dim1(),
					nbest, 
					m_scores.get_array(), m_states.get_array(), m_positions.get_array(),
					m_dict_weights.get_array(), m_dict_weights.get_dim1()*m_dict_weights.get_dim2(),
					PEN_values, PEN_input_values, 
					num_PEN_id, use_orf) ;

	delete[] PEN_values ;
	delete[] PEN_input_values ;

	m_step=9 ;
}

void CDynProg::best_path_2struct_call(INT nbest) 
{
	if (m_step!=8)
		CIO::message(M_ERROR, "please call best_path_set_orf_dict_weights first\n") ;
	if (m_call!=2)
		CIO::message(M_ERROR, "please call best_path_set_segment_sum_weights first\n") ;
	ASSERT(N==m_seq.get_dim1()) ;
	ASSERT(m_seq.get_dim2()==m_pos.get_dim1()) ;
	
	m_scores.resize_array(nbest) ;
	m_states.resize_array(nbest, m_seq.get_dim2()) ;
	m_positions.resize_array(nbest, m_seq.get_dim2()) ;

	m_call=2 ;

	DREAL* PEN_values = NULL ;
	DREAL * PEN_input_values = NULL ;
	INT num_PEN_id = 0 ;

	best_path_2struct(m_seq.get_array(), m_seq.get_dim2(), m_pos.get_array(), 
					  m_PEN.get_array(), 
					  m_genestr.get_array(), m_genestr.get_dim1(),
					  nbest, 
					  m_scores.get_array(), m_states.get_array(), m_positions.get_array(),
					  m_dict_weights.get_array(), m_dict_weights.get_dim1()*m_dict_weights.get_dim2(), 
					  m_segment_sum_weights.get_array(),
					  PEN_values, PEN_input_values, num_PEN_id) ;

	delete[] PEN_values ;
	delete[] PEN_input_values ;

	m_step=9 ;
}

void CDynProg::best_path_simple_call(INT nbest) 
{
	if (m_step!=2)
		CIO::message(M_ERROR, "please call best_path_set_seq first\n") ;
	if (m_call!=3)
		CIO::message(M_ERROR, "please call best_path_set_seq first\n") ;
	ASSERT(N==m_seq.get_dim1()) ;

	m_scores.resize_array(nbest) ;
	m_states.resize_array(nbest, m_seq.get_dim2()) ;

	m_call=3 ;

	best_path_trans_simple(m_seq.get_array(), m_seq.get_dim2(), 
						   nbest, 
						   m_scores.get_array(), m_states.get_array()) ;
	
	m_step=9 ;
}


void CDynProg::best_path_get_scores(DREAL **scores, INT *m) 
{
	if (m_step!=9)
		CIO::message(M_ERROR, "please call best_path*_call first\n") ;

	*scores=m_scores.get_array() ;
	*m=m_scores.get_dim1() ;

	m_step=10 ;
}

void CDynProg::best_path_get_states(INT **states, INT *m, INT *n) 
{
	if (m_step!=10)
		CIO::message(M_ERROR, "please call best_path_get_score first\n") ;
	
	*states=m_states.get_array() ;
	*m=m_states.get_dim1() ;
	*n=m_states.get_dim2() ;

	m_step=11 ;
}

void CDynProg::best_path_get_positions(INT **positions, INT *m, INT *n) 
{
	if (m_step!=11)
		CIO::message(M_ERROR, "please call best_path_get_positions first\n") ;
	if (m_call==3)
		CIO::message(M_ERROR, "no position information for best_path_simple\n") ;
	
	*positions=m_positions.get_array() ;
	*m=m_positions.get_dim1() ;
	*n=m_positions.get_dim2() ;
}

////////////////////////////////////////////////////////////////////////////////

DREAL CDynProg::best_path_no_b(INT max_iter, INT &best_iter, INT *my_path)
{
	CArray2<T_STATES> psi(max_iter, N) ;
	CArray<DREAL>* delta = new CArray<DREAL>(N) ;
	CArray<DREAL>* delta_new = new CArray<DREAL>(N) ;
	
	{ // initialization
		for (INT i=0; i<N; i++)
		{
			delta->element(i) = get_p(i) ;
			psi.element(0, i)= 0 ;
		}
	} 
	
	DREAL best_iter_prob = CMath::ALMOST_NEG_INFTY ;
	best_iter = 0 ;
	
	// recursion
	for (INT t=1; t<max_iter; t++)
	{
		CArray<DREAL>* dummy;
		INT NN=N ;
		for (INT j=0; j<NN; j++)
		{
			DREAL maxj = delta->element(0) + transition_matrix_a.element(0,j);
			INT argmax=0;
			
			for (INT i=1; i<NN; i++)
			{
				DREAL temp = delta->element(i) + transition_matrix_a.element(i,j);
				
				if (temp>maxj)
				{
					maxj=temp;
					argmax=i;
				}
			}
			delta_new->element(j)=maxj ;
			psi.element(t, j)=argmax ;
		}
		
		dummy=delta;
		delta=delta_new;
		delta_new=dummy;	//switch delta/delta_new
		
		{ //termination
			DREAL maxj=delta->element(0)+get_q(0);
			INT argmax=0;
			
			for (INT i=1; i<N; i++)
			{
				DREAL temp=delta->element(i)+get_q(i);
				
				if (temp>maxj)
				{
					maxj=temp;
					argmax=i;
				}
			}
			//pat_prob=maxj;
			
			if (maxj>best_iter_prob)
			{
				my_path[t]=argmax;
				best_iter=t ;
				best_iter_prob = maxj ;
			} ;
		} ;
	}

	
	{ //state sequence backtracking
		for (INT t = best_iter; t>0; t--)
		{
			my_path[t-1]=psi.element(t, my_path[t]);
		}
	}

	delete delta ;
	delete delta_new ;
	
	return best_iter_prob ;
}

void CDynProg::best_path_no_b_trans(INT max_iter, INT &max_best_iter, short int nbest, DREAL *prob_nbest, INT *my_paths)
{
	//T_STATES *psi=new T_STATES[max_iter*N*nbest] ;
	CArray3<T_STATES> psi(max_iter, N, nbest) ;
	CArray3<short int> ktable(max_iter, N, nbest) ;
	CArray2<short int> ktable_ends(max_iter, nbest) ;

	CArray<DREAL> tempvv(nbest*N) ;
	CArray<INT> tempii(nbest*N) ;

	CArray2<T_STATES> path_ends(max_iter, nbest) ;
	CArray2<DREAL> *delta=new CArray2<DREAL>(N, nbest) ;
	CArray2<DREAL> *delta_new=new CArray2<DREAL>(N, nbest) ;
	CArray2<DREAL> delta_end(max_iter, nbest) ;

	CArray2<INT> paths(max_iter, nbest) ;
	paths.set_array(my_paths, max_iter, nbest, false, false) ;

	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			delta->element(i,0) = get_p(i) ;
			for (short int k=1; k<nbest; k++)
			{
				delta->element(i,k)=-CMath::INFTY ;
				ktable.element(0,i,k)=0 ;
			}
		}
	}
	
	// recursion
	for (INT t=1; t<max_iter; t++)
	{
		CArray2<DREAL>* dummy=NULL;

		for (T_STATES j=0; j<N; j++)
		{
			const T_STATES num_elem   = trans_list_forward_cnt[j] ;
			const T_STATES *elem_list = trans_list_forward[j] ;
			const DREAL *elem_val = trans_list_forward_val[j] ;
			
			INT list_len=0 ;
			for (short int diff=0; diff<nbest; diff++)
			{
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					
					tempvv.element(list_len) = -(delta->element(ii,diff) + elem_val[i]) ;
					tempii.element(list_len) = diff*N + ii ;
					list_len++ ;
				}
			}
			CMath::qsort(tempvv.get_array(), tempii.get_array(), list_len) ;
			
			for (short int k=0; k<nbest; k++)
			{
				if (k<list_len)
				{
					delta_new->element(j,k)  = -tempvv[k] ;
					psi.element(t,j,k)      = (tempii[k]%N) ;
					ktable.element(t,j,k)   = (tempii[k]-(tempii[k]%N))/N ;
				}
				else
				{
					delta_new->element(j,k)  = -CMath::INFTY ;
					psi.element(t,j,k)      = 0 ;
					ktable.element(t,j,k)   = 0 ;
				}
			}
		}
		
		dummy=delta;
		delta=delta_new;
		delta_new=dummy;	//switch delta/delta_new
		
		{ //termination
			INT list_len = 0 ;
			for (short int diff=0; diff<nbest; diff++)
			{
				for (T_STATES i=0; i<N; i++)
				{
					tempvv.element(list_len) = -(delta->element(i,diff)+get_q(i));
					tempii.element(list_len) = diff*N + i ;
					list_len++ ;
				}
			}
			CMath::qsort(tempvv.get_array(), tempii.get_array(), list_len) ;
			
			for (short int k=0; k<nbest; k++)
			{
				delta_end.element(t-1,k) = -tempvv[k] ;
				path_ends.element(t-1,k) = (tempii[k]%N) ;
				ktable_ends.element(t-1,k) = (tempii[k]-(tempii[k]%N))/N ;
			}
		}
	}
	
	{ //state sequence backtracking
		max_best_iter=0 ;
		
		CArray<DREAL> sort_delta_end(max_iter*nbest) ;
		CArray<short int> sort_k(max_iter*nbest) ;
		CArray<INT> sort_t(max_iter*nbest) ;
		CArray<INT> sort_idx(max_iter*nbest) ;
		
		INT i=0 ;
		for (INT iter=0; iter<max_iter-1; iter++)
			for (short int k=0; k<nbest; k++)
			{
				sort_delta_end[i]=-delta_end.element(iter,k) ;
				sort_k[i]=k ;
				sort_t[i]=iter+1 ;
				sort_idx[i]=i ;
				i++ ;
			}
		
		CMath::qsort(sort_delta_end.get_array(), sort_idx.get_array(), (max_iter-1)*nbest) ;

		for (short int n=0; n<nbest; n++)
		{
			short int k=sort_k[sort_idx[n]] ;
			INT iter=sort_t[sort_idx[n]] ;
			prob_nbest[n]=-sort_delta_end[n] ;

			if (iter>max_best_iter)
				max_best_iter=iter ;
			
			ASSERT(k<nbest) ;
			ASSERT(iter<max_iter) ;
			
			paths.element(iter,n) = path_ends.element(iter-1, k) ;
			short int q   = ktable_ends.element(iter-1, k) ;
			
			for (INT t = iter; t>0; t--)
			{
				paths.element(t-1,n)=psi.element(t, paths.element(t,n), q);
				q = ktable.element(t, paths.element(t,n), q) ;
			}
		}
	}

	delete delta ;
	delete delta_new ;
}


void CDynProg::translate_from_single_order(WORD* obs, INT sequence_length, 
										   INT start, INT order, 
										   INT max_val)
{
	INT i,j;
	WORD value=0;
	
	for (i=sequence_length-1; i>= ((int) order)-1; i--)	//convert interval of size T
	{
		value=0;
		for (j=i; j>=i-((int) order)+1; j--)
			value= (value >> max_val) | (obs[j] << (max_val * (order-1)));
		
		obs[i]= (WORD) value;
	}
	
	for (i=order-2;i>=0;i--)
	{
		value=0;
		for (j=i; j>=i-order+1; j--)
		{
			value= (value >> max_val);
			if (j>=0)
				value|=obs[j] << (max_val * (order-1));
		}
		obs[i]=value;
		//ASSERT(value<num_words) ;
	}
	if (start>0)
		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
}

void CDynProg::reset_svm_value(INT pos, INT & last_svm_pos, DREAL * svm_value) 
{
	for (int i=0; i<num_words_single; i++)
		word_used_single[i]=false ;
	for (INT s=0; s<num_svms; s++)
		svm_value_unnormalized_single[s] = 0 ;
	for (INT s=0; s<num_svms; s++)
		svm_value[s] = 0 ;
	last_svm_pos = pos - 6+1 ;
	num_unique_words_single=0 ;
}

void CDynProg::extend_svm_value(WORD* wordstr, INT pos, INT &last_svm_pos, DREAL* svm_value) 
{
	bool did_something = false ;
	for (int i=last_svm_pos-1; (i>=pos) && (i>=0); i--)
	{
		if (wordstr[i]>=num_words_single)
			CIO::message(M_DEBUG, "wordstr[%i]=%i\n", i, wordstr[i]) ;
		
		if (!word_used_single[wordstr[i]])
		{
			for (INT s=0; s<num_svms_single; s++)
				svm_value_unnormalized_single[s]+=dict_weights.element(wordstr[i],s) ;
			
			word_used_single[wordstr[i]]=true ;
			num_unique_words_single++ ;
			did_something=true ;
		}
	} ;
	if (num_unique_words_single>0)
	{
		last_svm_pos=pos ;
		if (did_something)
			for (INT s=0; s<num_svms; s++)
				svm_value[s]= svm_value_unnormalized_single[s]/sqrt((double)num_unique_words_single) ;  // full normalization
	}
	else
	{
		// what should I do?
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
}


void CDynProg::reset_segment_sum_value(INT num_states, INT pos, INT & last_segment_sum_pos, DREAL * segment_sum_value) 
{
	for (INT s=0; s<num_states; s++)
		segment_sum_value[s] = 0 ;
	last_segment_sum_pos = pos ;
	//fprintf(stderr, "start: %i\n", pos) ;
}

void CDynProg::extend_segment_sum_value(DREAL *segment_sum_weights, INT seqlen, INT num_states,
							  INT pos, INT &last_segment_sum_pos, DREAL* segment_sum_value) 
{
	for (int i=last_segment_sum_pos-1; (i>=pos) && (i>=0); i--)
	{
		for (INT s=0; s<num_states; s++)
			segment_sum_value[s] += segment_sum_weights[i*num_states+s] ;
	} ;
	//fprintf(stderr, "extend %i: %f\n", pos, segment_sum_value[0]) ;
	last_segment_sum_pos = pos ;
}


void CDynProg::best_path_2struct(const DREAL *seq_array, INT seq_len, const INT *pos,
							 CPlif **Plif_matrix, 
							 const char *genestr, INT genestr_len,
							 short int nbest, 
							 DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
							 DREAL *dictionary_weights, INT dict_len, DREAL *segment_sum_weights, 
							 DREAL *&PEN_values, DREAL *&PEN_input_values, INT &num_PEN_id)
{
	const INT default_look_back = 100 ;
	INT max_look_back = default_look_back ;
	bool use_svm = false ;
	ASSERT(dict_len==num_svms*num_words_single) ;
	dict_weights.set_array(dictionary_weights, dict_len, num_svms, false, false) ;
	dict_weights_array=dict_weights.get_array() ;

	CArray2<CPlif*> PEN(Plif_matrix, N, N, false) ;
	CArray2<DREAL> seq((DREAL *)seq_array, N, seq_len, false) ;
	
	DREAL svm_value[num_svms] ;
	DREAL segment_sum_value[N] ;
	
	{ // initialize svm_svalue
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
	{ // determine maximal length of look-back
		for (INT i=0; i<N; i++)
			for (INT j=0; j<N; j++)
			{
				CPlif *penij=PEN.element(i,j) ;
				while (penij!=NULL)
				{
					if (penij->get_max_len()>max_look_back)
						max_look_back=penij->get_max_len() ;
					if (penij->get_use_svm())
						use_svm=true ;
					if (penij->get_id()+1>num_PEN_id)
						num_PEN_id=penij->get_id()+1 ;
					penij=penij->get_next_pen() ;
				} 
			}
	}
	max_look_back = CMath::min(genestr_len, max_look_back) ;
	//fprintf(stderr,"use_svm=%i\n", use_svm) ;
	fprintf(stderr,"max_look_back=%i\n", max_look_back) ;
	
	const INT look_back_buflen = (max_look_back+1)*nbest*N ;
	//fprintf(stderr,"look_back_buflen=%i\n", look_back_buflen) ;
	const DREAL mem_use = (DREAL)(seq_len*N*nbest*(sizeof(T_STATES)+sizeof(short int)+sizeof(INT))+
								look_back_buflen*(2*sizeof(DREAL)+sizeof(INT))+
								seq_len*(sizeof(T_STATES)+sizeof(INT))+
								genestr_len*sizeof(bool))/(1024*1024)
		 ;
    bool is_big = (mem_use>200) || (seq_len>5000) ;

	if (is_big)
	{
		CIO::message(M_DEBUG,"calling best_path_2struct: seq_len=%i, N=%i, lookback=%i nbest=%i\n", 
					 seq_len, N, max_look_back, nbest) ;
		CIO::message(M_DEBUG,"allocating %1.2fMB of memory\n", 
					 mem_use) ;
	}
	ASSERT(nbest<32000) ;
		
	CArray3<DREAL> delta(max_look_back+1, N, nbest) ;
	CArray3<T_STATES> psi(seq_len,N,nbest) ;
	CArray3<short int> ktable(seq_len,N,nbest) ;
	CArray3<INT> ptable(seq_len,N,nbest) ;

	CArray<DREAL> delta_end(nbest) ;
	CArray<T_STATES> path_ends(nbest) ;
	CArray<short int> ktable_end(nbest) ;

	CArray<DREAL> tempvv(look_back_buflen) ;
	CArray<INT> tempii(look_back_buflen) ;

	CArray<T_STATES> state_seq(seq_len) ;
	CArray<INT> pos_seq(seq_len) ;

	// translate to words, if svm is used
	WORD* wordstr=NULL ;
	if (use_svm)
	{
		ASSERT(dictionary_weights!=NULL) ;
		wordstr=new WORD[genestr_len] ;
		for (INT i=0; i<genestr_len; i++)
			switch (genestr[i])
			{
			case 'a': wordstr[i]=0 ; break ;
			case 'c': wordstr[i]=1 ; break ;
			case 'g': wordstr[i]=2 ; break ;
			case 't': wordstr[i]=3 ; break ;
			default: ASSERT(0) ;
			}
		translate_from_single_order(wordstr, genestr_len, word_degree_single-1, word_degree_single) ;
	}
	
	
	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			delta.element(0,i,0) = get_p(i) + seq.element(i,0) ;
			psi.element(0,i,0)   = 0 ;
			ktable.element(0,i,0)  = 0 ;
			ptable.element(0,i,0)  = 0 ;
			for (short int k=1; k<nbest; k++)
			{
				delta.element(0,i,k)    = -CMath::INFTY ;
				psi.element(0,i,0)      = 0 ;
				ktable.element(0,i,k)     = 0 ;
				ptable.element(0,i,k)     = 0 ;
			}
		}
	}

	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		if (is_big && t%(seq_len/1000)==1)
			CIO::progress(t, 0, seq_len);
		
		for (T_STATES j=0; j<N; j++)
		{
			if (seq.element(j,t)<-1e20)
			{ // if we cannot observe the symbol here, then we can omit the rest
				for (short int k=0; k<nbest; k++)
				{
					delta.element(t%max_look_back,j,k)    = seq.element(j,t) ;
					psi.element(t,j,k)      = 0 ;
					ktable.element(t,j,k)     = 0 ;
					ptable.element(t,j,k)     = 0 ;
				}
			}
			else
			{
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;
				const DREAL *elem_val      = trans_list_forward_val[j] ;
				
				INT list_len=0 ;
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					//fprintf(stderr, "i=%i  ii=%i  num_elem=%i  PEN=%ld\n", i, ii, num_elem, PEN(j,ii)) ;
					
					const CPlif * penalty = PEN.element(j,ii) ;
					INT look_back = default_look_back ;
					if (penalty!=NULL)
						look_back=penalty->get_max_len() ;
					
					INT last_svm_pos ;
					if (use_svm)
						reset_svm_value(pos[t], last_svm_pos, svm_value) ;

					INT last_segment_sum_pos ;
					reset_segment_sum_value(N, pos[t], last_segment_sum_pos, segment_sum_value) ;

					for (INT ts=t-1; ts>=0 && pos[t]-pos[ts]<=look_back; ts--)
					{
						if (use_svm)
							extend_svm_value(wordstr, pos[ts], last_svm_pos, svm_value) ;

						extend_segment_sum_value(segment_sum_weights, seq_len, N, pos[ts], last_segment_sum_pos, segment_sum_value) ;
						
						DREAL input_value ;
						DREAL pen_val = 0.0 ;
						if (penalty)
							pen_val=penalty->lookup_penalty(pos[t]-pos[ts], svm_value, true, input_value) + segment_sum_value[j] ;
						for (short int diff=0; diff<nbest; diff++)
						{
							DREAL  val        = delta.element(ts%max_look_back,ii,diff) + elem_val[i] ;
							val             += pen_val ;
							
							tempvv[list_len] = -val ;
							tempii[list_len] =  ii + diff*N + ts*N*nbest;
							//fprintf(stderr, "%i (%i,%i,%i, %i, %i) ", list_len, diff, ts, i, pos[t]-pos[ts], look_back) ;
							list_len++ ;
						}
					}
				}
				CMath::nmin<INT>(tempvv.get_array(), tempii.get_array(), list_len, nbest) ;
				
				for (short int k=0; k<nbest; k++)
				{
					if (k<list_len)
					{
						delta.element(t%max_look_back,j,k)    = -tempvv[k] + seq.element(j,t);
						psi.element(t,j,k)      = (tempii[k]%N) ;
						ktable.element(t,j,k)     = (tempii[k]%(N*nbest)-psi.element(t,j,k))/N ;
						ptable.element(t,j,k)     = (tempii[k]-(tempii[k]%(N*nbest)))/(N*nbest) ;
					}
					else
					{
						delta.element(t%max_look_back,j,k)    = -CMath::INFTY ;
						psi.element(t,j,k)      = 0 ;
						ktable.element(t,j,k)     = 0 ;
						ptable.element(t,j,k)     = 0 ;
					}
				}
			}
		}
	}
	
	{ //termination
		INT list_len = 0 ;
		for (short int diff=0; diff<nbest; diff++)
		{
			for (T_STATES i=0; i<N; i++)
			{
				tempvv[list_len] = -(delta.element((seq_len-1)%max_look_back,i,diff)+get_q(i)) ;
				tempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(tempvv.get_array(), tempii.get_array(), list_len, nbest) ;
		
		for (short int k=0; k<nbest; k++)
		{
			delta_end.element(k) = -tempvv[k] ;
			path_ends.element(k) = (tempii[k]%N) ;
			ktable_end.element(k) = (tempii[k]-path_ends.element(k))/N ;
		}
	}
	
	{ //state sequence backtracking		
		for (short int k=0; k<nbest; k++)
		{
			prob_nbest[k]= delta_end.element(k) ;
			
			INT i         = 0 ;
			state_seq[i]  = path_ends.element(k) ;
			short int q   = ktable_end.element(k) ;
			pos_seq[i]    = seq_len-1 ;

			while (pos_seq[i]>0)
			{
				//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				state_seq[i+1] = psi.element(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = ptable.element(pos_seq[i], state_seq[i], q) ;
				q              = ktable.element(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
			INT num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i+k*(seq_len+1)] = state_seq[num_states-i-1] ;
				my_pos_seq[i+k*(seq_len+1)]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states+k*(seq_len+1)]=-1 ;
			my_pos_seq[num_states+k*(seq_len+1)]=-1 ;
		}
		DREAL svm_value[num_svms] ;
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
		// one more for the emissions: the first
		num_PEN_id++ ;
		PEN_values = new DREAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_values[s]=0 ;
		PEN_input_values = new DREAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_input_values[s]=0 ;
		char * PEN_names[num_PEN_id] ;
		for (INT s=0; s<num_PEN_id; s++)
			PEN_names[s]=NULL ;

		for (short int k=0; k<nbest; k++)
		{
			for (INT i=0; i<seq_len-1; i++)
			{
				if (my_state_seq[i+1+k*(seq_len+1)]==-1)
					break ;
				INT from_state = my_state_seq[i+k*(seq_len+1)] ;
				INT to_state   = my_state_seq[i+1+k*(seq_len+1)] ;
				INT from_pos   = my_pos_seq[i+k*(seq_len+1)] ;
				INT to_pos     = my_pos_seq[i+1+k*(seq_len+1)] ;
				
				//CIO::message(M_DEBUG, "%i. from state %i pos %i[%i]  to  state %i pos %i[%i]  penalties:", k, from_state, pos[from_pos], from_pos, to_state, pos[to_pos], to_pos) ;
				INT last_svm_pos = -1 ;
				INT last_segment_sum_pos=-1 ;
								
				if (use_svm)
				{
					reset_svm_value(pos[to_pos], last_svm_pos, svm_value) ;
					extend_svm_value(wordstr, pos[from_pos], last_svm_pos, svm_value) ;
				}
				reset_segment_sum_value(N, pos[to_pos], last_segment_sum_pos, segment_sum_value) ;
				extend_segment_sum_value(segment_sum_weights, seq_len, N, pos[from_pos], last_segment_sum_pos, segment_sum_value) ;
				//fprintf(stderr, "%i -> %i  %f %f\n", pos[from_pos], pos[to_pos], segment_sum_value[0], segment_sum_value[1]) ;
				
				PEN_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] = seq.element(to_state, to_pos) + segment_sum_value[to_state] ;
				//PEN_input_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] = segment_sum_value[to_state] ;

				CPlif *penalty = PEN.element(to_state, from_state) ;
				while (penalty)
				{
					DREAL input_value=0 ;
					DREAL pen_val = penalty->lookup_penalty(pos[to_pos]-pos[from_pos], svm_value, false, input_value) ;
					PEN_values[penalty->get_id() + i*num_PEN_id + seq_len*num_PEN_id*k] += pen_val ;
					PEN_input_values[penalty->get_id() + i*num_PEN_id + seq_len*num_PEN_id*k] += input_value ;
					PEN_names[penalty->get_id()] = penalty->get_name() ;
					//CIO::message(M_DEBUG, "%s(%i;%1.2f), ", penalty->name, penalty->id, pen_val) ;
					penalty = penalty->get_next_pen() ;
				}
				//CIO::message(M_DEBUG, "\n") ;
			}
			/*for (INT s=0; s<num_PEN_id; s++)
			{
				if (PEN_names[s])
					CIO::message(M_DEBUG, "%s:\t%1.2f\n", PEN_names[s], PEN_values[s+num_PEN_id*k]) ;
				else
					ASSERT(PEN_values[s]==0.0) ;
					}*/
		}
	}
	if (is_big)
		CIO::message(M_MESSAGEONLY, "DONE.     \n") ;
}

void CDynProg::reset_svm_values(INT pos, INT * last_svm_pos, DREAL * svm_value) 
{
	for (INT j=0; j<num_degrees; j++)
	{
		for (INT i=0; i<num_words_array[j]; i++)
			word_used.element(word_used_array, j, i, num_degrees)=false ;
		for (INT s=0; s<num_svms; s++)
			svm_values_unnormalized.element(j,s) = 0 ;
		num_unique_words[j]=0 ;
		last_svm_pos[j] = pos - word_degree[j]+1 ;
		svm_pos_start[j] = pos - word_degree[j] ;
	}
	for (INT s=0; s<num_svms; s++)
		svm_value[s] = 0 ;
}

void CDynProg::extend_svm_values(WORD** wordstr, INT pos, INT *last_svm_pos, DREAL* svm_value) 
{
	bool did_something = false ;
	for (INT j=0; j<num_degrees; j++)
	{
		for (int i=last_svm_pos[j]-1; (i>=pos) && (i>=0); i--)
		{
			if (wordstr[j][i]>=num_words_array[j])
				CIO::message(M_DEBUG, "wordstr[%i]=%i\n", i, wordstr[j][i]) ;

			ASSERT(wordstr[j][i]<num_words_array[j]) ;
			if (!word_used.element(word_used_array, j, wordstr[j][i], num_degrees))
			{
				for (INT s=0; s<num_svms; s++)
					svm_values_unnormalized.element(j,s)+=dict_weights_array[wordstr[j][i]+cum_num_words_array[j]+s*cum_num_words_array[num_degrees]] ;
					//svm_values_unnormalized.element(j,s)+=dict_weights.element(wordstr[j][i]+cum_num_words_array[j],s) ;
				
				//word_used.element(j,wordstr[j][i])=true ;
				word_used.element(word_used_array, j, wordstr[j][i], num_degrees)=true ;
				num_unique_words[j]++ ;
				did_something=true ;
			} ;
		} ;
		if (num_unique_words[j]>0)
			last_svm_pos[j]=pos ;
	} ;
	
	if (did_something)
		for (INT s=0; s<num_svms; s++)
		{
			svm_value[s]=0.0 ;
			for (INT j=0; j<num_degrees; j++)
				if (num_unique_words[j]>0)
					svm_value[s]+= svm_values_unnormalized.element(j,s)/sqrt((double)num_unique_words[j]) ;  // full normalization
		}
}


void CDynProg::init_segment_loss(struct segment_loss_struct & loss, INT start_pos, INT seqlen, INT howmuchlookback)
{
	if (!loss.num_segment_id)
	{
		loss.segments_changed       = new INT[seqlen] ;
		loss.num_segment_id         = new INT[(max_a_id+1)*seqlen] ;
		loss.length_segment_id      = new INT[(max_a_id+1)*seqlen] ;
	}
	
	for (INT j=0; j<seqlen; j++)
	{
		loss.segments_changed[j]=0 ;
		for (INT i=0; i<max_a_id+1; i++)       
		{
			loss.num_segment_id[i*seqlen+j] = 0;
			loss.length_segment_id[i*seqlen+j] = 0;
		}
	}

	loss.maxlookback = howmuchlookback ;
	loss.seqlen = seqlen;
}

void CDynProg::clear_segment_loss(struct segment_loss_struct & loss) 
{
	if (loss.num_segment_id != NULL)
	{
		delete[] loss.segments_changed ;
		delete[] loss.num_segment_id ;
		delete[] loss.length_segment_id ;
		loss.segments_changed = NULL ;
		loss.num_segment_id = NULL ;
		loss.length_segment_id = NULL ;
	}
}

DREAL CDynProg::extend_segment_loss(struct segment_loss_struct & loss, const INT * pos_array, INT segment_id, INT pos, INT & last_pos, DREAL &last_value) 
{
	if (pos==last_pos)
		return last_value ;
	ASSERT(pos<last_pos) ;

	last_pos-- ;
	bool changed = false ;
	while (last_pos>=pos)
	{
		if (loss.segments_changed[last_pos])
		{
			changed=true ;
			break ;
		}
		last_pos-- ;
	}
	if (last_pos<pos)
		last_pos = pos ;
	
	if (!changed)
	{
		ASSERT(last_pos>=0) ;
		ASSERT(last_pos<loss.seqlen) ;
		DREAL length_contrib = (pos_array[last_pos]-pos_array[pos])*m_segment_loss.element(1, m_segment_ids_mask.element(0, pos), segment_id) ;
		DREAL ret = last_value + length_contrib ;
		last_pos = pos ;
		return ret ;
	}

	CArray2<INT> num_segment_id(loss.num_segment_id, loss.seqlen, max_a_id+1, false, false) ;
	CArray2<INT> length_segment_id(loss.length_segment_id, loss.seqlen, max_a_id+1, false, false) ;
	DREAL ret = 0.0 ;
	for (INT i=0; i<max_a_id+1; i++)
	{
		if (num_segment_id.element(pos, i)!=0)
			ret += num_segment_id.element(loss.num_segment_id, pos, i, loss.seqlen)*m_segment_loss.element(0, i, segment_id) ;
		if (length_segment_id.element(pos, i)!=0)
			ret += length_segment_id.element(loss.num_segment_id, pos, i, loss.seqlen)*m_segment_loss.element(1, i, segment_id) ;
	}
	last_pos = pos ;
	last_value = ret ;
	return ret ;
}

void CDynProg::find_segment_loss_till_pos(const INT * pos, INT t_end, CArray2<INT>& segment_ids_mask, struct segment_loss_struct & loss) 
{
	CArray2<INT> num_segment_id(loss.num_segment_id, loss.seqlen, max_a_id+1, false, false) ;
	CArray2<INT> length_segment_id(loss.length_segment_id, loss.seqlen, max_a_id+1, false, false) ;

	for (INT i=0; i<max_a_id+1; i++)
	{
		num_segment_id.element(t_end, i) = 0 ;
		length_segment_id.element(t_end, i) = 0 ;
	}

	
	INT wobble_pos_segment_id_switch = 0 ;
	INT last_segment_id = -1 ;
	INT ts = t_end-1 ;       
	while ((ts>=0) && (pos[t_end] - pos[ts] <= loss.maxlookback))
	{
		//fprintf(stderr, "%i x %i (%i)\n", segment_ids_mask.get_dim1(), segment_ids_mask.get_dim2(), ts) ;

		INT cur_segment_id = segment_ids_mask.element(0, ts) ;
		bool wobble_pos = (segment_ids_mask.element(1, ts)==0) ;
		//fprintf(stderr, "ts:%i  ", ts) ;
		//fprintf(stderr, "max_a_id:%i  ", max_a_id) ;
		//fprintf(stderr, "s_id:%i  ", cur_segment_id) ;
		ASSERT(cur_segment_id<=max_a_id) ;
		ASSERT(cur_segment_id>=0) ;
		//fprintf(stderr, "wp=%i\n", wobble_pos) ;
		
		for (INT i=0; i<max_a_id+1; i++)
		{
			num_segment_id.element(ts, i) = num_segment_id.element(ts+1, i) ;
			length_segment_id.element(ts, i) = length_segment_id.element(ts+1, i) ;
		}

		if (cur_segment_id!=last_segment_id)
		{
			if (wobble_pos)
			{
				wobble_pos_segment_id_switch++ ;
				//ASSERT(wobble_pos_segment_id_switch<=1) ;
			}
			else
			{
				loss.segments_changed[ts]=true ;
				num_segment_id.element(ts, cur_segment_id)++ ;
				length_segment_id.element(ts, cur_segment_id)+= pos[ts+1]-pos[ts] ;
				wobble_pos_segment_id_switch = 0 ;
			}
			last_segment_id = cur_segment_id ;
		} 
		else
			if (!wobble_pos)
				length_segment_id.element(ts, cur_segment_id) += pos[ts+1] - pos[ts] ;

		ts--;
	}
}

void CDynProg::init_svm_values(struct svm_values_struct & svs, INT start_pos, INT seqlen, INT howmuchlookback)
{
	/*
	  See find_svm_values_till_pos for comments
	  
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  where t_end is the end of all segments we are currently looking at
	*/
	
	if (!svs.svm_values)
	{
		svs.num_unique_words        = new INT[num_degrees] ;
		svs.svm_values              = new DREAL[seqlen*num_svms] ;
		svs.svm_values_unnormalized = new DREAL*[num_degrees] ;
		svs.word_used               = new bool*[num_degrees] ;
		for (INT j=0; j<num_degrees; j++)
		{
			//svs.svm_values[j]              = new DREAL[seqlen*num_svms] ;
			svs.svm_values_unnormalized[j] = new DREAL[num_svms] ;
			svs.word_used[j]               = new bool[num_words_array[j]] ;
		}
	}
	
	for (INT i=0; i<seqlen*num_svms; i++)       // initializing this for safety, though we should be able to live without it
		svs.svm_values[i] = 0;

	for (INT j=0; j<num_degrees; j++)
	{		
		for (INT s=0; s<num_svms; s++)
			svs.svm_values_unnormalized[j][s] = 0 ;
		
		for (INT i=0; i<num_words_array[j]; i++)
			svs.word_used[j][i] = false ;

		svs.num_unique_words[j] = 0 ;
	}
	
	svs.maxlookback = howmuchlookback ;
	svs.seqlen = seqlen;
}

void CDynProg::clear_svm_values(struct svm_values_struct & svs) 
{
	if (NULL != svs.svm_values)
	{
		for (INT j=0; j<num_degrees; j++)
			delete[] svs.word_used[j] ;
		for (INT j=0; j<num_degrees; j++)
			delete[] svs.svm_values_unnormalized[j] ;
		
		delete[] svs.svm_values_unnormalized;
		delete[] svs.svm_values;
		delete[] svs.word_used;
		svs.word_used=NULL ;
		svs.svm_values=NULL ;
		svs.svm_values_unnormalized=NULL ;
	}
}


void CDynProg::find_svm_values_till_pos(WORD** wordstr,  const INT *pos,  INT t_end, struct svm_values_struct &svs)
{
	/*
	  wordstr is a vector of L n-gram indices, with wordstr(i) representing a number betweeen 0 and 4095 
	  corresponding to the 6-mer in genestr(i-5:i) 
	  pos is a vector of candidate transition positions (it is input to best_path_trans)
	  t_end is some index in pos
	  
	  svs has been initialized by init_svm_values
	  
	  At the end of this procedure, 
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  The SVM weights are precomputed in dict_weights
	*/
	for (INT j=0; j<num_degrees; j++)
	{
		INT plen = 1;
		INT ts = t_end-1;        // index in pos; pos(ts) and pos(t) are indices of wordstr
		INT offset;
		
		/*
		  for (INT s=0; s<num_svms; s++)
		  {
		  offset = s*svs.seqlen;
		  for (INT i=0;i<word_degree; i++)
		  svs.svm_values[i+offset] = 0;
		  }
		*/
		
		INT posprev = pos[t_end]-word_degree[j]+1;
		INT poscurrent = pos[ts];
		
		if (poscurrent<0)
			poscurrent = 0;
		
		INT len = pos[t_end] - poscurrent;
		
		while ((ts>=0) && (len <= svs.maxlookback))
		{
			for (int i=posprev-1 ; (i>=poscurrent) && (i>=0) ; i--)
			{
				// 	  if (word_degree > (pos[t_end]-pos[ts]))
				//	    fprintf(fid, " *******  i=%d , wordstr[i]=%d   dict_weights[1,wordstr[i]]=%f  t_end=%d, ts=%d  pos[t_end]=%d  pos[ts]=%d   posprev=%d\n", i, wordstr[i], dict_weights[wordstr[i]], t_end,ts,pos[t_end],pos[ts],posprev);
				
				if (wordstr[j][i]>=num_words_array[j])
					fprintf(stderr, "wordstr[%i][%i]=%i\n", j, i, wordstr[j][i]) ;
				ASSERT(wordstr[j][i]<num_words_array[j]) ;
				if (!svs.word_used[j][wordstr[j][i]])
				{
					for (INT s=0; s<num_svms; s++)
						svs.svm_values_unnormalized[j][s]+=dict_weights_array[wordstr[j][i]+cum_num_words_array[j]+s*cum_num_words_array[num_degrees]] ;
					//svs.svm_values_unnormalized[j][s]+=dict_weights.element(wordstr[j][i]+cum_num_words_array[j], s) ;
					
					svs.word_used[j][wordstr[j][i]]=true ;
					svs.num_unique_words[j]++ ;
				}
			}
			double normalization_factor = 1.0;
			if (svs.num_unique_words[j] > 0)
				normalization_factor = sqrt((double)svs.num_unique_words[j]);
			for (INT s=0; s<num_svms; s++)
			{
				offset = s*svs.seqlen;
				if (j==0)
					svs.svm_values[offset+plen]=0 ;
				svs.svm_values[offset+plen] += svs.svm_values_unnormalized[j][s] / normalization_factor;
			}
			
			if (posprev > poscurrent)         // remember posprev initially set to pos[t_end]-word_degree+1... pos[ts] could be e.g. pos[t_end]-2
				posprev = poscurrent;           
			
			ts--;
			plen++;
			
			if (ts>=0)
			{
				poscurrent=pos[ts];
				if (poscurrent<0)
					poscurrent = 0;
				len = pos[t_end] - poscurrent;
			}
		}
	}
}


bool CDynProg::extend_orf(const CArray<bool>& genestr_stop, INT orf_from, INT orf_to, INT start, INT &last_pos, INT to)
{
	if (start<0) 
		start=0 ;
	if (to<0)
		to=0 ;
	
	INT orf_target = orf_to-orf_from ;
	if (orf_target<0) orf_target+=3 ;
	
	INT pos ;
	if (last_pos==to)
		pos = to-orf_to-3 ;
	else
		pos=last_pos ;

	if (pos<0)
		return true ;
	
	for (; pos>=start; pos-=3)
		if (genestr_stop[pos])
			return false ;
	
	last_pos = CMath::min(pos+3,to-orf_to-3) ;

	return true ;
}


void CDynProg::best_path_trans(const DREAL *seq_array, INT seq_len, const INT *pos, const INT *orf_info_array,
							   CPlif **Plif_matrix, CPlif **Plif_state_signals,
							   const char *genestr, INT genestr_len,
							   short int nbest, 
							   DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
							   DREAL *dictionary_weights, INT dict_len, 
							   DREAL *&PEN_values, DREAL *&PEN_input_values, 
							   INT &num_PEN_id, bool use_orf)
{
	const INT default_look_back = 30000 ;
	INT max_look_back = default_look_back ;
	bool use_svm = false ;
	ASSERT(dict_len==num_svms*cum_num_words_array[num_degrees]) ;
	dict_weights.set_array(dictionary_weights, cum_num_words_array[num_degrees], num_svms, false, false) ;
	dict_weights_array=dict_weights.get_array() ;
	int offset=0;
	
	CArray2<CPlif*> PEN(Plif_matrix, N, N, false, false) ;
	CArray2<CPlif*> PEN_state_signals(Plif_state_signals, N, 2, false, false) ;
	CArray2<DREAL> seq_input(seq_array, N, seq_len) ;
	CArray2<DREAL> seq(N, seq_len) ;
	//seq.zero() ;
	CArray2<INT> orf_info(orf_info_array, N, 2) ;
	
	DREAL svm_value[num_svms] ;
	{ // initialize svm_svalue
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}

	{ // convert seq_input to seq
      // this is independent of the svm values 
		for (INT i=0; i<N; i++)
			for (INT j=0; j<seq_len; j++)
				seq.element(i,j) = 0 ;

		for (INT i=0; i<N; i++)
			for (INT j=0; j<seq_len; j++)
			{
				DREAL tmp ;
				if (PEN_state_signals.element(i,0)==NULL)
					// no plif
					seq.element(i,j) = seq_input.element(i,j) ;
				else
				{
					if (PEN_state_signals.element(i,1)==NULL)
					{
						// just one plif
						if (finite(seq_input.element(i,j)))
							seq.element(i,j) = PEN_state_signals.element(i,0)->lookup_penalty(seq_input.element(i,j), svm_value, true, tmp) ;
						else
							seq.element(i,j) = seq_input.element(i,j) ;
					}
					else
					{
						// decode the two parts and use them with the two plifs
						if (finite(seq_input.element(i,j)))
						{
							INT part1 = (INT) seq_input.element(i,j) ;
							INT part2 = (INT) ((seq_input.element(i,j)-(DREAL)part1)*100.) ;
							DREAL input1 = part1/1000000. - 50. ;
							DREAL input2 = part2 - 50. ;
							if (!((input1>-50) && (input1<50)) || !((input2>-50) && (input2<50)))
								fprintf(stderr, "(%i, %i), (%f, %f), %e", part1, part2, input1, input2, seq_input.element(i,j)) ;
							
							ASSERT((input1>-50) && (input1<50)) ;
							ASSERT((input2>-50) && (input2<50)) ;
							
							seq.element(i,j) = PEN_state_signals.element(i,0)->lookup_penalty(input1, svm_value, true, tmp) ;
							seq.element(i,j) += PEN_state_signals.element(i,1)->lookup_penalty(input2, svm_value, true, tmp) ;
						}
						else
							seq.element(i,j) = seq_input.element(i,j) ;
				    }
				}
			}
	}
	
	{ // determine maximal length of look-back
		for (INT i=0; i<N; i++)
			for (INT j=0; j<N; j++)
			{
				CPlif *penij=PEN.element(i,j) ;
				while (penij!=NULL)
				{
					if (penij->get_max_len()>max_look_back)
						max_look_back=penij->get_max_len() ;
					if (penij->get_use_svm())
						use_svm=true ;
					if (penij->get_id()+1>num_PEN_id)
						num_PEN_id=penij->get_id()+1 ;
					penij=penij->get_next_pen() ;
				} 
			}
	}
	max_look_back = CMath::min(genestr_len, max_look_back) ;
	//fprintf(stderr,"use_svm=%i\n", use_svm) ;
	
	const INT look_back_buflen = max_look_back*nbest*N ;
	const DREAL mem_use = (DREAL)(seq_len*N*nbest*(sizeof(T_STATES)+sizeof(short int)+sizeof(INT))+
								  look_back_buflen*(2*sizeof(DREAL)+sizeof(INT))+
								  seq_len*(sizeof(T_STATES)+sizeof(INT))+
								  genestr_len*sizeof(bool))/(1024*1024)
		 ;
    bool is_big = (mem_use>200) || (seq_len>5000) ;

	if (is_big)
	{
		CIO::message(M_DEBUG,"calling best_path_trans: seq_len=%i, N=%i, lookback=%i nbest=%i\n", 
					 seq_len, N, max_look_back, nbest) ;
		CIO::message(M_DEBUG,"allocating %1.2fMB of memory\n", 
					 mem_use) ;
	}
	ASSERT(nbest<32000) ;
		
	CArray<bool> genestr_stop(genestr_len) ;
	//genestr_stop.zero() ;
	
	CArray3<DREAL> delta(max_look_back, N, nbest) ;
	DREAL* delta_array = delta.get_array() ;
	//delta.zero() ;
	
	CArray3<T_STATES> psi(seq_len, N, nbest) ;
	//psi.zero() ;
	
	CArray3<short int> ktable(seq_len, N, nbest) ;
	//ktable.zero() ;
	
	CArray3<INT> ptable(seq_len, N, nbest) ;
	//ptable.zero() ;

	CArray<DREAL> delta_end(nbest) ;
	//delta_end.zero() ;
	
	CArray<T_STATES> path_ends(nbest) ;
	//path_ends.zero() ;
	
	CArray<short int> ktable_end(nbest) ;
	//ktable_end.zero() ;

#if USEFIXEDLENLIST > 0
#ifdef USE_TMP_ARRAYCLASS
	CArray<DREAL> fixedtempvv(look_back_buflen) ;
	CArray<INT> fixedtempii(look_back_buflen) ;
	//fixedtempvv.zero() ;
	//fixedtempii.zero() ;
#else
	DREAL * fixedtempvv=new DREAL[look_back_buflen] ;
	memset(fixedtempvv, 0, look_back_buflen*sizeof(DREAL)) ;
	INT * fixedtempii=new INT[look_back_buflen] ;
	memset(fixedtempii, 0, look_back_buflen*sizeof(INT)) ;
#endif
#endif

	// we always use oldtempvv and oldtempii, even if USEORIGINALLIST is 0
	// as i didnt change the backtracking stuff
	
	CArray<DREAL> oldtempvv(look_back_buflen) ;
	//oldtempvv.zero() ;
	
	CArray<INT> oldtempii(look_back_buflen) ;
	//oldtempii.zero() ;

	CArray<T_STATES> state_seq(seq_len) ;
	//state_seq.zero() ;
	
	CArray<INT> pos_seq(seq_len) ;
	//pos_seq.zero() ;

	
#ifdef ARRAY_STATISTICS
	dict_weights.set_name("dict_weights") ;
	word_degree.set_name("word_degree") ;
	cum_num_words.set_name("cum_num_words") ;
	num_words.set_name("num_words") ;
	word_used.set_name("word_used") ;
	svm_values_unnormalized.set_name("svm_values_unnormalized") ;
	svm_pos_start.set_name("svm_pos_start") ;
	num_unique_words.set_name("num_unique_words") ;

	PEN.set_name("PEN") ;
	seq.set_name("seq") ;
	orf_info.set_name("orf_info") ;
	
	genestr_stop.set_name("genestr_stop") ;
	delta.set_name("delta") ;
	psi.set_name("psi") ;
	ktable.set_name("ktable") ;
	ptable.set_name("ptable") ;
	delta_end.set_name("delta_end") ;
	path_ends.set_name("path_ends") ;
	ktable_end.set_name("ktable_end") ;

#ifdef USE_TMP_ARRAYCLASS
	fixedtempvv.set_name("fixedtempvv") ;
	fixedtempii.set_name("fixedtempvv") ;
#endif

	oldtempvv.set_name("oldtempvv") ;
	oldtempii.set_name("oldtempii") ;

	state_seq.set_name("state_seq") ;
	pos_seq.set_name("pos_seq") ;
#endif

	{ // precompute stop codons
		for (INT i=0; i<genestr_len-2; i++)
			if (genestr[i]=='t' && 
				((genestr[i+1]=='a' && 
				  (genestr[i+2]=='a' || genestr[i+2]=='g')) ||
				 (genestr[i+1]=='g' && genestr[i+2]=='a')))
				genestr_stop[i]=true ;
			else
				genestr_stop[i]=false ;
		genestr_stop[genestr_len-1]=false ;
		genestr_stop[genestr_len-1]=false ;
	}

	// translate to words, if svm is used
	WORD* wordstr[num_degrees] ;
	{
		for (INT j=0; j<num_degrees; j++)
		{
			wordstr[j]=NULL ;
			if (use_svm)
			{
				ASSERT(dictionary_weights!=NULL) ;
				wordstr[j]=new WORD[genestr_len] ;
				for (INT i=0; i<genestr_len; i++)
					switch (genestr[i])
					{
					case 'a': wordstr[j][i]=0 ; break ;
					case 'c': wordstr[j][i]=1 ; break ;
					case 'g': wordstr[j][i]=2 ; break ;
					case 't': wordstr[j][i]=3 ; break ;
					default: ASSERT(0) ;
					}
				translate_from_single_order(wordstr[j], genestr_len,
											word_degree[j]-1, word_degree[j]) ;
			}
		}
	}
	
  	
	{ // initialization

		for (T_STATES i=0; i<N; i++)
		{
			//delta.element(0, i, 0) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
			delta.element(delta_array, 0, i, 0, max_look_back, N) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
			psi.element(0,i,0)   = 0 ;
			ktable.element(0,i,0)  = 0 ;
			ptable.element(0,i,0)  = 0 ;
			for (short int k=1; k<nbest; k++)
			{
				INT dim1, dim2, dim3 ;
				delta.get_array_size(dim1, dim2, dim3) ;
				//fprintf(stderr, "i=%i, k=%i -- %i, %i, %i\n", i, k, dim1, dim2, dim3) ;
				//delta.element(0, i, k)    = -CMath::INFTY ;
				delta.element(delta_array, 0, i, k, max_look_back, N)    = -CMath::INFTY ;
				psi.element(0,i,0)      = 0 ;                  // <--- what's this for?
				ktable.element(0,i,k)     = 0 ;
				ptable.element(0,i,k)     = 0 ;
			}
		}
	}

	struct svm_values_struct svs;
	svs.num_unique_words = NULL;
	svs.svm_values = NULL;
	svs.svm_values_unnormalized = NULL;
	svs.word_used = NULL;

	struct segment_loss_struct loss;
	loss.segments_changed = NULL;
	loss.num_segment_id = NULL;

	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		if (is_big && t%(seq_len/1000)==1)
			CIO::progress(t, 0, seq_len);
		
		init_svm_values(svs, t, seq_len, max_look_back);
		find_svm_values_till_pos(wordstr, pos, t, svs);  

		init_segment_loss(loss, t, seq_len, max_look_back);
		find_segment_loss_till_pos(pos, t, m_segment_ids_mask, loss);  
	
		for (T_STATES j=0; j<N; j++)
		{
			if (seq.element(j,t)<=-1e20)
			{ // if we cannot observe the symbol here, then we can omit the rest
				for (short int k=0; k<nbest; k++)
				{
					delta.element(delta_array, t%max_look_back, j, k, max_look_back, N)    = seq.element(j,t) ;
					psi.element(t,j,k)      = 0 ;
					ktable.element(t,j,k)     = 0 ;
					ptable.element(t,j,k)     = 0 ;
				}
			}
			else
			{
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;
				const DREAL *elem_val      = trans_list_forward_val[j] ;
				const INT *elem_id      = trans_list_forward_id[j] ;
				
#if USEFIXEDLENLIST > 0
				INT fixed_list_len = 0 ;
#endif
				
#if USEORIGINALLIST > 0
				INT old_list_len = 0 ;
#endif
				
#if USEHEAP > 0
				Heap* tempheap = new Heap;
#endif
				
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					
					const CPlif * penalty = PEN.element(j,ii) ;
					INT look_back = default_look_back ;
					{ // find lookback length
						CPlif *pen = (CPlif*) penalty ;
						if (pen!=NULL)
						{
							look_back=pen->get_max_len() ;
							while (pen->get_next_pen()!=NULL)
							{
								pen=pen->get_next_pen() ;
								if (pen->get_max_len()<look_back)
									look_back=pen->get_max_len() ;
							}
						}
						ASSERT(look_back<1e6);
					}
					INT orf_from = orf_info.element(ii,0) ;
					INT orf_to   = orf_info.element(j,1) ;
					if((orf_from!=-1)!=(orf_to!=-1))
						fprintf(stderr,"j=%i  ii=%i  orf_from=%i orf_to=%i p=%1.2f\n", j, ii, orf_from, orf_to, elem_val[i]) ;
					ASSERT((orf_from!=-1)==(orf_to!=-1)) ;
					
					INT orf_target = -1 ;
					if (orf_from!=-1)
					{
						orf_target=orf_to-orf_from ;
						if (orf_target<0) orf_target+=3 ;
						ASSERT(orf_target>=0 && orf_target<3) ;
					}
					
					INT orf_last_pos = pos[t] ;
					INT loss_last_pos = t ;
					DREAL last_loss = 0.0 ;
					for (INT ts=t-1; ts>=0 && pos[t]-pos[ts]<=look_back; ts--)
					{
						bool ok ;
						int plen=t-ts;
						
						if (orf_target==-1)
							ok=true ;
						else if (pos[ts]!=-1 && (pos[t]-pos[ts])%3==orf_target)
						{
							ok=(!use_orf) || extend_orf(genestr_stop, orf_from, orf_to, pos[ts], orf_last_pos, pos[t]) ;
							if (!ok) 
							{
								//CIO::message(M_DEBUG, "no orf from %i[%i] to %i[%i]\n", pos[ts], orf_from, pos[t], orf_to) ;
								break ;
							}
						} else
							ok=false ;
						
						if (ok)
						{
							DREAL segment_loss = extend_segment_loss(loss, pos, elem_id[i], ts, loss_last_pos, last_loss) ;

							for (INT ss=0; ss<num_svms; ss++)
						    {
								offset = ss*svs.seqlen;
								svm_value[ss]=svs.svm_values[offset+plen];
						    }
							
							DREAL input_value ;
							DREAL pen_val = 0.0 ;
							if (penalty)
								pen_val = penalty->lookup_penalty(pos[t]-pos[ts], svm_value, true, input_value) ;

							for (short int diff=0; diff<nbest; diff++)
						    {
								DREAL  val        = delta.element(delta_array, ts%max_look_back, ii, diff, max_look_back, N) + elem_val[i] ;
								//DREAL  val        = delta.element(ts%max_look_back, ii, diff) + elem_val[i] ;
								val              += pen_val ;
								val              += segment_loss ;
								ASSERT((val>-1e20) && (val<1e20))
								DREAL mval = -val;
								
#if USEHEAP > 0
								tempheap->Insert(mval, ii + diff*N + ts*N*nbest);
#endif
								
#if USEORIGINALLIST > 0
								oldtempvv[old_list_len] = mval ;
								oldtempii[old_list_len] = ii + diff*N + ts*N*nbest;
								old_list_len++ ;
#endif
								
#if USEFIXEDLENLIST > 0
								
								/* only place -val in fixedtempvv if it is one of the nbest lowest values in there */
								/* fixedtempvv[i], i=0:nbest-1, is sorted so that fixedtempvv[0] <= fixedtempvv[1] <= ...*/
								/* fixed_list_len has the number of elements in fixedtempvv */
								
								if ((fixed_list_len < nbest) || ((0==fixed_list_len) || (mval < fixedtempvv[fixed_list_len-1])))
								{
									//fprintf(stderr, "nbest=%i\n", nbest) ;
									//fprintf(stderr, "fixed_list_len=%i\n", fixed_list_len) ;
									//fprintf(stderr, "mval=%f\n", mval) ;
									//fprintf(stderr, "max_look_back=%i\n", max_look_back) ;
									//if (fixed_list_len!=0)
									//	fprintf(stderr, "fixedtempvv[fixed_list_len-1]=%f\n", fixedtempvv[fixed_list_len-1]) ;

									if ( (fixed_list_len<nbest) && ((0==fixed_list_len) || (mval>fixedtempvv[fixed_list_len-1])) )
									{
										fixedtempvv[fixed_list_len] = mval ;
										fixedtempii[fixed_list_len] = ii + diff*N + ts*N*nbest;
										fixed_list_len++ ;
									}
									else  // must have mval < fixedtempvv[fixed_list_len-1]
									{
										int addhere = fixed_list_len;
										while ((addhere > 0) && (mval < fixedtempvv[addhere-1]))
											addhere--;
										
										// move everything from addhere+1 one forward 
										
										for (int jj=fixed_list_len-1; jj>addhere; jj--)
										{
											fixedtempvv[jj] = fixedtempvv[jj-1];
											fixedtempii[jj] = fixedtempii[jj-1];
										}
										
										fixedtempvv[addhere] = mval;
										fixedtempii[addhere] = ii + diff*N + ts*N*nbest;
										
										if (fixed_list_len < nbest)
											fixed_list_len++;
									}
								}
#endif
								
						    }
						}
					}
				}
				
#if USEORIGINALLIST > 0
				CMath::nmin<INT>(oldtempvv, oldtempii, old_list_len, nbest) ;
#endif
				
				int numEnt = 0;
#if USEHEAP == 2
				numEnt = tempheap->GetNumNodes();
#elif USEORIGINALLIST == 2
				numEnt = old_list_len;
#elif USEFIXEDLENLIST == 2
				numEnt = fixed_list_len;
#endif
				
				double minusscore;
				long int fromtjk;
				
				for (short int k=0; k<nbest; k++)
				{
					if (k<numEnt)
					{
#if (USEHEAP == 2)
					    tempheap->ExtractMin(minusscore,fromtjk);
#elif (USEORIGINALLIST == 2)
					    minusscore = oldtempvv[k];
					    fromtjk = oldtempii[k];
#elif (USEFIXEDLENLIST == 2)
					    minusscore = fixedtempvv[k];
					    fromtjk = fixedtempii[k];
#endif
					    delta.element(delta_array, t%max_look_back, j, k, max_look_back, N)    = -minusscore + seq.element(j,t);
					    psi.element(t,j,k)      = (fromtjk%N) ;
					    ktable.element(t,j,k)     = (fromtjk%(N*nbest)-psi.element(t,j,k))/N ;
					    ptable.element(t,j,k)     = (fromtjk-(fromtjk%(N*nbest)))/(N*nbest) ;
					}
					else
					{
						delta.element(delta_array, t%max_look_back, j, k, max_look_back, N)    = -CMath::INFTY ;
						psi.element(t,j,k)      = 0 ;
						ktable.element(t,j,k)     = 0 ;
						ptable.element(t,j,k)     = 0 ;
					}
				}
				
#if USEHEAP > 0
				delete tempheap;
#endif
			}
		}
	}
	
	clear_segment_loss(loss);
	clear_svm_values(svs);

	{ //termination
		INT list_len = 0 ;
		for (short int diff=0; diff<nbest; diff++)
		{
			for (T_STATES i=0; i<N; i++)
			{
				oldtempvv[list_len] = -(delta.element(delta_array, (seq_len-1)%max_look_back, i, diff, max_look_back, N)+get_q(i)) ;
				oldtempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(oldtempvv.get_array(), oldtempii.get_array(), list_len, nbest) ;
		
		for (short int k=0; k<nbest; k++)
		{
			delta_end.element(k) = -oldtempvv[k] ;
			path_ends.element(k) = (oldtempii[k]%N) ;
			ktable_end.element(k) = (oldtempii[k]-path_ends.element(k))/N ;
		}
	}
	
	{ //state sequence backtracking		
		for (short int k=0; k<nbest; k++)
		{
			prob_nbest[k]= delta_end.element(k) ;
			
			INT i         = 0 ;
			state_seq[i]  = path_ends.element(k) ;
			short int q   = ktable_end.element(k) ;
			pos_seq[i]    = seq_len-1 ;

			while (pos_seq[i]>0)
			{
				//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				state_seq[i+1] = psi.element(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = ptable.element(pos_seq[i], state_seq[i], q) ;
				q              = ktable.element(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
			INT num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i+k*seq_len] = state_seq[num_states-i-1] ;
				my_pos_seq[i+k*seq_len]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states+k*seq_len]=-1 ;
			my_pos_seq[num_states+k*seq_len]=-1 ;
		}

		DREAL svm_value[num_svms] ;
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;

		// one more for the emissions: the first
		num_PEN_id++ ;
        // allocate memory
		PEN_values = new DREAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_values[s]=0 ;
		PEN_input_values = new DREAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_input_values[s]=0 ;
		char * PEN_names[num_PEN_id] ;
		for (INT s=0; s<num_PEN_id; s++)
			PEN_names[s]=NULL ;

		for (short int k=0; k<nbest; k++)
		{
			for (INT i=0; i<seq_len-1; i++)
			{
				if (my_state_seq[i+1+k*seq_len]==-1)
					break ;
				INT from_state = my_state_seq[i+k*seq_len] ;
				INT to_state   = my_state_seq[i+1+k*seq_len] ;
				INT from_pos   = my_pos_seq[i+k*seq_len] ;
				INT to_pos     = my_pos_seq[i+1+k*seq_len] ;
				
				//CIO::message(M_DEBUG, "%i. from state %i pos %i[%i]  to  state %i pos %i[%i]  penalties:", k, from_state, pos[from_pos], from_pos, to_state, pos[to_pos], to_pos) ;
				
				INT last_svm_pos[num_degrees] ;
				for (INT qq=0; qq<num_degrees; qq++)
					last_svm_pos[qq]=-1 ;
				
				reset_svm_values(pos[to_pos], last_svm_pos, svm_value) ;
				extend_svm_values(wordstr, pos[from_pos], last_svm_pos, svm_value) ;

				PEN_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] += seq.element(to_state, to_pos) ;
				PEN_input_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] = to_state + to_pos*1000 ;

				CPlif *penalty = PEN.element(to_state, from_state) ;
				while (penalty)
				{
					DREAL input_value=0 ;
					DREAL pen_val = penalty->lookup_penalty(pos[to_pos]-pos[from_pos], svm_value, false, input_value) ;
					PEN_values[penalty->get_id() + i*num_PEN_id + seq_len*num_PEN_id*k] += pen_val ;
					PEN_input_values[penalty->get_id() + i*num_PEN_id + seq_len*num_PEN_id*k] += input_value ;
					PEN_names[penalty->get_id()] = penalty->get_name() ;
					//CIO::message(M_DEBUG, "%s(%i;%1.2f), ", penalty->name, penalty->id, pen_val) ;
					penalty = penalty->get_next_pen() ;
				}
				//CIO::message(M_DEBUG, "\n") ;
			}
			/*for (INT s=0; s<num_PEN_id; s++)
			{
				if (PEN_names[s])
					CIO::message(M_DEBUG, "%s:\t%1.2f\n", PEN_names[s], PEN_values[s+num_PEN_id*k]) ;
				else
					ASSERT(PEN_values[s]==0.0) ;
					}*/
		}
	}
	if (is_big)
		CIO::message(M_MESSAGEONLY, "DONE.     \n") ;

	for (INT j=0; j<num_degrees; j++)
		delete[] wordstr[j] ;

#if USEFIXEDLENLIST > 0
#ifndef USE_TMP_ARRAYCLASS
	delete[] fixedtempvv ;
	delete[] fixedtempii ;
#endif
#endif
}

void CDynProg::best_path_trans_deriv(INT *my_state_seq, INT *my_pos_seq, DREAL *my_scores, DREAL* my_losses,
									 INT my_seq_len, 
									 const DREAL *seq_array, INT seq_len, const INT *pos,
									 CPlif **Plif_matrix, CPlif **Plif_state_signals,
									 const char *genestr, INT genestr_len,
									 DREAL *dictionary_weights, INT dict_len)
{	
	bool use_svm = false ;
	ASSERT(dict_len==num_svms*cum_num_words_array[num_degrees]) ;
	dict_weights.set_array(dictionary_weights, cum_num_words_array[num_degrees], num_svms, false, false) ;
	dict_weights_array=dict_weights.get_array() ;
	
	CArray2<CPlif*> PEN(Plif_matrix, N, N, false, false) ;
	CArray2<CPlif*> PEN_state_signals(Plif_state_signals, N, 2, false, false) ;
	CArray2<DREAL> seq_input(seq_array, N, seq_len) ;
	
	{ // determine whether to use svm outputs and clear derivatives
		for (INT i=0; i<N; i++)
			for (INT j=0; j<N; j++)
			{
				CPlif *penij=PEN.element(i,j) ;
				while (penij!=NULL)
				{
					if (penij->get_use_svm())
						use_svm=true ;
					penij->penalty_clear_derivative(false) ;
					penij=penij->get_next_pen() ;
				} 
			}
		for (INT i=0; i<N; i++)
			for (INT j=0; j<2; j++)
			{
				CPlif *penij=PEN_state_signals.element(i,j) ;
				while (penij!=NULL)
				{
					if (penij->get_use_svm())
						use_svm=true ;
					penij->penalty_clear_derivative(false) ;
					penij=penij->get_next_pen() ;
				} 
			}
	}

	// translate to words, if svm is used
	WORD* wordstr[num_degrees] ;
	{
		for (INT j=0; j<num_degrees; j++)
		{
			wordstr[j]=NULL ;
			if (use_svm)
			{
				ASSERT(dictionary_weights!=NULL) ;
				wordstr[j]=new WORD[genestr_len] ;
				for (INT i=0; i<genestr_len; i++)
					switch (genestr[i])
					{
					case 'a': wordstr[j][i]=0 ; break ;
					case 'c': wordstr[j][i]=1 ; break ;
					case 'g': wordstr[j][i]=2 ; break ;
					case 't': wordstr[j][i]=3 ; break ;
					default: ASSERT(0) ;
					}
				translate_from_single_order(wordstr[j], genestr_len,
											word_degree[j]-1, word_degree[j]) ;
			}
		}
	}
	
	{ // set derivatives of p, q and a to zero
		for (INT i=0; i<N; i++)
		{
			initial_state_distribution_p_deriv.element(i)=0 ;
			end_state_distribution_q_deriv.element(i)=0 ;
			for (INT j=0; j<N; j++)
				transition_matrix_a_deriv.element(i,j)=0 ;
		}
	}
	
	{ // clear score vector
		for (INT i=0; i<my_seq_len; i++)
		{
			my_scores[i]=0.0 ;
			my_losses[i]=0.0 ;
		}
	}
	
	{ // compute derivatives for given path
		DREAL svm_value[num_svms] ;
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
		
		for (INT i=0; i<my_seq_len; i++)
		{
			my_scores[i]=0.0 ;
			my_losses[i]=0.0 ;
		}
		
		ASSERT(my_state_seq[0]>=0) ;
		initial_state_distribution_p_deriv.element(my_state_seq[0])++ ;
		my_scores[0] += initial_state_distribution_p.element(my_state_seq[0]) ;

		ASSERT(my_state_seq[my_seq_len-1]>=0) ;
		end_state_distribution_q_deriv.element(my_state_seq[my_seq_len-1])++ ;
		my_scores[my_seq_len-1] += end_state_distribution_q.element(my_state_seq[my_seq_len-1]);
		
		struct segment_loss_struct loss;
		loss.segments_changed = NULL;
		loss.num_segment_id = NULL;
		
		//fprintf(stderr, "seq_len=%i\n", my_seq_len) ;
		for (INT i=0; i<my_seq_len-1; i++)
		{
			if (my_state_seq[i+1]==-1)
				break ;
			INT from_state = my_state_seq[i] ;
			INT to_state   = my_state_seq[i+1] ;
			INT from_pos   = my_pos_seq[i] ;
			INT to_pos     = my_pos_seq[i+1] ;

			// compute loss relative to another segmentation using the segment_loss function
			init_segment_loss(loss, to_pos, seq_len, pos[to_pos]-pos[from_pos]+10);
			find_segment_loss_till_pos(pos, to_pos, m_segment_ids_mask, loss);  
			INT loss_last_pos = to_pos ;
			DREAL last_loss = 0.0 ;
			INT elem_id = transition_matrix_a_id.element(from_state, to_state) ;
			my_losses[i] = extend_segment_loss(loss, pos, elem_id, from_pos, loss_last_pos, last_loss) ;

			// increase usage of this transition
			transition_matrix_a_deriv.element(from_state, to_state)++ ;
			my_scores[i] += transition_matrix_a.element(from_state, to_state) ;

			INT last_svm_pos[num_degrees] ;
			for (INT qq=0; qq<num_degrees; qq++)
				last_svm_pos[qq]=-1 ;
			
			if (use_svm)
			{
				reset_svm_values(pos[to_pos], last_svm_pos, svm_value) ;
				extend_svm_values(wordstr, pos[from_pos], last_svm_pos, svm_value) ;
			}
			
			if (PEN.element(to_state, from_state)!=NULL)
			{
				DREAL tmp=0.0 ;
				DREAL nscore = PEN.element(to_state, from_state)->lookup_penalty(pos[to_pos]-pos[from_pos], svm_value, true, tmp) ;
				my_scores[i] += nscore ;
				CIO::message(M_DEBUG, "%i. transition penalty: from_state=%i to_state=%i from_pos=%i to_pos=%i value=%i\n", i, from_state, to_state, from_pos, to_pos, pos[to_pos]-pos[from_pos]) ;
				PEN.element(to_state, from_state)->penalty_add_derivative(pos[to_pos]-pos[from_pos], svm_value, true) ;
			}

			//fprintf(stderr, "emmission penalty skipped: to_state=%i to_pos=%i value=%1.2f score=%1.2f\n", to_state, to_pos, seq_input.element(to_state, to_pos), 0.0) ;
			if (PEN_state_signals.element(to_state,0)!=NULL)
			{
				if (PEN_state_signals.element(to_state,1)!=NULL)
				{
					// decode the two parts and use them with the two plifs
					INT part1 = (INT) seq_input.element(to_state,to_pos) ;
					INT part2 = (INT) ((seq_input.element(to_state,to_pos)-(DREAL)part1)*100.) ;
					DREAL input1 = part1/1000000. - 50. ;
					DREAL input2 = part2 - 50. ;
					ASSERT((input1>-50) && (input1<50)) ;
					ASSERT((input2>-50) && (input2<50)) ;

					DREAL tmp=0.0 ;
					DREAL nscore1 = PEN_state_signals.element(to_state,0)->lookup_penalty(input1, svm_value, true, tmp) ;
					DREAL nscore2 = PEN_state_signals.element(to_state,1)->lookup_penalty(input2, svm_value, true, tmp) ;
					my_scores[i] += nscore1 + nscore2 ;
					CIO::message(M_DEBUG, "%i. emmission penalty: to_state=%i to_pos=%i value1=%1.2f value2=%1.2f score1=%1.2f score2=%1.2f\n", i, to_state, to_pos, input1, input2, nscore1, nscore2) ;

					PEN_state_signals.element(to_state,0)->penalty_add_derivative(input1, svm_value, true) ;
					PEN_state_signals.element(to_state,1)->penalty_add_derivative(input2, svm_value, true) ;
				}
				else
				{
					DREAL tmp=0.0 ;
					DREAL nscore = PEN_state_signals.element(to_state,0)->lookup_penalty(seq_input.element(to_state, to_pos), svm_value, true, tmp) ;
					my_scores[i] += nscore ;
					CIO::message(M_DEBUG, "%i. emmission penalty: to_state=%i to_pos=%i value=%1.2f score=%1.2f\n", i, to_state, to_pos, seq_input.element(to_state, to_pos), nscore) ;

					PEN_state_signals.element(to_state,0)->penalty_add_derivative(seq_input.element(to_state, to_pos), svm_value, true) ;
				}
			} else
			{
				CIO::message(M_DEBUG, "%i. emmission penalty: to_state=%i to_pos=%i score=%1.2f\n", i, to_state, to_pos, seq_input.element(to_state, to_pos)) ;
				my_scores[i] += seq_input.element(to_state, to_pos) ;
			}
		}
		clear_segment_loss(loss);
	}

	for (INT j=0; j<num_degrees; j++)
		delete[] wordstr[j] ;
}


void CDynProg::best_path_trans_simple(const DREAL *seq_array, INT seq_len, short int nbest, 
									  DREAL *prob_nbest, INT *my_state_seq)
{
	INT max_look_back = 2 ;
	const INT look_back_buflen = max_look_back*nbest*N ;
	ASSERT(nbest<32000) ;
		
	CArray2<DREAL> seq((DREAL *)seq_array, N, seq_len, false) ;

	CArray3<DREAL> delta(max_look_back, N, nbest) ;
	CArray3<T_STATES> psi(seq_len, N, nbest) ;
	CArray3<short int> ktable(seq_len,N,nbest) ;
	CArray3<INT> ptable(seq_len,N,nbest) ;

	CArray<DREAL> delta_end(nbest) ;
	CArray<T_STATES> path_ends(nbest) ;
	CArray<short int> ktable_end(nbest) ;

	CArray<DREAL> oldtempvv(look_back_buflen) ;
	CArray<INT> oldtempii(look_back_buflen) ;

	CArray<T_STATES> state_seq(seq_len) ;
	CArray<INT> pos_seq(seq_len) ;

	{ // initialization

		for (T_STATES i=0; i<N; i++)
		{
			delta.element(0,i,0) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
			psi.element(0,i,0)   = 0 ;
			ktable.element(0,i,0)  = 0 ;
			ptable.element(0,i,0)  = 0 ;
			for (short int k=1; k<nbest; k++)
			{
				delta.element(0,i,k)    = -CMath::INFTY ;
				psi.element(0,i,0)      = 0 ;                  // <--- what's this for?
				ktable.element(0,i,k)     = 0 ;
				ptable.element(0,i,k)     = 0 ;
			}
		}
	}

	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		for (T_STATES j=0; j<N; j++)
		{
			if (seq.element(j,t)<-1e20)
			{ // if we cannot observe the symbol here, then we can omit the rest
				for (short int k=0; k<nbest; k++)
				{
					delta.element(t%max_look_back,j,k)    = seq.element(j,t) ;
					psi.element(t,j,k)      = 0 ;
					ktable.element(t,j,k)     = 0 ;
					ptable.element(t,j,k)     = 0 ;
				}
			}
			else
			{
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;
				const DREAL *elem_val      = trans_list_forward_val[j] ;
				
				INT old_list_len = 0 ;
				
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;

					INT ts=t-1; 
					if (ts>=0)
					{
						bool ok=true ;
						
						if (ok)
						{

						  
						  for (short int diff=0; diff<nbest; diff++)
						    {
						      DREAL  val        = delta.element(ts%max_look_back,ii,diff) + elem_val[i] ;
						      DREAL mval = -val;

						      oldtempvv[old_list_len] = mval ;
						      oldtempii[old_list_len] = ii + diff*N + ts*N*nbest;
						      old_list_len++ ;
						    }
						}
					}
				}
				
				CMath::nmin<INT>(oldtempvv.get_array(), oldtempii.get_array(), old_list_len, nbest) ;

				int numEnt = 0;
				numEnt = old_list_len;

				double minusscore;
				long int fromtjk;

				for (short int k=0; k<nbest; k++)
				{
					if (k<numEnt)
					{
					    minusscore = oldtempvv[k];
					    fromtjk = oldtempii[k];
					    
					    delta.element(t%max_look_back,j,k)    = -minusscore + seq.element(j,t);
					    psi.element(t,j,k)      = (fromtjk%N) ;
					    ktable.element(t,j,k)     = (fromtjk%(N*nbest)-psi.element(t,j,k))/N ;
					    ptable.element(t,j,k)     = (fromtjk-(fromtjk%(N*nbest)))/(N*nbest) ;
					}
					else
					{
						delta.element(t%max_look_back,j,k)    = -CMath::INFTY ;
						psi.element(t,j,k)      = 0 ;
						ktable.element(t,j,k)     = 0 ;
						ptable.element(t,j,k)     = 0 ;
					}
				}
				
			}
		}
	}

	
	{ //termination
		INT list_len = 0 ;
		for (short int diff=0; diff<nbest; diff++)
		{
			for (T_STATES i=0; i<N; i++)
			{
				oldtempvv[list_len] = -(delta.element((seq_len-1)%max_look_back,i,diff)+get_q(i)) ;
				oldtempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(oldtempvv.get_array(), oldtempii.get_array(), list_len, nbest) ;
		
		for (short int k=0; k<nbest; k++)
		{
			delta_end.element(k) = -oldtempvv[k] ;
			path_ends.element(k) = (oldtempii[k]%N) ;
			ktable_end.element(k) = (oldtempii[k]-path_ends.element(k))/N ;
		}
	}
	
	{ //state sequence backtracking		
		for (short int k=0; k<nbest; k++)
		{
			prob_nbest[k]= delta_end.element(k) ;
			
			INT i         = 0 ;
			state_seq[i]  = path_ends.element(k) ;
			short int q   = ktable_end.element(k) ;
			pos_seq[i]    = seq_len-1 ;

			while (pos_seq[i]>0)
			{
				//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				state_seq[i+1] = psi.element(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = ptable.element(pos_seq[i], state_seq[i], q) ;
				q              = ktable.element(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
			INT num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i+k*seq_len] = state_seq[num_states-i-1] ;
			}
			//my_state_seq[num_states+k*seq_len]=-1 ;
		}

	}
}

