
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

// HMM.h: interface for the CHMM class.
//
//////////////////////////////////////////////////////////////////////

#ifndef __CDYNPROG_H__
#define __CDYNPROG_H__

#include "lib/Mathematics.h"
#include "lib/common.h"
#include "lib/io.h"
#include "lib/config.h"
#include "structure/Plif.h"
#include "features/StringFeatures.h"
#include "distributions/Distribution.h"
#include "lib/Array.h"
#include "lib/Array2.h"

#include <stdio.h>

#ifdef USE_BIGSTATES
typedef WORD T_STATES ;
#else
typedef BYTE T_STATES ;
#endif
typedef T_STATES* P_STATES ;


/** Dynamic Programming Class.
 * Structure and Function collection.
 * This Class implements a Dynamic Programming functions.
 */
class CDynProg 
{
private:
	
	T_STATES trans_list_len ;
	T_STATES **trans_list_forward  ;
	T_STATES *trans_list_forward_cnt  ;
	DREAL **trans_list_forward_val ;
	T_STATES **trans_list_backward  ;
	T_STATES *trans_list_backward_cnt  ;
	bool mem_initialized ;
	
public:
	/**@name Constructor/Destructor and helper function
	 */
	//@{
	/** Constructor
	 * @param N number of states
	 * @param M number of emissions
	 * @param model model which holds definitions of states to be learned + consts
	 */
	CDynProg() ;
	//CDynProg(INT N, double* p, double* q, double* a) ;
	//CDynProg(INT N, double* p, double* q, int num_trans, double* a_trans) ;

	/// Destructor - Cleanup
	~CDynProg();
	

	/** calculates probability of best state sequence s_0,...,s_T-1 AND path itself using viterbi algorithm.
	 * The path can be found in the array PATH(dimension)[0..T-1] afterwards
	 * @param dimension dimension of observation for which the most probable path is calculated (observations are a matrix, where a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1} 
	 */
	DREAL best_path_no_b(INT max_iter, INT & best_iter, INT *my_path) ;
	void best_path_no_b_trans(INT max_iter, INT & max_best_iter, short int nbest, DREAL *prob_nbest, INT *my_paths) ;
	//void best_path_no_b_trans1(INT max_iter, INT & max_best_iter, DREAL *prob_nbest, INT *my_paths) ;
	void model_prob_no_b_trans(INT max_iter, DREAL *prob_iter) ;

////////////////////////////////////////////////////////////////////////////////
protected:
	// control info
	INT m_step ;
	INT m_call ;
	// input arguments
	CArray2<DREAL> m_seq ;
	CArray<INT> m_pos ;
	CArray2<INT> m_orf_info ;
	CArray2<DREAL> m_segment_sum_weights ;
	CArray<CPlif*> m_plif_list ;
	CArray2<CPlif*> m_PEN ;
	CArray<CHAR> m_genestr ;
	CArray2<DREAL> m_dict_weights ;

	// output arguments
	CArray<DREAL> m_scores ;
	CArray2<INT> m_states ;
	CArray2<INT> m_positions ;
	
public:
	// model related stuff
	void set_p(DREAL *p, INT N) ;
	void set_q(DREAL *q, INT N) ;
	void set_a(DREAL *a, INT M, INT N) ;
	void set_a_trans(DREAL *a_trans, INT num_trans, INT N) ;

	// content svm related setup functions
	void init_svm_arrays(INT p_num_degrees, INT p_num_svms) ;
	void init_word_degree_array(INT * p_word_degree_array, INT num_elem) ;
	void init_cum_num_words_array(INT * p_cum_num_words_array, INT num_elem) ;
	void init_num_words_array(INT * p_num_words_array, INT num_elem) ;
	bool check_svm_arrays() ; // call this function to check consistency

	// best_path preparation functions
	void best_path_set_seq(DREAL *seq, INT N, INT seq_len) ;
	void best_path_set_pos(INT *pos, INT seq_len)  ;
	void best_path_set_orf_info(INT *orf_info, INT m, INT n) ;            // only for best_path_trans
	void best_path_set_segment_sum_weights(DREAL *segment_sum_weights, INT num_states, INT seq_len) ; // only for best_path_2struct
	void best_path_set_plif_list(CPlif **plif_list, INT num_plif) ;
	void best_path_set_plif_id_matrix(INT *plif_id_matrix, INT m, INT n) ;
	void best_path_set_genestr(CHAR* genestr, INT genestr_len) ;
	void best_path_set_dict_weights(DREAL* dictionary_weights, INT dict_len, INT n) ;
	
	// best_path functions
	void best_path_call(INT nbest, bool use_orf) ;
	void best_path_2struct_call(INT nbest) ;
	void best_path_simple_call(INT nbest) ;
	
	// best_path result retrieval functions
	void best_path_get_scores(DREAL **scores, INT *n) ;
	void best_path_get_states(INT **states, INT *m, INT *n) ;
	void best_path_get_positions(INT **positions, INT *m, INT *n) ;

////////////////////////////////////////////////////////////////////////////////

	void best_path_trans(const DREAL *seq, INT seq_len, const INT *pos, const INT *orf_info,
						 CPlif **PEN_matrix, 
						 const char *genestr, INT genestr_len,
						 short int nbest, 
						 DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
						 DREAL *dictionary_weights, INT dict_len,
						 DREAL *&PEN_values, DREAL *&PEN_input_values, 
						 INT &num_PEN_id, bool use_orf) ;

	void best_path_2struct(const DREAL *seq, INT seq_len, const INT *pos, 
						   CPlif **PEN_matrix, 
						   const char *genestr, INT genestr_len,
						   short int nbest, 
						   DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
						   DREAL *dictionary_weights, INT dict_len, DREAL *segment_sum_weights, 
						   DREAL *&PEN_values, DREAL *&PEN_input_values, 
						   INT &num_PEN_id) ;
	void best_path_trans_simple(const DREAL *seq, INT seq_len, short int nbest, 
								DREAL *prob_nbest, INT *my_state_seq) ;

	
	
	/// access function for number of states N
	inline T_STATES get_N() const
	  {
	    return N ;
	  }
	
	/** access function for probability of end states
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_q(T_STATES offset, DREAL value)
	{
		end_state_distribution_q[offset]=value;
	}

	/** access function for probability of first state
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_p(T_STATES offset, DREAL value)
	{
		initial_state_distribution_p[offset]=value;
	}

	/** access function for matrix a 
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @param value value to be set
	 */
	inline void set_a(T_STATES line_, T_STATES column, DREAL value)
	{
	  transition_matrix_a.element(line_,column)=value; // look also best_path!
	}

	/** access function for probability of end states
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_q(T_STATES offset) const 
	{
		return end_state_distribution_q[offset];
	}

	/** access function for probability of initial states
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_p(T_STATES offset) const 
	{
		return initial_state_distribution_p[offset];
	}

	/** access function for matrix a
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline DREAL get_a(T_STATES line_, T_STATES column) const
	{
	  return transition_matrix_a.element(line_,column) ; // look also best_path()!
	}
	//@}
protected:

	/* helper functions */
	void translate_from_single_order(WORD* obs, INT sequence_length, 
									 INT start, INT order, 
									 INT max_val=2/*DNA->2bits*/) ;

	void reset_svm_value(INT pos, INT & last_svm_pos, DREAL * svm_value)  ;
	void extend_svm_value(WORD* wordstr, INT pos, INT &last_svm_pos, DREAL* svm_value) ;
	void reset_segment_sum_value(INT num_states, INT pos, INT & last_segment_sum_pos, DREAL * segment_sum_value) ;
	void extend_segment_sum_value(DREAL *segment_sum_weights, INT seqlen, INT num_states,
								  INT pos, INT &last_segment_sum_pos, DREAL* segment_sum_value) ;

	struct svm_values_struct
	{
		INT maxlookback ;
		INT seqlen;
		INT *num_unique_words ;
		
		DREAL ** svm_values_unnormalized ;
		DREAL * svm_values ;
		bool ** word_used ;
	} ;

	void reset_svm_values(INT pos, INT * last_svm_pos, DREAL * svm_value) ;
	void extend_svm_values(WORD** wordstr, INT pos, INT *last_svm_pos, DREAL* svm_value) ;
	void init_svm_values(struct svm_values_struct & svs, INT start_pos, INT seqlen, INT howmuchlookback) ;
	void clear_svm_values(struct svm_values_struct & svs) ;
	void find_svm_values_till_pos(WORD** wordstr,  const INT *pos,  INT t_end, struct svm_values_struct &svs) ;
	bool extend_orf(const CArray<bool>& genestr_stop, INT orf_from, INT orf_to, INT start, INT &last_pos, INT to) ;

	/**@name model specific variables.
	 * these are p,q,a,b,N,M etc 
	 */
	//@{
	/// number of states
	INT N;

	/// transition matrix 
	CArray2<DREAL> transition_matrix_a;

	/// initial distribution of states
	CArray<DREAL> initial_state_distribution_p;

	/// distribution of end-states
	CArray<DREAL> end_state_distribution_q;		

	//@}
	
	CArray2<DREAL> dict_weights ;
	DREAL * dict_weights_array ;

	INT num_degrees ;
	INT num_svms  ;
	
	CArray<INT> word_degree ;
	CArray<INT> cum_num_words ;
	INT * cum_num_words_array ;
	CArray<INT> num_words ;
	INT * num_words_array ;

	CArray2<bool> word_used ;
	bool *word_used_array ;
	CArray2<DREAL> svm_values_unnormalized ;
	CArray<INT> svm_pos_start ;
	CArray<INT> num_unique_words ;
	bool svm_arrays_clean ;

	INT num_svms_single  ;
	INT word_degree_single ;
	INT cum_num_words_single ;
	INT num_words_single ;

	CArray<bool> word_used_single ;
	CArray<DREAL> svm_value_unnormalized_single ;
	INT num_unique_words_single ;
};
#endif
