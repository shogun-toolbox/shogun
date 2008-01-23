/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __CDYNPROG_H__
#define __CDYNPROG_H__

#include "lib/Mathematics.h"
#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/config.h"
#include "structure/PlifBase.h"
#include "features/StringFeatures.h"
#include "distributions/Distribution.h"
#include "lib/DynamicArray.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"
#include "lib/Time.h"

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
class CDynProg : public CSGObject
{
private:
	
	T_STATES trans_list_len ;
	T_STATES **trans_list_forward  ;
	T_STATES *trans_list_forward_cnt  ;
	DREAL **trans_list_forward_val ;
	INT **trans_list_forward_id ;
	bool mem_initialized ;

#ifdef DYNPROG_TIMING
	CTime MyTime ;
	CTime MyTime2 ;
	
	DREAL segment_init_time ;
	DREAL segment_pos_time ;
	DREAL segment_clean_time ;
	DREAL segment_extend_time ;
	DREAL orf_time ;
	DREAL svm_init_time ;
	DREAL svm_pos_time ;
	DREAL svm_clean_time ;
#endif
	
public:
	CDynProg(INT p_num_svms = 8) ;
	~CDynProg();
	
	DREAL best_path_no_b(INT max_iter, INT & best_iter, INT *my_path) ;
	void best_path_no_b_trans(INT max_iter, INT & max_best_iter, short int nbest, DREAL *prob_nbest, INT *my_paths) ;
	
	// model related functions
	void set_N(INT p_N); // use this function to set N first
	void set_p_vector(DREAL* p, INT N);
	void set_q_vector(DREAL* q, INT N);
	void set_a(DREAL* a, INT M, INT N);
	void set_a_id(INT *a, INT M, INT N);
	void set_a_trans_matrix(DREAL *a_trans, INT num_trans, INT N);

	// content svm related setup functions
	void init_svm_arrays(INT p_num_degrees, INT p_num_svms) ;
	void init_word_degree_array(INT * p_word_degree_array, INT num_elem) ;
	void init_cum_num_words_array(INT * p_cum_num_words_array, INT num_elem) ;
	void init_num_words_array(INT * p_num_words_array, INT num_elem) ;
	void init_mod_words_array(INT * p_mod_words_array, INT num_elem, INT num_columns) ;
	void init_sign_words_array(bool * p_sign_words_array, INT num_elem) ;
	void init_string_words_array(INT * p_string_words_array, INT num_elem) ;
	bool check_svm_arrays() ; // call this function to check consistency

	// best_path_trans preparation functions
	void best_path_set_seq(DREAL *seq, INT N, INT seq_len) ;
	void best_path_set_seq3d(DREAL *seq, INT p_N, INT seq_len, INT max_num_signals) ;
	void best_path_set_pos(INT *pos, INT seq_len)  ;
	void best_path_set_orf_info(INT *orf_info, INT m, INT n) ;            // only for best_path_trans
	void best_path_set_segment_sum_weights(DREAL *segment_sum_weights, INT num_states, INT seq_len) ; // only for best_path_2struct
	void best_path_set_plif_list(CDynamicArray<CPlifBase*>* plifs);
	void best_path_set_plif_id_matrix(INT *plif_id_matrix, INT m, INT n) ;
	void best_path_set_plif_state_signal_matrix(INT *plif_id_matrix, INT m, INT n) ;
	void best_path_set_genestr(CHAR* genestr, INT genestr_len, INT genestr_num) ; // genestr_num is typically 1

	// additional best_path_trans_deriv functions
	void best_path_set_my_state_seq(INT* my_state_seq, INT seq_len);
	void best_path_set_my_pos_seq(INT* my_pos_seq, INT seq_len);

	inline void best_path_set_single_genestr(CHAR* genestr, INT genestr_len)
	{
		SG_DEBUG("genestrpy: %d", genestr_len);
		best_path_set_genestr(genestr, genestr_len, 1);
	}
	void best_path_set_dict_weights(DREAL* dictionary_weights, INT dict_len, INT n) ;
	void best_path_set_segment_loss(DREAL * segment_loss, INT num_segment_id1, INT num_segment_id2) ;
	void best_path_set_segment_ids_mask(INT* segment_ids_mask, INT m, INT n) ;

	// best_path functions
	void best_path_call(INT nbest, bool use_orf) ;
	void best_path_deriv_call() ;
	void best_path_2struct_call(INT nbest) ;
	void best_path_simple_call(INT nbest) ;
	void best_path_deriv_call(INT nbest) ;
	
	// best_path result retrieval functions
	void best_path_get_scores(DREAL **scores, INT *n) ;
	void best_path_get_states(INT **states, INT *m, INT *n) ;
	void best_path_get_positions(INT **positions, INT *m, INT *n) ;

	//best_path_trans_deriv result retrieval functions
	void best_path_get_losses(DREAL** my_losses, INT* seq_len);

////////////////////////////////////////////////////////////////////////////////

	template <short int nbest, bool with_loss, bool with_multiple_sequences>
	void best_path_trans(const DREAL *seq, INT seq_len, const INT *pos, 
						 const INT *orf_info, CPlifBase **PLif_matrix, 
						 CPlifBase **Plif_state_signals, INT max_num_signals, 
						 const char *genestr, INT genestr_len, INT genestr_num, 
						 DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
						 DREAL *dictionary_weights, INT dict_len, bool use_orf) ;

	void best_path_trans_deriv(INT *my_state_seq, INT *my_pos_seq, DREAL *my_scores, DREAL* my_losses,
							   INT my_seq_len, 
							   const DREAL *seq_array, INT seq_len, const INT *pos,
							   CPlifBase **Plif_matrix, 
							   CPlifBase **Plif_state_signals, INT max_num_signals, 
							   const char *genestr, INT genestr_len, INT genestr_num,
							   DREAL *dictionary_weights, INT dict_len) ;
	
	void best_path_2struct(const DREAL *seq, INT seq_len, const INT *pos, 
						   CPlifBase **Plif_matrix, 
						   const char *genestr, INT genestr_len,
						   short int nbest, 
						   DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
						   DREAL *dictionary_weights, INT dict_len, DREAL *segment_sum_weights) ;
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

	inline DREAL get_q_deriv(T_STATES offset) const 
	{
		return end_state_distribution_q_deriv[offset];
	}

	/** access function for probability of initial states
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_p(T_STATES offset) const 
	{
		return initial_state_distribution_p[offset];
	}

	inline DREAL get_p_deriv(T_STATES offset) const 
	{
		return initial_state_distribution_p_deriv[offset];
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

	inline DREAL get_a_deriv(T_STATES line_, T_STATES column) const
	{
	  return transition_matrix_a_deriv.element(line_,column) ; // look also best_path()!
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
		
		INT* start_pos ;
		DREAL ** svm_values_unnormalized ;
		DREAL * svm_values ;
		bool *** word_used ;
		INT **num_unique_words ;
	} ;

	//void reset_svm_values(INT pos, INT * last_svm_pos, DREAL * svm_value) ;
	//void extend_svm_values(WORD** wordstr, INT pos, INT *last_svm_pos, DREAL* svm_value) ;
	void init_svm_values(struct svm_values_struct & svs, INT start_pos, INT seqlen, INT howmuchlookback) ;
	void clear_svm_values(struct svm_values_struct & svs) ;
	void find_svm_values_till_pos(WORD*** wordstr,  const INT *pos,  INT t_end, struct svm_values_struct &svs) ;
	void find_svm_values_till_pos(WORD** wordstr,  const INT *pos,  INT t_end, struct svm_values_struct &svs) ;
	void update_svm_values_till_pos(WORD*** wordstr,  const INT *pos,  INT t_end, INT prev_t_end, struct svm_values_struct &svs) ;
	bool extend_orf(const CArray<bool>& genestr_stop, INT orf_from, INT orf_to, INT start, INT &last_pos, INT to) ;

	struct segment_loss_struct
	{
		INT maxlookback ;
		INT seqlen;
		INT *segments_changed ;
		INT *num_segment_id ;
		INT *length_segment_id ;
	} ;

	void init_segment_loss(struct segment_loss_struct & loss, INT seqlen, INT howmuchlookback);
	void clear_segment_loss(struct segment_loss_struct & loss) ;
	DREAL extend_segment_loss(struct segment_loss_struct & loss, const INT * pos_array, INT segment_id, INT pos, INT& last_pos, DREAL &last_value) ;
	void find_segment_loss_till_pos(const INT * pos, INT t_end, CArray2<INT>& segment_ids, struct segment_loss_struct & loss) ;

	/**@name model specific variables.
	 * these are p,q,a,b,N,M etc 
	 */
	//@{
	/// number of states
	INT N;

	/// transition matrix 
	CArray2<INT> transition_matrix_a_id ;
	CArray2<DREAL> transition_matrix_a;
	CArray2<DREAL> transition_matrix_a_deriv ;

	/// initial distribution of states
	CArray<DREAL> initial_state_distribution_p;
	CArray<DREAL> initial_state_distribution_p_deriv ;

	/// distribution of end-states
	CArray<DREAL> end_state_distribution_q;		
	CArray<DREAL> end_state_distribution_q_deriv ;		

	//@}
	
	CArray2<DREAL> dict_weights ;
	DREAL * dict_weights_array ;

	INT num_degrees ;
	INT num_svms  ;
	INT num_strings ;
	
	CArray<INT> word_degree ;
	CArray<INT> cum_num_words ;
	INT * cum_num_words_array ;
	CArray<INT> num_words ;
	INT * num_words_array ;
	CArray2<INT> mod_words ;
	INT * mod_words_array ;
	CArray<bool> sign_words ;
	bool * sign_words_array ;
	CArray<INT> string_words ;
	INT * string_words_array ;

//	CArray3<INT> word_used ;
//	INT *word_used_array ;
//	CArray2<DREAL> svm_values_unnormalized ;
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

	INT max_a_id ;
	
	// control info
	INT m_step ;
	INT m_call ;
	// input arguments
	CArray3<DREAL> m_seq ;
	CArray<INT> m_pos ;
	CArray2<INT> m_orf_info ;
	CArray2<DREAL> m_segment_sum_weights ;
	CArray<CPlifBase*> m_plif_list ;
	CArray2<CPlifBase*> m_PEN ;
	CArray2<CPlifBase*> m_PEN_state_signals ;
	CArray2<CHAR> m_genestr ;
	CArray2<DREAL> m_dict_weights ;
	CArray3<DREAL> m_segment_loss ;
	CArray2<INT> m_segment_ids_mask ;
	CArray<INT> m_my_state_seq;
	CArray<INT> m_my_pos_seq;
	CArray<DREAL> m_my_scores;
	CArray<DREAL> m_my_losses;

	// output arguments
	CArray<DREAL> m_scores ;
	CArray2<INT> m_states ;
	CArray2<INT> m_positions ;
};
#endif
