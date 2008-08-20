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
#include "structure/Plif.h"
#include "features/StringFeatures.h"
#include "distributions/Distribution.h"
#include "lib/DynamicArray.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"
#include "lib/Time.h"

#include <stdio.h>

//#define DYNPROG_TIMING

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

	T_STATES trans_list_len;
	T_STATES **trans_list_forward;
	T_STATES *trans_list_forward_cnt;
	DREAL **trans_list_forward_val;
	INT **trans_list_forward_id;
	bool mem_initialized;

#ifdef DYNPROG_TIMING
	CTime MyTime;
	CTime MyTime2;
	
	DREAL segment_init_time;
	DREAL segment_pos_time;
	DREAL segment_clean_time;
	DREAL segment_extend_time;
	DREAL orf_time;
	DREAL content_time;
	DREAL content_penalty_time;
	DREAL svm_init_time;
	DREAL svm_pos_time;
	DREAL svm_clean_time;
#endif
	
public:
	/** constructor
	 *
	 * @param p_num_svms number of SVMs
	 */
	CDynProg(INT p_num_svms=8);
	~CDynProg();

	/** best path no b
	 *
	 * @param max_iter max iter
	 * @param best_iter best iter
	 * @param my_path my path
	 *
	 * @return best path no b
	 */
	DREAL best_path_no_b(INT max_iter, INT & best_iter, INT *my_path);

	/** best path no b transition
	 *
	 * @param max_iter max iter
	 * @param max_best_iter max best iter
	 * @param nbest nbest
	 * @param prob_nbest prob_nbest
	 * @param my_paths my paths
	 */
	void best_path_no_b_trans(INT max_iter, INT & max_best_iter, SHORT nbest, DREAL *prob_nbest, INT *my_paths);
	
	// model related functions
	/** set number of states
	 * use this to set N first
	 *
	 * @param p_N new N
	 */
	void set_num_states(INT p_N);

	/** get num states */
	INT get_num_states();

	/** get num svms*/
	INT get_num_svms();

	/** init CArray for precomputed content svm values
	 *  with size seq_len x num_svms
	 *
	 *  @param seq_len: number of candidate positions
	 */
	void init_content_svm_value_array(const INT seq_len);

	/** init CArray for precomputed tiling intensitie-plif-values
	 *  with size seq_len x num_svms
	 *
	 *  @param probe_pos local positions of probes
	 *  @param intensities intensities of probes
	 *  @param num_probes number of probes
	 *  @param seq_len: number of candidate positions
	 */
	void init_tiling_data(INT* probe_pos, DREAL* intensities, const INT num_probes, const INT seq_len);

	/** precompute tiling Plifs
	 *
	 * @param PEN Plif PEN
	 * @param tiling_plif_ids tiling plif id's
	 * @param num_tiling_plifs number of tiling plifs
	 * @param seq_len sequence length
	 * @param pos pos
	 */
	void precompute_tiling_plifs(CPlif** PEN, const INT* tiling_plif_ids, const INT num_tiling_plifs, const INT seq_len, const INT* pos);	

	/** append rows to linear features array
 	 * 
 	 * @param num_new_feat number of new rows to add
 	 * @param seq_len number of columns (must be equal to the existing num of cols)
 	 */
	void resize_lin_feat(INT num_new_feat, INT seq_len);
	/** set vector p
	 *
	 * @param p new vector p
	 * @param N size of vector p
	 */
	void set_p_vector(DREAL* p, INT N);

	/** set vector q
	 *
	 * @param q new vector q
	 * @param N size of vector q
	 */
	void set_q_vector(DREAL* q, INT N);
	
	/** set matrix a
	 *
	 * @param a new matrix a
	 * @param M dimension M of matrix a
	 * @param N dimension N of matrix a
	 */
	void set_a(DREAL* a, INT M, INT N);
	
	/** set a id
	 *
	 * @param a new a id (identity?)
	 * @param M dimension M of matrix a
	 * @param N dimension N of matrix a
	 */
	void set_a_id(INT *a, INT M, INT N);
	
	/** set a transition matrix
	 *
	 * @param a_trans transition matrix a
	 * @param num_trans number of transitions
	 * @param N dimension N of matrix a
	 */
	void set_a_trans_matrix(DREAL *a_trans, INT num_trans, INT N);

	// content svm related setup functions
	/** init SVM arrays
	 *
	 * @param p_num_degrees number of degrees
	 * @param p_num_svms number of SVMs
	 */
	void init_svm_arrays(INT p_num_degrees, INT p_num_svms);

	/** init word degree array
	 *
	 * @param p_word_degree_array new word degree array
	 * @param num_elem number of array elements
	 */
	void init_word_degree_array(INT * p_word_degree_array, INT num_elem);

	/** init cum num words array
	 *
	 * @param p_cum_num_words_array new cum num words array
	 * @param num_elem number of array elements
	 */
	void init_cum_num_words_array(INT * p_cum_num_words_array, INT num_elem);

	/** init num words array
	 *
	 * @param p_num_words_array new num words array
	 * @param num_elem number of array elements
	 */
	void init_num_words_array(INT * p_num_words_array, INT num_elem);

	/** init mod words array
	 *
	 * @param p_mod_words_array new mod words array
	 * @param num_elem number of array elements
	 * @param num_columns number of columns
	 */
	void init_mod_words_array(INT * p_mod_words_array, INT num_elem, INT num_columns);

	/** init sign words array
	 *
	 * @param p_sign_words_array new sign words array
	 * @param num_elem number of array elements
	 */
	void init_sign_words_array(bool * p_sign_words_array, INT num_elem);

	/** init string words array
	 *
	 * @param p_string_words_array new string words array
	 * @param num_elem number of array elements
	 */
	void init_string_words_array(INT * p_string_words_array, INT num_elem);

	/** check SVM arrays
	 * call this function to check consistency
	 *
	 * @return whether arrays are ok
	 */
	bool check_svm_arrays();

	// best_path_trans preparation functions
	/** set best path seq
	 *
	 * @param seq the sequence
	 * @param N dimension N
	 * @param seq_len length of sequence
	 */
	void best_path_set_seq(DREAL *seq, INT N, INT seq_len);

	/** set best path seq3d
	 *
	 * @param seq the 3D sequence
	 * @param p_N dimension N
	 * @param seq_len length of sequence
	 * @param max_num_signals maximal number of signals
	 */
	void best_path_set_seq3d(DREAL *seq, INT p_N, INT seq_len, INT max_num_signals);

	/** set best path pos
	 *
	 * @param pos the position
	 * @param seq_len length of sequence
	 */
	void best_path_set_pos(INT *pos, INT seq_len);

	/** set best path orf info
	 * only for best_path_trans
	 *
	 * @param orf_info the orf info
	 * @param m dimension m
	 * @param n dimension n
	 */
	void best_path_set_orf_info(INT *orf_info, INT m, INT n);

	/** set best path segment sum weights
	 * only for best_path_2struct
	 *
	 * @param segment_sum_weights segment sum weights
	 * @param num_states number of states
	 * @param seq_len length of sequence
	 */
	void best_path_set_segment_sum_weights(DREAL *segment_sum_weights, INT num_states, INT seq_len);

	/** set best path Plif list
	 *
	 * @param plifs list of Plifs
	 */
	void best_path_set_plif_list(CDynamicArray<CPlifBase*>* plifs);

	/** set best path plif id(entity?) matrix
	 *
	 * @param plif_id_matrix plif id matrix
	 * @param m dimension m of matrix
	 * @param n dimension n of matrix
	 */
	void best_path_set_plif_id_matrix(INT *plif_id_matrix, INT m, INT n);

	/** set best path plif state signal matrix
	 *
	 * @param plif_id_matrix plif id matrix
	 * @param m dimension m of matrix
	 * @param n dimension n of matrix
	 */
	void best_path_set_plif_state_signal_matrix(INT *plif_id_matrix, INT m, INT n);

	/** set best path genesstr
	 *
	 * @param genestr gene string
	 * @param genestr_len length of gene string
	 * @param genestr_num number of gene strings, typically 1
	 */
	void best_path_set_genestr(CHAR* genestr, INT genestr_len, INT genestr_num);

	// additional best_path_trans_deriv functions
	/** set best path my state sequence
	 *
	 * @param my_state_seq my state sequence
	 * @param seq_len length of sequence
	 */
	void best_path_set_my_state_seq(INT* my_state_seq, INT seq_len);

	/** set best path my position sequence
	 *
	 * @param my_pos_seq my position sequence
	 * @param seq_len length of sequence
	 */
	void best_path_set_my_pos_seq(INT* my_pos_seq, INT seq_len);

	/** set best path single gene string
	 *
	 * @param genestr gene string
	 * @param genestr_len length of gene string
	 */
	inline void best_path_set_single_genestr(CHAR* genestr, INT genestr_len)
	{
		SG_DEBUG("genestrpy: %d", genestr_len);
		best_path_set_genestr(genestr, genestr_len, 1);
	}

	/** set best path dict weights
	 *
	 * @param dictionary_weights dictionary weights
	 * @param dict_len length of dictionary weights
	 * @param n dimension n
	 */
	void best_path_set_dict_weights(DREAL* dictionary_weights, INT dict_len, INT n);

	/** set best path segment loss
	 *
	 * @param segment_loss segment loss
	 * @param num_segment_id1 number of segment id1
	 * @param num_segment_id2 number of segment id2
	 */
	void best_path_set_segment_loss(DREAL * segment_loss, INT num_segment_id1, INT num_segment_id2);

	/** set best path segmend ids mask
	 *
	 * @param segment_ids segment ids
	 * @param segment_mask segment mask
	 * @param m dimension m
	 */
	void best_path_set_segment_ids_mask(INT* segment_ids, DREAL* segment_mask, INT m);

	// best_path functions
	/** best path call
	 *
	 * @param nbest nbest
	 * @param use_orf whether to use orf
	 */
	void best_path_call(INT nbest, bool use_orf);

	/** best path derivative call */
	void best_path_deriv_call();

	/** best path 2struct call
	 *
	 * @param nbest nbest
	 */
	void best_path_2struct_call(INT nbest);

	/** best path simple call
	 *
	 * @param nbest nbest
	 */
	void best_path_simple_call(INT nbest);

	/** best path derivative call
	 *
	 * @param nbest nbest
	 */
	void best_path_deriv_call(INT nbest);
	
	// best_path result retrieval functions
	/** best path get scores
	 *
	 * @param scores scores
	 * @param n dimension n
	 */
	void best_path_get_scores(DREAL **scores, INT *n);

	/** best path get states
	 *
	 * @param states states
	 * @param m dimension m
	 * @param n dimension n
	 */
	void best_path_get_states(INT **states, INT *m, INT *n);

	/** best path get positions
	 *
	 * @param positions positions
	 * @param m dimension m
	 * @param n dimension n
	 */
	void best_path_get_positions(INT **positions, INT *m, INT *n);

	//best_path_trans_deriv result retrieval functions
	/** get best path losses
	 *
	 * @param my_losses my losses
	 * @param seq_len length of sequence
	 */
	void best_path_get_losses(DREAL** my_losses, INT* seq_len);

////////////////////////////////////////////////////////////////////////////////

	/** best path trans
	 *
	 * @param seq sequence
	 * @param seq_len length of sequence
	 * @param pos position
	 * @param orf_info orf info
	 * @param PLif_matrix Plif matrix
	 * @param Plif_state_signals Plif state signals
	 * @param max_num_signals maximal number of signals
	 * @param genestr_num number of gene strings
	 * @param prob_nbest prob nbest
	 * @param my_state_seq my state seq
	 * @param my_pos_seq my pos seq
	 * @param use_orf whether orf shall be used
	 */
	template <short int nbest, bool with_loss, bool with_multiple_sequences>
	void best_path_trans(const DREAL *seq, INT seq_len, const INT *pos,
						 const INT *orf_info, CPlifBase **PLif_matrix,
						 CPlifBase **Plif_state_signals, INT max_num_signals,
						 INT genestr_num,
						 DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
						 bool use_orf);

	/** best path trans derivative
	 *
	 * @param my_state_seq my state seq
	 * @param my_pos_seq my pos seq
	 * @param my_scores my scores
	 * @param my_losses my losses
	 * @param my_seq_len my sequence length
	 * @param seq_array sequence array
	 * @param seq_len length of sequence
	 * @param pos position
	 * @param Plif_matrix Plif matrix
	 * @param Plif_state_signals Plif state signals
	 * @param max_num_signals maximal number of signals
	 * @param genestr_num number of gene strings
	 */
	void best_path_trans_deriv(INT *my_state_seq, INT *my_pos_seq, DREAL *my_scores, DREAL* my_losses, INT my_seq_len,
					const DREAL *seq_array, INT seq_len, const INT *pos, CPlifBase **Plif_matrix,
					CPlifBase **Plif_state_signals, INT max_num_signals, INT genestr_num);
	
	/** best path 2struct
	 *
	 * @param seq sequence
	 * @param seq_len length of sequence
	 * @param pos position
	 * @param Plif_matrix Plif matrix
	 * @param genestr gene string
	 * @param genestr_len length of gene string
	 * @param nbest nbest
	 * @param prob_nbest prob(ability?) nbest
	 * @param my_state_seq my state seq
	 * @param my_pos_seq my pos seq
	 * @param dictionary_weights dictionary weights
	 * @param dict_len length of dictionary weights
	 * @param segment_sum_weights segment sum weights
	 */
	void best_path_2struct(const DREAL *seq, INT seq_len, const INT *pos,
						   CPlifBase **Plif_matrix,
						   const char *genestr, INT genestr_len,
						   SHORT nbest,
						   DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
						   DREAL *dictionary_weights, INT dict_len, DREAL *segment_sum_weights);

	/** best path trans simple
	 *
	 * @param seq sequence
	 * @param seq_len length of sequence
	 * @param nbest nbest
	 * @param prob_nbest prob(ability?) nbest
	 * @param my_state_seq my state seq
	 */
	void best_path_trans_simple(const DREAL *seq, INT seq_len, SHORT nbest,
								DREAL *prob_nbest, INT *my_state_seq);



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
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @param value value to be set
	 */
	inline void set_a(T_STATES line_, T_STATES column, DREAL value)
	{
	  transition_matrix_a.element(line_,column)=value; // look also best_path!
	}

	/** access function for probability of end states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_q(T_STATES offset) const
	{
		return end_state_distribution_q[offset];
	}

	/** access function for derivated probability of end states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_q_deriv(T_STATES offset) const
	{
		return end_state_distribution_q_deriv[offset];
	}

	/** access function for probability of initial states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_p(T_STATES offset) const
	{
		return initial_state_distribution_p[offset];
	}

	/** access function for derivated probability of initial states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_p_deriv(T_STATES offset) const
	{
		return initial_state_distribution_p_deriv[offset];
	}
	
	/** create array of precomputed content svm values
	 * Jonas
	 *
	 * @param wordstr word strings
	 * @param pos position
	 * @param num_cand_pos number of cand position
	 * @param genestr_len length of gene string
	 * @param dictionary_weights dictionary weights
	 * @param dict_len lenght of dictionary
	 */
	void precompute_content_values(WORD*** wordstr, const INT *pos,
		const INT num_cand_pos, const INT genestr_len,
		DREAL *dictionary_weights, INT dict_len);

	/** create word string from char*
	 * Jonas
	 *
	 * @param genestr gene string
	 * @param genestr_num number of gene string
	 * @param genestr_len length of gene string
	 * @param wordstr word strings
	 */
	void create_word_string(const CHAR* genestr, INT genestr_num, INT genestr_len, WORD*** wordstr);

	/** precompute stop codons
	 *
	 * @param genestr gene string
	 * @param genestr_len length of gene string
	 */
	void precompute_stop_codons(const CHAR* genestr, INT genestr_len);

	/** set genestr len
	 *
	 * @param genestr_len length of gene string
	 *
	 */
	void set_genestr_len(INT genestr_len);

	/** access function for matrix a
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline DREAL get_a(T_STATES line_, T_STATES column) const
	{
	  return transition_matrix_a.element(line_,column); // look also best_path()!
	}

	/** access function for matrix a derivated
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline DREAL get_a_deriv(T_STATES line_, T_STATES column) const
	{
	  return transition_matrix_a_deriv.element(line_,column); // look also best_path()!
	}
	//@}
protected:

	/* helper functions */

	/** lookup content SVM values
	 *
	 * @param from_state from state
	 * @param to_state to state
	 * @param from_pos from position
	 * @param to_pos to position
	 * @param svm_values SVM values
	 * @param frame frame
	 */
	inline void lookup_content_svm_values(const INT from_state,
		const INT to_state, const INT from_pos, const INT to_pos,
		DREAL* svm_values, INT frame);

	/** lookup tiling Plif values
	 *
	 * @param from_state from state
	 * @param to_state to state
	 * @param len length
	 * @param svm_values SVM values
	 */
	inline void lookup_tiling_plif_values(const INT from_state,
		const INT to_state, const INT len, DREAL* svm_values);

	/** find frame
	 *
	 * @param from_state from state
	 */
	inline INT find_frame(const INT from_state);

	/** raw intensities interval query
	 *
	 * @param from_pos from position
	 * @param to_pos to position
	 * @param intensities intensities
	 * @return an integer
	 */
	inline INT raw_intensities_interval_query(
		const INT from_pos, const INT to_pos, DREAL* intensities, INT type);

	/** translate from single order
	 *
	 * @param obs observation matrix
	 * @param sequence_length length of sequence
	 * @param start start
	 * @param order order
	 * @param max_val maximum number of bits, e.g. 2 for DNA
	 */
	void translate_from_single_order(WORD* obs, INT sequence_length, INT start,
		INT order, INT max_val=2);

	/** reset SVM value
	 *
	 * @param pos position
	 * @param last_svm_pos last SVM position
	 * @param svm_value value to set
	 */
	void reset_svm_value(INT pos, INT & last_svm_pos, DREAL * svm_value);

	/** extend SVM value
	 *
	 * @param wordstr word string
	 * @param pos position
	 * @param last_svm_pos lsat SVM position
	 * @param svm_value value to set
	 */
	void extend_svm_value(WORD* wordstr, INT pos, INT &last_svm_pos,
		DREAL* svm_value);

	/** reset segment sum value
	 *
	 * @param num_states number of states
	 * @param pos position
	 * @param last_segment_sum_pos last segment sum position
	 * @param segment_sum_value value to set
	 */
	void reset_segment_sum_value(INT num_states, INT pos,
		INT & last_segment_sum_pos, DREAL * segment_sum_value);

	/** extend segment sum value
	 *
	 * @param segment_sum_weights segment sum weights
	 * @param seqlen length of sequence
	 * @param num_states number of states
	 * @param pos position
	 * @param last_segment_sum_pos last segment sum position
	 * @param segment_sum_value value to set
	 */
	void extend_segment_sum_value(DREAL *segment_sum_weights, INT seqlen,
		INT num_states, INT pos, INT &last_segment_sum_pos,
		DREAL* segment_sum_value);

	/** SVM values */
	struct svm_values_struct
	{
		/** maximum lookback */
		INT maxlookback;
		/** sequence length */
		INT seqlen;

		/** start position */
		INT* start_pos;
		/** SVM values normalized */
		DREAL ** svm_values_unnormalized;
		/** SVM values */
		DREAL * svm_values;
		/** word used */
		bool *** word_used;
		/** number of unique words */
		INT **num_unique_words;
	};

	//void reset_svm_values(INT pos, INT * last_svm_pos, DREAL * svm_value) ;
	//void extend_svm_values(WORD** wordstr, INT pos, INT *last_svm_pos, DREAL* svm_value) ;
	/** init SVM values
	 *
	 * @param svs SVM values
	 * @param start_pos start position
	 * @param seqlen length of sequence
	 * @param howmuchlookback how far to look back
	 */
	void init_svm_values(struct svm_values_struct & svs, INT start_pos,
		INT seqlen, INT howmuchlookback);

	/** clear SVM values
	 *
	 * @param svs SVM values
	 */
	void clear_svm_values(struct svm_values_struct & svs);

	/** find SVM values till position (swig compatible?)
	 *
	 * @param wordstr word string
	 * @param pos position
	 * @param t_end t end
	 * @param svs SVM values
	 */
	void find_svm_values_till_pos(WORD*** wordstr, const INT *pos, INT t_end,
		struct svm_values_struct &svs);

	/** find SVM values till position
	 *
	 * @param wordstr word string
	 * @param pos position
	 * @param t_end t end
	 * @param svs SVM values
	 */
	void find_svm_values_till_pos(WORD** wordstr, const INT *pos, INT t_end,
		struct svm_values_struct &svs);

	/** update SVM values till position
	 *
	 * @param wordstr word string
	 * @param pos position
	 * @param t_end t end
	 * @param prev_t_end previous t end
	 * @param svs SVM values
	 */
	void update_svm_values_till_pos(WORD*** wordstr, const INT *pos, INT t_end,
		INT prev_t_end, struct svm_values_struct &svs);

	/** extend orf
	 *
	 * @param orf_from orf from
	 * @param orf_to orf to
	 * @param start start
	 * @param last_pos last position
	 * @param to to
	 */
	bool extend_orf(INT orf_from, INT orf_to, INT start, INT &last_pos, INT to);

	/** segment loss */
	struct segment_loss_struct
	{
		/** maximum lookback */
		INT maxlookback;
		/** sequence length */
		INT seqlen;
		/** segments changed */
		INT *segments_changed;
		/** numb segment ID */
		DREAL *num_segment_id;
		/** length of segmend ID */
		INT *length_segment_id ;
	};

	/** init segment loss
	 *
	 * @param loss segment loss to init
	 * @param seqlen length of sequence
	 * @param howmuchlookback how far to look back
	 */
	void init_segment_loss(struct segment_loss_struct & loss, INT seqlen,
		INT howmuchlookback);

	/** clear segment loss
	 *
	 * @param loss segment loss to clear
	 */
	void clear_segment_loss(struct segment_loss_struct & loss);

	/** extend segment loss
	 *
	 * @param loss segment loss to extend
	 * @param pos_array position array
	 * @param segment_id ID of segment
	 * @param pos position
	 * @param last_pos last position
	 * @param last_value last value
	 * @return last value
	 */
	DREAL extend_segment_loss(struct segment_loss_struct & loss,
		const INT * pos_array, INT segment_id, INT pos, INT& last_pos,
		DREAL &last_value);

	/** find segment loss till pos
	 *
	 * @param pos position
	 * @param t_end t end
	 * @param segment_ids segment IDs
	 * @param segment_mask segmend mask
	 * @param loss segment loss
	 */
	void find_segment_loss_till_pos(const INT * pos, INT t_end,
		CArray<INT>& segment_ids, CArray<DREAL>& segment_mask,
		struct segment_loss_struct& loss);

	
	/**@name model specific variables.
	 * these are p,q,a,b,N,M etc
	 */
	//@{
	/// number of states
	INT N;

	/// transition matrix
	CArray2<INT> transition_matrix_a_id;
	CArray2<DREAL> transition_matrix_a;
	CArray2<DREAL> transition_matrix_a_deriv;

	/// initial distribution of states
	CArray<DREAL> initial_state_distribution_p;
	CArray<DREAL> initial_state_distribution_p_deriv;

	/// distribution of end-states
	CArray<DREAL> end_state_distribution_q;
	CArray<DREAL> end_state_distribution_q_deriv;

	//@}
	
	/** dict weights */
	CArray2<DREAL> dict_weights;
	/** dict weights array */
	DREAL * dict_weights_array;

	/** number of degress */
	INT num_degrees;
	/** number of SVMs */
	INT num_svms;
	/** number of strings */
	INT num_strings;
	
	/** word degree */
	CArray<INT> word_degree;
	/** cum num words */
	CArray<INT> cum_num_words;
	/** cum num words array */
	INT * cum_num_words_array;
	/** num words */
	CArray<INT> num_words;
	/** num words array */
	INT * num_words_array;
	/** mod words */
	CArray2<INT> mod_words;
	/** mod words array */
	INT * mod_words_array;
	/** sign words */
	CArray<bool> sign_words;
	/** sign words array */
	bool * sign_words_array;
	/** string words */
	CArray<INT> string_words;
	/** string words array */
	INT * string_words_array;

//	CArray3<INT> word_used ;
//	INT *word_used_array ;
//	CArray2<DREAL> svm_values_unnormalized ;
	/** SVM start position */
	CArray<INT> svm_pos_start;
	/** number of unique words */
	CArray<INT> num_unique_words;
	/** SVM arrays clean */
	bool svm_arrays_clean;

	/** number of SVMs single */
	INT num_svms_single;
	/** word degree single */
	INT word_degree_single;
	/** cum num words single */
	INT cum_num_words_single;
	/** num words single */
	INT num_words_single;

	/** word used single */
	CArray<bool> word_used_single;
	/** SVM value unnormalised single */
	CArray<DREAL> svm_value_unnormalized_single;
	/** number of unique words single */
	INT num_unique_words_single;

	/** max a id */
	INT max_a_id;
	
	// control info
	/** m step */
	INT m_step;
	/** m call */
	INT m_call;

	// input arguments
	/** m sequence */
	CArray3<DREAL> m_seq;
	/** m position */
	CArray<INT> m_pos;
	/** m orf info */
	CArray2<INT> m_orf_info;
	/** m segment sum weights */
	CArray2<DREAL> m_segment_sum_weights;
	/** m Plif list */
	CArray<CPlifBase*> m_plif_list;
	/** m PEN */
	CArray2<CPlifBase*> m_PEN;
	/** m PEN state signals */
	CArray2<CPlifBase*> m_PEN_state_signals;
	/** m genestr */
	CArray2<CHAR> m_genestr;
	/** m dict weights */
	CArray2<DREAL> m_dict_weights;
	/** m segment loss */
	CArray3<DREAL> m_segment_loss;
	/** m segment IDs */
	CArray<INT> m_segment_ids;
	/** m segment mask */
	CArray<DREAL> m_segment_mask;
	/** m my state seq */
	CArray<INT> m_my_state_seq;
	/** m my position sequence */
	CArray<INT> m_my_pos_seq;
	/** m my scores */
	CArray<DREAL> m_my_scores;
	/** m my losses */
	CArray<DREAL> m_my_losses;

	// output arguments
	/** m scores */
	CArray<DREAL> m_scores;
	/** m states */
	CArray2<INT> m_states;
	/** m positions */
	CArray2<INT> m_positions;

	/** storeage of stop codons
	 *  array of size length(sequence)
	 */
	CArray<bool> m_genestr_stop;

	/**
	 *  array for storage of precomputed linear features linge content svm values or pliffed tiling data
	 * Jonas
	 */
	CArray2<DREAL> m_lin_feat;
	/**number of  linear features*/ 
	INT m_num_lin_feat;


	/** raw intensities */
	DREAL* m_raw_intensities;
	/** prope position */
	INT* m_probe_pos;
	/** number of probes */
	INT* m_num_probes_cum;
	/** num lin feat plifs cum */
	INT* m_num_lin_feat_plifs_cum;
	/** number of additional data tracks like tiling, RNA-Seq, ...*/
	INT m_num_raw_data;
	/** length of gene string */
	INT m_genestr_len;
};

inline INT CDynProg::raw_intensities_interval_query(const INT from_pos, const INT to_pos, DREAL* intensities, INT type)
{
	ASSERT(from_pos<to_pos);
	//SG_PRINT("m_num_probes:%i, m_raw_intensities[1]:%f, m_probe_pos[1]:%i \n",m_num_probes, m_raw_intensities[10], m_probe_pos[10]);
	INT num_intensities = 0;
	INT* p_tiling_pos  = &m_probe_pos[m_num_probes_cum[type-1]];
	DREAL* p_tiling_data = &m_raw_intensities[m_num_probes_cum[type-1]];
	INT last_pos;
	INT num = m_num_probes_cum[type-1];
	while (*p_tiling_pos<to_pos)
	{
		if (*p_tiling_pos>=from_pos)
		{
			intensities[num_intensities] = *p_tiling_data;
			num_intensities++;
			//SG_PRINT("*p_tiling_data:%f, *p_tiling_pos:%i\n",*p_tiling_data,*p_tiling_pos);
		}
		num++;
		if (num>=m_num_probes_cum[type])
			break;
		last_pos = *p_tiling_pos;
		p_tiling_pos++;
		p_tiling_data++;
		SG_PRINT("num:%i, m_num_probes_cum[%i]:%i\n", num, type-1, m_num_probes_cum[type-1]);
		SG_PRINT("last_pos:%i, tiling_pos:%i\n", last_pos, *p_tiling_pos);
		ASSERT(last_pos<*p_tiling_pos);
	}
	return num_intensities;
}
inline void CDynProg::lookup_content_svm_values(const INT from_state, const INT to_state, const INT from_pos, const INT to_pos, DREAL* svm_values, INT frame)
{
//	ASSERT(from_state<to_state);
//	if (!(from_pos<to_pos))
//		SG_ERROR("from_pos!<to_pos, from_pos: %i to_pos: %i \n",from_pos,to_pos);
	for (INT i=0;i<num_svms;i++)
	{
		DREAL to_val   = m_lin_feat.get_element(i,  to_state);
		DREAL from_val = m_lin_feat.get_element(i,from_state);
		svm_values[i]=(to_val-from_val)/(to_pos-from_pos);
	}
	for (INT i=num_svms;i<m_num_lin_feat;i++)
	{
		DREAL to_val   = m_lin_feat.get_element(i,  to_state);
		DREAL from_val = m_lin_feat.get_element(i,from_state);
		svm_values[i]=to_val-from_val;
	}
	// find the correct row with precomputed 
	if (frame!=-1)
	{
		svm_values[4] = 1e10;
		svm_values[5] = 1e10;
		svm_values[6] = 1e10;
		INT global_frame = from_pos%3;
        	INT row = ((global_frame+frame)%3)+4;
		//SG_PRINT("global_frame:%i row:%i frame:%i \n", global_frame, row, frame);
		DREAL to_val   = m_lin_feat.get_element(row,  to_state);
		DREAL from_val = m_lin_feat.get_element(row,from_state);
		svm_values[frame+4] = (to_val-from_val)/(to_pos-from_pos);
	}
}
#endif
