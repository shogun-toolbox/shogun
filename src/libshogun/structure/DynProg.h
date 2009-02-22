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
typedef uint16_t T_STATES ;
#else
typedef uint8_t T_STATES ;
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
	float64_t **trans_list_forward_val;
	int32_t **trans_list_forward_id;
	bool mem_initialized;

#ifdef DYNPROG_TIMING
	CTime MyTime;
	CTime MyTime2;
	
	float64_t segment_init_time;
	float64_t segment_pos_time;
	float64_t segment_clean_time;
	float64_t segment_extend_time;
	float64_t orf_time;
	float64_t content_time;
	float64_t content_penalty_time;
	float64_t svm_init_time;
	float64_t svm_pos_time;
	float64_t svm_clean_time;
#endif
	
public:
	/** constructor
	 *
	 * @param p_num_svms number of SVMs
	 */
	CDynProg(int32_t p_num_svms=8);
	~CDynProg();

	/** best path no b
	 *
	 * @param max_iter max iter
	 * @param best_iter best iter
	 * @param my_path my path
	 *
	 * @return best path no b
	 */
	float64_t best_path_no_b(int32_t max_iter, int32_t & best_iter, int32_t *my_path);

	/** best path no b transition
	 *
	 * @param max_iter max iter
	 * @param max_best_iter max best iter
	 * @param nbest nbest
	 * @param prob_nbest prob_nbest
	 * @param my_paths my paths
	 */
	void best_path_no_b_trans(int32_t max_iter, int32_t & max_best_iter, int16_t nbest, float64_t *prob_nbest, int32_t *my_paths);
	
	// model related functions
	/** set number of states
	 * use this to set N first
	 *
	 * @param p_N new N
	 */
	void set_num_states(int32_t p_N);

	/** get num states */
	int32_t get_num_states();

	/** get num svms*/
	int32_t get_num_svms();

	/** init CArray for precomputed content svm values
	 *  with size seq_len x num_svms
	 *
	 *  @param p_num_svms: number of svm weight vectors for content prediction
	 *  @param seq_len: number of candidate positions
	 */
	void init_content_svm_value_array(const int32_t p_num_svms, const int32_t seq_len);

	/** init CArray for precomputed tiling intensitie-plif-values
	 *  with size seq_len x num_svms
	 *
	 *  @param probe_pos local positions of probes
	 *  @param intensities intensities of probes
	 *  @param num_probes number of probes
	 *  @param seq_len: number of candidate positions
	 */
	void init_tiling_data(int32_t* probe_pos, float64_t* intensities, const int32_t num_probes, const int32_t seq_len);

	/** precompute tiling Plifs
	 *
	 * @param PEN Plif PEN
	 * @param tiling_plif_ids tiling plif id's
	 * @param num_tiling_plifs number of tiling plifs
	 * @param seq_len sequence length
	 * @param pos pos
	 */
	void precompute_tiling_plifs(CPlif** PEN, const int32_t* tiling_plif_ids, const int32_t num_tiling_plifs, const int32_t seq_len, const int32_t* pos);	

	/** append rows to linear features array
 	 * 
 	 * @param num_new_feat number of new rows to add
 	 * @param seq_len number of columns == number of candidate positions
 	 * 			(must be equal to the existing num of cols) 
 	 */
	void resize_lin_feat(int32_t num_new_feat, int32_t seq_len);
	/** set vector p
	 *
	 * @param p new vector p
	 * @param N size of vector p
	 */
	void set_p_vector(float64_t* p, int32_t N);

	/** set vector q
	 *
	 * @param q new vector q
	 * @param N size of vector q
	 */
	void set_q_vector(float64_t* q, int32_t N);
	
	/** set matrix a
	 *
	 * @param a new matrix a
	 * @param M dimension M of matrix a
	 * @param N dimension N of matrix a
	 */
	void set_a(float64_t* a, int32_t M, int32_t N);
	
	/** set a id
	 *
	 * @param a new a id (identity?)
	 * @param M dimension M of matrix a
	 * @param N dimension N of matrix a
	 */
	void set_a_id(int32_t *a, int32_t M, int32_t N);
	
	/** set a transition matrix
	 *
	 * @param a_trans transition matrix a
	 * @param num_trans number of transitions
	 * @param N dimension N of matrix a
	 */
	void set_a_trans_matrix(float64_t *a_trans, int32_t num_trans, int32_t N);

	// content svm related setup functions
	/** init SVM arrays
	 *
	 * @param p_num_degrees number of degrees
	 * @param p_num_svms number of SVMs
	 */
	void init_svm_arrays(int32_t p_num_degrees, int32_t p_num_svms);

	/** init word degree array
	 *
	 * @param p_word_degree_array new word degree array
	 * @param num_elem number of array elements
	 */
	void init_word_degree_array(int32_t * p_word_degree_array, int32_t num_elem);

	/** init cum num words array
	 *
	 * @param p_cum_num_words_array new cum num words array
	 * @param num_elem number of array elements
	 */
	void init_cum_num_words_array(int32_t * p_cum_num_words_array, int32_t num_elem);

	/** init num words array
	 *
	 * @param p_num_words_array new num words array
	 * @param num_elem number of array elements
	 */
	void init_num_words_array(int32_t * p_num_words_array, int32_t num_elem);

	/** init mod words array
	 *
	 * @param p_mod_words_array new mod words array
	 * @param num_elem number of array elements
	 * @param num_columns number of columns
	 */
	void init_mod_words_array(int32_t * p_mod_words_array, int32_t num_elem, int32_t num_columns);

	/** init sign words array
	 *
	 * @param p_sign_words_array new sign words array
	 * @param num_elem number of array elements
	 */
	void init_sign_words_array(bool * p_sign_words_array, int32_t num_elem);

	/** init string words array
	 *
	 * @param p_string_words_array new string words array
	 * @param num_elem number of array elements
	 */
	void init_string_words_array(int32_t * p_string_words_array, int32_t num_elem);

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
	void best_path_set_seq(float64_t *seq, int32_t N, int32_t seq_len);

	/** set best path seq3d
	 *
	 * @param seq the 3D sequence
	 * @param p_N dimension N
	 * @param seq_len length of sequence
	 * @param max_num_signals maximal number of signals
	 */
	void best_path_set_seq3d(float64_t *seq, int32_t p_N, int32_t seq_len, int32_t max_num_signals);

	/** set best path pos
	 *
	 * @param pos the position
	 * @param seq_len length of sequence
	 */
	void best_path_set_pos(int32_t *pos, int32_t seq_len);

	/** set best path orf info
	 * only for best_path_trans
	 *
	 * @param orf_info the orf info
	 * @param m dimension m
	 * @param n dimension n
	 */
	void best_path_set_orf_info(int32_t *orf_info, int32_t m, int32_t n);

	/** set best path segment sum weights
	 * only for best_path_2struct
	 *
	 * @param segment_sum_weights segment sum weights
	 * @param num_states number of states
	 * @param seq_len length of sequence
	 */
	void best_path_set_segment_sum_weights(float64_t *segment_sum_weights, int32_t num_states, int32_t seq_len);

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
	void best_path_set_plif_id_matrix(int32_t *plif_id_matrix, int32_t m, int32_t n);

	/** set best path plif state signal matrix
	 *
	 * @param plif_id_matrix plif id matrix
	 * @param m dimension m of matrix
	 * @param n dimension n of matrix
	 */
	void best_path_set_plif_state_signal_matrix(int32_t *plif_id_matrix, int32_t m, int32_t n);

	/** set best path genesstr
	 *
	 * @param genestr gene string
	 * @param genestr_len length of gene string
	 * @param genestr_num number of gene strings, typically 1
	 */
	void best_path_set_genestr(char* genestr, int32_t genestr_len, int32_t genestr_num);

	// additional best_path_trans_deriv functions
	/** set best path my state sequence
	 *
	 * @param my_state_seq my state sequence
	 * @param seq_len length of sequence
	 */
	void best_path_set_my_state_seq(int32_t* my_state_seq, int32_t seq_len);

	/** set best path my position sequence
	 *
	 * @param my_pos_seq my position sequence
	 * @param seq_len length of sequence
	 */
	void best_path_set_my_pos_seq(int32_t* my_pos_seq, int32_t seq_len);

	/** set best path single gene string
	 *
	 * @param genestr gene string
	 * @param genestr_len length of gene string
	 */
	inline void best_path_set_single_genestr(char* genestr, int32_t genestr_len)
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
	void best_path_set_dict_weights(float64_t* dictionary_weights, int32_t dict_len, int32_t n);

	/** set best path segment loss
	 *
	 * @param segment_loss segment loss
	 * @param num_segment_id1 number of segment id1
	 * @param num_segment_id2 number of segment id2
	 */
	void best_path_set_segment_loss(float64_t * segment_loss, int32_t num_segment_id1, int32_t num_segment_id2);

	/** set best path segmend ids mask
	 *
	 * @param segment_ids segment ids
	 * @param segment_mask segment mask
	 * @param m dimension m
	 */
	void best_path_set_segment_ids_mask(int32_t* segment_ids, float64_t* segment_mask, int32_t m);

	// best_path functions
	/** best path call
	 *
	 * @param nbest nbest
	 * @param use_orf whether to use orf
	 */
	void best_path_call(int32_t nbest, bool use_orf);

	/** best path derivative call */
	void best_path_deriv_call();

	/** best path 2struct call
	 *
	 * @param nbest nbest
	 */
	void best_path_2struct_call(int32_t nbest);

	/** best path simple call
	 *
	 * @param nbest nbest
	 */
	void best_path_simple_call(int32_t nbest);

	/** best path derivative call
	 *
	 * @param nbest nbest
	 */
	void best_path_deriv_call(int32_t nbest);
	
	// best_path result retrieval functions
	/** best path get scores
	 *
	 * @param scores scores
	 * @param n dimension n
	 */
	void best_path_get_scores(float64_t **scores, int32_t *n);

	/** best path get states
	 *
	 * @param states states
	 * @param m dimension m
	 * @param n dimension n
	 */
	void best_path_get_states(int32_t **states, int32_t *m, int32_t *n);

	/** best path get positions
	 *
	 * @param positions positions
	 * @param m dimension m
	 * @param n dimension n
	 */
	void best_path_get_positions(int32_t **positions, int32_t *m, int32_t *n);

	//best_path_trans_deriv result retrieval functions
	/** get best path losses
	 *
	 * @param my_losses my losses
	 * @param seq_len length of sequence
	 */
	void best_path_get_losses(float64_t** my_losses, int32_t* seq_len);

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
	template <int16_t nbest, bool with_loss, bool with_multiple_sequences>
	void best_path_trans(const float64_t *seq, int32_t seq_len, const int32_t *pos,
						 const int32_t *orf_info, CPlifBase **PLif_matrix,
						 CPlifBase **Plif_state_signals, int32_t max_num_signals,
						 int32_t genestr_num,
						 float64_t *prob_nbest, int32_t *my_state_seq, int32_t *my_pos_seq,
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
	void best_path_trans_deriv(int32_t *my_state_seq, int32_t *my_pos_seq, float64_t *my_scores, float64_t* my_losses, int32_t my_seq_len,
					const float64_t *seq_array, int32_t seq_len, const int32_t *pos, CPlifBase **Plif_matrix,
					CPlifBase **Plif_state_signals, int32_t max_num_signals, int32_t genestr_num);
	
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
	void best_path_2struct(const float64_t *seq, int32_t seq_len, const int32_t *pos,
						   CPlifBase **Plif_matrix,
						   const char *genestr, int32_t genestr_len,
						   int16_t nbest,
						   float64_t *prob_nbest, int32_t *my_state_seq, int32_t *my_pos_seq,
						   float64_t *dictionary_weights, int32_t dict_len, float64_t *segment_sum_weights);

	/** best path trans simple
	 *
	 * @param seq sequence
	 * @param seq_len length of sequence
	 * @param nbest nbest
	 * @param prob_nbest prob(ability?) nbest
	 * @param my_state_seq my state seq
	 */
	void best_path_trans_simple(const float64_t *seq, int32_t seq_len, int16_t nbest,
								float64_t *prob_nbest, int32_t *my_state_seq);



	/// access function for number of states N
	inline T_STATES get_N() const
	  {
	    return N ;
	  }
	
	/** access function for probability of end states
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_q(T_STATES offset, float64_t value)
	{
		end_state_distribution_q[offset]=value;
	}

	/** access function for probability of first state
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_p(T_STATES offset, float64_t value)
	{
		initial_state_distribution_p[offset]=value;
	}

	/** access function for matrix a
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @param value value to be set
	 */
	inline void set_a(T_STATES line_, T_STATES column, float64_t value)
	{
	  transition_matrix_a.element(line_,column)=value; // look also best_path!
	}

	/** access function for probability of end states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_q(T_STATES offset) const
	{
		return end_state_distribution_q[offset];
	}

	/** access function for derivated probability of end states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_q_deriv(T_STATES offset) const
	{
		return end_state_distribution_q_deriv[offset];
	}

	/** access function for probability of initial states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_p(T_STATES offset) const
	{
		return initial_state_distribution_p[offset];
	}

	/** access function for derivated probability of initial states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_p_deriv(T_STATES offset) const
	{
		return initial_state_distribution_p_deriv[offset];
	}
	
	/** create array of precomputed content svm values
	 * Jonas
	 *
	 * @param wordstr word strings
	 * @param pos position
	 * @param num_cand_pos number of cand position
	 * @param genestr_len length of DNA-sequence
	 * @param dictionary_weights SVM weight vectors for content prediction
	 * @param dict_len number of weight vectors 
	 */
	void precompute_content_values(uint16_t*** wordstr, const int32_t *pos,
		const int32_t num_cand_pos, const int32_t genestr_len,
		float64_t *dictionary_weights, int32_t dict_len);


	/** return array of precomputed linear features like content predictions
	 *  and PLiFed tiling array data
	 * Jonas
	 *
	 * @return lin_feat_array
	 */
	inline float64_t* get_lin_feat(int32_t & dim1, int32_t & dim2) 
	{
		m_lin_feat.get_array_size(dim1, dim2);
		return m_lin_feat.get_array();
	}
	/** return array of precomputed linear features like content predictions
	 *  and PLiFed tiling array data
	 * Jonas
	 *
	 * @return lin_feat_array
	 */
	inline void set_lin_feat(float64_t* p_lin_feat, int32_t p_num_svms, int32_t p_seq_len) 
	{
		m_lin_feat.set_array(p_lin_feat, p_num_svms, p_seq_len, true);
	}
	/** create word string from char*
	 * Jonas
	 *
	 * @param genestr gene string
	 * @param genestr_num number of gene string
	 * @param genestr_len length of gene string
	 * @param wordstr word strings
	 */
	void create_word_string(const char* genestr, int32_t genestr_num, int32_t genestr_len, uint16_t*** wordstr);

	/** precompute stop codons
	 *
	 * @param genestr gene string
	 * @param genestr_len length of gene string
	 */
	void precompute_stop_codons(const char* genestr, int32_t genestr_len);

	/** set genestr len
	 *
	 * @param genestr_len length of gene string
	 *
	 */
	void set_genestr_len(int32_t genestr_len);

	/** access function for matrix a
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline float64_t get_a(T_STATES line_, T_STATES column) const
	{
	  return transition_matrix_a.element(line_,column); // look also best_path()!
	}

	/** access function for matrix a derivated
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline float64_t get_a_deriv(T_STATES line_, T_STATES column) const
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
	inline void lookup_content_svm_values(const int32_t from_state,
		const int32_t to_state, const int32_t from_pos, const int32_t to_pos,
		float64_t* svm_values, int32_t frame);

	/** lookup tiling Plif values
	 *
	 * @param from_state from state
	 * @param to_state to state
	 * @param len length
	 * @param svm_values SVM values
	 */
	inline void lookup_tiling_plif_values(const int32_t from_state,
		const int32_t to_state, const int32_t len, float64_t* svm_values);

	/** find frame
	 *
	 * @param from_state from state
	 */
	inline int32_t find_frame(const int32_t from_state);

	/** raw intensities interval query
	 *
	 * @param from_pos from position
	 * @param to_pos to position
	 * @param intensities intensities
	 * @param type type
	 * @return an integer
	 */
	inline int32_t raw_intensities_interval_query(
		const int32_t from_pos, const int32_t to_pos, float64_t* intensities, int32_t type);

	/** translate from single order
	 *
	 * @param obs observation matrix
	 * @param sequence_length length of sequence
	 * @param start start
	 * @param order order
	 * @param max_val maximum number of bits, e.g. 2 for DNA
	 */
	void translate_from_single_order(uint16_t* obs, int32_t sequence_length, int32_t start,
		int32_t order, int32_t max_val=2);

	/** reset SVM value
	 *
	 * @param pos position
	 * @param last_svm_pos last SVM position
	 * @param svm_value value to set
	 */
	void reset_svm_value(int32_t pos, int32_t & last_svm_pos, float64_t * svm_value);

	/** extend SVM value
	 *
	 * @param wordstr word string
	 * @param pos position
	 * @param last_svm_pos lsat SVM position
	 * @param svm_value value to set
	 */
	void extend_svm_value(uint16_t* wordstr, int32_t pos, int32_t &last_svm_pos,
		float64_t* svm_value);

	/** reset segment sum value
	 *
	 * @param num_states number of states
	 * @param pos position
	 * @param last_segment_sum_pos last segment sum position
	 * @param segment_sum_value value to set
	 */
	void reset_segment_sum_value(int32_t num_states, int32_t pos,
		int32_t & last_segment_sum_pos, float64_t * segment_sum_value);

	/** extend segment sum value
	 *
	 * @param segment_sum_weights segment sum weights
	 * @param seqlen length of sequence
	 * @param num_states number of states
	 * @param pos position
	 * @param last_segment_sum_pos last segment sum position
	 * @param segment_sum_value value to set
	 */
	void extend_segment_sum_value(float64_t *segment_sum_weights, int32_t seqlen,
		int32_t num_states, int32_t pos, int32_t &last_segment_sum_pos,
		float64_t* segment_sum_value);

	/** SVM values */
	struct svm_values_struct
	{
		/** maximum lookback */
		int32_t maxlookback;
		/** sequence length */
		int32_t seqlen;

		/** start position */
		int32_t* start_pos;
		/** SVM values normalized */
		float64_t ** svm_values_unnormalized;
		/** SVM values */
		float64_t * svm_values;
		/** word used */
		bool *** word_used;
		/** number of unique words */
		int32_t **num_unique_words;
	};

	//void reset_svm_values(int32_t pos, int32_t * last_svm_pos, float64_t * svm_value) ;
	//void extend_svm_values(uint16_t** wordstr, int32_t pos, int32_t *last_svm_pos, float64_t* svm_value) ;
	/** init SVM values
	 *
	 * @param svs SVM values
	 * @param start_pos start position
	 * @param seqlen length of sequence
	 * @param howmuchlookback how far to look back
	 */
	void init_svm_values(struct svm_values_struct & svs, int32_t start_pos,
		int32_t seqlen, int32_t howmuchlookback);

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
	void find_svm_values_till_pos(uint16_t*** wordstr, const int32_t *pos, int32_t t_end,
		struct svm_values_struct &svs);

	/** find SVM values till position
	 *
	 * @param wordstr word string
	 * @param pos position
	 * @param t_end t end
	 * @param svs SVM values
	 */
	void find_svm_values_till_pos(uint16_t** wordstr, const int32_t *pos, int32_t t_end,
		struct svm_values_struct &svs);

	/** update SVM values till position
	 *
	 * @param wordstr word string
	 * @param pos position
	 * @param t_end t end
	 * @param prev_t_end previous t end
	 * @param svs SVM values
	 */
	void update_svm_values_till_pos(uint16_t*** wordstr, const int32_t *pos, int32_t t_end,
		int32_t prev_t_end, struct svm_values_struct &svs);

	/** extend orf
	 *
	 * @param orf_from orf from
	 * @param orf_to orf to
	 * @param start start
	 * @param last_pos last position
	 * @param to to
	 */
	bool extend_orf(int32_t orf_from, int32_t orf_to, int32_t start, int32_t &last_pos, int32_t to);

	/** segment loss */
	struct segment_loss_struct
	{
		/** maximum lookback */
		int32_t maxlookback;
		/** sequence length */
		int32_t seqlen;
		/** segments changed */
		int32_t *segments_changed;
		/** numb segment ID */
		float64_t *num_segment_id;
		/** length of segmend ID */
		int32_t *length_segment_id ;
	};

	/** init segment loss
	 *
	 * @param loss segment loss to init
	 * @param seqlen length of sequence
	 * @param howmuchlookback how far to look back
	 */
	void init_segment_loss(struct segment_loss_struct & loss, int32_t seqlen,
		int32_t howmuchlookback);

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
	float64_t extend_segment_loss(struct segment_loss_struct & loss,
		const int32_t * pos_array, int32_t segment_id, int32_t pos, int32_t& last_pos,
		float64_t &last_value);

	/** find segment loss till pos
	 *
	 * @param pos position
	 * @param t_end t end
	 * @param segment_ids segment IDs
	 * @param segment_mask segmend mask
	 * @param loss segment loss
	 */
	void find_segment_loss_till_pos(const int32_t * pos, int32_t t_end,
		CArray<int32_t>& segment_ids, CArray<float64_t>& segment_mask,
		struct segment_loss_struct& loss);

		/** @return object name */
		inline virtual const char* get_name() const { return "DynProg"; }
	
protected:
	/**@name model specific variables.
	 * these are p,q,a,b,N,M etc
	 */
	//@{
	/// number of states
	int32_t N;

	/// transition matrix
	CArray2<int32_t> transition_matrix_a_id;
	CArray2<float64_t> transition_matrix_a;
	CArray2<float64_t> transition_matrix_a_deriv;

	/// initial distribution of states
	CArray<float64_t> initial_state_distribution_p;
	CArray<float64_t> initial_state_distribution_p_deriv;

	/// distribution of end-states
	CArray<float64_t> end_state_distribution_q;
	CArray<float64_t> end_state_distribution_q_deriv;

	//@}
	
	/** dict weights */
	CArray2<float64_t> dict_weights;
	/** dict weights array */
	float64_t * dict_weights_array;

	/** number of degress */
	int32_t num_degrees;
	/** number of SVMs */
	int32_t num_svms;
	/** number of strings */
	int32_t num_strings;
	
	/** word degree */
	CArray<int32_t> word_degree;
	/** cum num words */
	CArray<int32_t> cum_num_words;
	/** cum num words array */
	int32_t * cum_num_words_array;
	/** num words */
	CArray<int32_t> num_words;
	/** num words array */
	int32_t * num_words_array;
	/** mod words */
	CArray2<int32_t> mod_words;
	/** mod words array */
	int32_t * mod_words_array;
	/** sign words */
	CArray<bool> sign_words;
	/** sign words array */
	bool * sign_words_array;
	/** string words */
	CArray<int32_t> string_words;
	/** string words array */
	int32_t * string_words_array;

//	CArray3<int32_t> word_used ;
//	int32_t *word_used_array ;
//	CArray2<float64_t> svm_values_unnormalized ;
	/** SVM start position */
	CArray<int32_t> svm_pos_start;
	/** number of unique words */
	CArray<int32_t> num_unique_words;
	/** SVM arrays clean */
	bool svm_arrays_clean;

	/** number of SVMs single */
	int32_t num_svms_single;
	/** word degree single */
	int32_t word_degree_single;
	/** cum num words single */
	int32_t cum_num_words_single;
	/** num words single */
	int32_t num_words_single;

	/** word used single */
	CArray<bool> word_used_single;
	/** SVM value unnormalised single */
	CArray<float64_t> svm_value_unnormalized_single;
	/** number of unique words single */
	int32_t num_unique_words_single;

	/** max a id */
	int32_t max_a_id;
	
	// control info
	/** m step */
	int32_t m_step;
	/** m call */
	int32_t m_call;

	// input arguments
	/** m sequence */
	CArray3<float64_t> m_seq;
	/** m position */
	CArray<int32_t> m_pos;
	/** m orf info */
	CArray2<int32_t> m_orf_info;
	/** m segment sum weights */
	CArray2<float64_t> m_segment_sum_weights;
	/** m Plif list */
	CArray<CPlifBase*> m_plif_list;
	/** m PEN */
	CArray2<CPlifBase*> m_PEN;
	/** m PEN state signals */
	CArray2<CPlifBase*> m_PEN_state_signals;
	/** m genestr */
	CArray2<char> m_genestr;
	/** m dict weights */
	CArray2<float64_t> m_dict_weights;
	/** m segment loss */
	CArray3<float64_t> m_segment_loss;
	/** m segment IDs */
	CArray<int32_t> m_segment_ids;
	/** m segment mask */
	CArray<float64_t> m_segment_mask;
	/** m my state seq */
	CArray<int32_t> m_my_state_seq;
	/** m my position sequence */
	CArray<int32_t> m_my_pos_seq;
	/** m my scores */
	CArray<float64_t> m_my_scores;
	/** m my losses */
	CArray<float64_t> m_my_losses;

	// output arguments
	/** m scores */
	CArray<float64_t> m_scores;
	/** m states */
	CArray2<int32_t> m_states;
	/** m positions */
	CArray2<int32_t> m_positions;

	/** storeage of stop codons
	 *  array of size length(sequence)
	 */
	CArray<bool> m_genestr_stop;

	/**
	 *  array for storage of precomputed linear features linge content svm values or pliffed tiling data
	 * Jonas
	 */
	CArray2<float64_t> m_lin_feat;
	/**number of  linear features*/ 
	//int32_t m_num_lin_feat;


	/** raw intensities */
	float64_t* m_raw_intensities;
	/** prope position */
	int32_t* m_probe_pos;
	/** number of probes */
	int32_t* m_num_probes_cum;
	/** num lin feat plifs cum */
	int32_t* m_num_lin_feat_plifs_cum;
	/** number of additional data tracks like tiling, RNA-Seq, ...*/
	int32_t m_num_raw_data;
	/** length of gene string */
	int32_t m_genestr_len;
};

inline int32_t CDynProg::raw_intensities_interval_query(const int32_t from_pos, const int32_t to_pos, float64_t* intensities, int32_t type)
{
	ASSERT(from_pos<to_pos);
	int32_t num_intensities = 0;
	int32_t* p_tiling_pos  = &m_probe_pos[m_num_probes_cum[type-1]];
	float64_t* p_tiling_data = &m_raw_intensities[m_num_probes_cum[type-1]];
	int32_t last_pos;
	int32_t num = m_num_probes_cum[type-1];
	while (*p_tiling_pos<to_pos)
	{
		if (*p_tiling_pos>=from_pos)
		{
			intensities[num_intensities] = *p_tiling_data;
			num_intensities++;
		}
		num++;
		if (num>=m_num_probes_cum[type])
			break;
		last_pos = *p_tiling_pos;
		p_tiling_pos++;
		p_tiling_data++;
		ASSERT(last_pos<*p_tiling_pos);
	}
	return num_intensities;
}
inline void CDynProg::lookup_content_svm_values(const int32_t from_state, const int32_t to_state, const int32_t from_pos, const int32_t to_pos, float64_t* svm_values, int32_t frame)
{
//	ASSERT(from_state<to_state);
//	if (!(from_pos<to_pos))
//		SG_ERROR("from_pos!<to_pos, from_pos: %i to_pos: %i \n",from_pos,to_pos);
	for (int32_t i=0;i<num_svms;i++)
	{
		float64_t to_val   = m_lin_feat.get_element(i,  to_state);
		float64_t from_val = m_lin_feat.get_element(i,from_state);
		svm_values[i]=(to_val-from_val)/(to_pos-from_pos);
	}
	for (int32_t i=num_svms;i<m_num_lin_feat_plifs_cum[m_num_raw_data];i++)
	{
		float64_t to_val   = m_lin_feat.get_element(i,  to_state);
		float64_t from_val = m_lin_feat.get_element(i,from_state);
		svm_values[i]=to_val-from_val;
	}
	// find the correct row with precomputed 
	if (frame!=-1)
	{
		svm_values[4] = 1e10;
		svm_values[5] = 1e10;
		svm_values[6] = 1e10;
		int32_t global_frame = from_pos%3;
        	int32_t row = ((global_frame+frame)%3)+4;
		float64_t to_val   = m_lin_feat.get_element(row,  to_state);
		float64_t from_val = m_lin_feat.get_element(row,from_state);
		svm_values[frame+4] = (to_val-from_val)/(to_pos-from_pos);
	}
}
#endif
