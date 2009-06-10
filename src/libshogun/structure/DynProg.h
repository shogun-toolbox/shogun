/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Gunnar Raetsch
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Written (W) 2008-2009 Jonas Behr
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
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
#include "features/SparseFeatures.h"
#include "distributions/Distribution.h"
#include "lib/DynamicArray.h"
#include "lib/Array.h"
#include "lib/Array2.h"
#include "lib/Array3.h"
#include "lib/Time.h"

#include <stdio.h>
#include <limits.h>

//#define DYNPROG_TIMING

#ifdef USE_BIGSTATES
typedef uint16_t T_STATES ;
#else
typedef uint8_t T_STATES ;
#endif
typedef T_STATES* P_STATES ;

/** @brief Dynamic Programming Class.
 *
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
	CTime MyTime3;
	
	float64_t segment_init_time;
	float64_t segment_pos_time;
	float64_t segment_clean_time;
	float64_t segment_extend_time;
	float64_t orf_time;
	float64_t content_time;
	float64_t content_penalty_time;
	float64_t content_svm_values_time ;
	float64_t content_plifs_time ;	
	float64_t svm_init_time;
	float64_t svm_pos_time;
	float64_t inner_loop_time;
	float64_t inner_loop_max_time ;	
	float64_t svm_clean_time;
	float64_t long_transition_time ;
#endif
	
public:
	/** constructor
	 *
	 * @param p_num_svms number of SVMs
	 */
	CDynProg(int32_t p_num_svms=8);
	virtual ~CDynProg();

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
	 * @param N new N
	 */
	void set_num_states(int32_t N);

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

	/** set best path seq
	 *
	 * @param seq the sequence
	 * @param N dimension N
	 * @param seq_len length of sequence
	 */
	void set_seq(float64_t* seq, int32_t N, int32_t seq_len);

	/** set best path seq3d
	 *
	 * @param seq the 3D sequence
	 * @param N dimension N
	 * @param seq_len length of sequence
	 * @param max_num_signals maximal number of signals
	 */
	void best_path_set_seq3d(float64_t *seq, int32_t N, int32_t seq_len, int32_t max_num_signals);

	/** set best path pos
	 *
	 * @param pos the position
	 * @param seq_len length of sequence
	 */
	void set_pos(int32_t* pos, int32_t seq_len);

	/** set best path orf info
	 * only for best_path_trans
	 *
	 * @param orf_info the orf info
	 * @param m dimension m
	 * @param n dimension n
	 */
	void set_orf_info(int32_t* orf_info, int32_t m, int32_t n);

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
	 */
	void set_gene_string(char* genestr, int32_t genestr_len);

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

	/** set sparse feature matrices */
	void set_sparse_features(CSparseFeatures<float64_t>* seq_sparse1, CSparseFeatures<float64_t>* seq_sparse2);

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
	 * @param PLif_matrix Plif matrix
	 * @param Plif_state_signals Plif state signals
	 * @param max_num_signals maximal number of signals
	 * @param prob_nbest prob nbest
	 * @param my_state_seq my state seq
	 * @param my_pos_seq my pos seq
	 * @param use_orf whether orf shall be used
	 */
	template <int16_t nbest, bool with_loss, bool with_multiple_sequences>
	void best_path_trans(CPlifBase **PLif_matrix,
						 CPlifBase **Plif_state_signals, int32_t max_num_signals,
						 float64_t *prob_nbest, int32_t *my_state_seq, int32_t *my_pos_seq,
						 bool use_orf)
	{
		const float64_t* seq_array = m_seq.get_array();
		const int32_t seq_len = m_seq.get_dim2();

#ifdef DYNPROG_TIMING
		segment_init_time = 0.0 ;
		segment_pos_time = 0.0 ;
		segment_extend_time = 0.0 ;
		segment_clean_time = 0.0 ;
		orf_time = 0.0 ;
		svm_init_time = 0.0 ;
		svm_pos_time = 0.0 ;
		svm_clean_time = 0.0 ;
		inner_loop_time = 0.0 ;
		content_svm_values_time = 0.0 ;
		content_plifs_time = 0.0 ;
		inner_loop_max_time = 0.0 ;
		long_transition_time = 0.0 ;

		MyTime2.start() ;
#endif

		if (!m_svm_arrays_clean)
		{
			SG_ERROR( "SVM arrays not clean") ;
			return ;
		}

#ifdef DYNPROG_DEBUG
		m_transition_matrix_a.set_name("transition_matrix");
		m_transition_matrix_a.display_array();
		m_mod_words.display_array() ;
		m_sign_words.display_array() ;
		m_string_words.display_array() ;
		//SG_PRINT("use_orf = %i\n", use_orf) ;
#endif

		int32_t max_look_back = 1000 ;
		bool use_svm = false ;

		SG_DEBUG("m_N:%i, seq_len:%i, max_num_signals:%i\n",m_N, seq_len, max_num_signals) ;

		//	for (int32_t i=0;i<m_N*seq_len*max_num_signals;i++)
		//		SG_PRINT("(%i)%0.2f ",i,seq_array[i]);

		//CArray2<CPlifBase*> PEN(PLif_matrix, m_N, m_N, false, false) ;
		CArray2<CPlifBase*> PEN(PLif_matrix, m_N, m_N, false, true) ;
		PEN.set_name("PEN");
		//CArray2<CPlifBase*> PEN_state_signals(Plif_state_signals, m_N, max_num_signals, false, false) ;
		CArray2<CPlifBase*> PEN_state_signals(Plif_state_signals, m_N, max_num_signals, false, true) ;
		PEN_state_signals.set_name("state_signals");

		CArray2<float64_t> seq(m_N, seq_len) ;
		seq.set_name("seq") ;
		seq.zero() ;

		float64_t svm_value[m_num_lin_feat_plifs_cum[m_num_raw_data]] ;
		{ // initialize svm_svalue
			for (int32_t s=0; s<m_num_lin_feat_plifs_cum[m_num_raw_data]; s++)
				svm_value[s]=0 ;
		}

		{ // convert seq_input to seq
			// this is independent of the svm values 

			//CArray3<float64_t> seq_input(seq_array, m_N, seq_len, max_num_signals) ;
			CArray3<float64_t> *seq_input=NULL ;
			if (seq_array!=NULL)
			{
				SG_PRINT("using dense seq_array\n") ;

				seq_input=new CArray3<float64_t>(seq_array, m_N, seq_len, max_num_signals) ;
				seq_input->set_name("seq_input") ;
				//seq_input.display_array() ;

				ASSERT(m_seq_sparse1==NULL) ;
				ASSERT(m_seq_sparse2==NULL) ;
			} else
			{
				SG_PRINT("using sparse seq_array\n") ;

				ASSERT(m_seq_sparse1!=NULL) ;
				ASSERT(m_seq_sparse2!=NULL) ;
				ASSERT(max_num_signals==2) ;
			}

			for (int32_t i=0; i<m_N; i++)
				for (int32_t j=0; j<seq_len; j++)
					seq.element(i,j) = 0 ;

			for (int32_t i=0; i<m_N; i++)
				for (int32_t j=0; j<seq_len; j++)
					for (int32_t k=0; k<max_num_signals; k++)
					{
						if ((PEN_state_signals.element(i,k)==NULL) && (k==0))
						{
							// no plif
							if (seq_input!=NULL)
								seq.element(i,j) = seq_input->element(i,j,k) ;
							else
							{
								if (k==0)
									seq.element(i,j) = m_seq_sparse1->get_element(i,j) ;
								if (k==1)
									seq.element(i,j) = m_seq_sparse2->get_element(i,j) ;
							}
							break ;
						}
						if (PEN_state_signals.element(i,k)!=NULL)
						{
							if (seq_input!=NULL)
							{
								// just one plif
								if (CMath::is_finite(seq_input->element(i,j,k)))
									seq.element(i,j) += PEN_state_signals.element(i,k)->lookup_penalty(seq_input->element(i,j,k), svm_value) ;
								else
									// keep infinity values
									seq.element(i,j) = seq_input->element(i, j, k) ;
							}
							else
							{
								if (k==0)
								{
									// just one plif
									if (CMath::is_finite(m_seq_sparse1->get_element(i,j)))
										seq.element(i,j) += PEN_state_signals.element(i,k)->lookup_penalty(m_seq_sparse1->get_element(i,j), svm_value) ;
									else
										// keep infinity values
										seq.element(i,j) = m_seq_sparse1->get_element(i, j) ;
								}
								if (k==1)
								{
									// just one plif
									if (CMath::is_finite(m_seq_sparse2->get_element(i,j)))
										seq.element(i,j) += PEN_state_signals.element(i,k)->lookup_penalty(m_seq_sparse2->get_element(i,j), svm_value) ;
									else
										// keep infinity values
										seq.element(i,j) = m_seq_sparse2->get_element(i, j) ;
								}
							}
						} 
						else
							break ;
					}
			delete seq_input ;
		}

		// allow longer transitions than look_back
		bool long_transitions = false; //m_long_transitions ;
		CArray2<int32_t> long_transition_content_position(m_N,m_N) ;
		CArray2<int32_t> long_transition_content_start(m_N,m_N) ;
		CArray2<float64_t> long_transition_content_scores(m_N,m_N) ;
		CArray2<float64_t> long_transition_content_scores_pen(m_N,m_N) ;
		CArray2<float64_t> long_transition_content_scores_prev(m_N,m_N) ;
		CArray2<float64_t> long_transition_content_scores_elem(m_N,m_N) ;

		if (with_loss || nbest!=1)
		{
			SG_DEBUG("disabling long transitions\n") ;
			long_transitions = false ;
		}
		long_transition_content_scores.set_const(-CMath::INFTY);
		long_transition_content_scores_pen.set_const(0) ;
		long_transition_content_scores_elem.set_const(0) ;
		long_transition_content_scores_prev.set_const(0) ;
		long_transition_content_start.zero() ;
		long_transition_content_position.zero() ;

		CArray2<int32_t> look_back(m_N,m_N) ;

		{ // determine maximal length of look-back
			for (int32_t i=0; i<m_N; i++)
				for (int32_t j=0; j<m_N; j++)
					look_back.set_element(INT_MAX, i, j) ;

			for (int32_t j=0; j<m_N; j++)
			{
				// only consider transitions that are actually allowed
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;

				for (int32_t i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;

					CPlifBase *penij=PEN.element(j, ii) ;
					if (penij==NULL)
					{
						if (long_transitions)
							look_back.set_element(m_long_transition_threshold, j, ii) ;
						continue ;
					}

					/* if the transition is an ORF or we do computation with loss, we have to disable long transitions */
					if ((m_orf_info.element(ii,0)!=-1) || (m_orf_info.element(j,1)!=-1) || (!long_transitions))
					{
						look_back.set_element(CMath::ceil(penij->get_max_value()), j, ii) ;
						if (CMath::ceil(penij->get_max_value()) > max_look_back)
						{
							SG_DEBUG( "%d %d -> value: %f\n", ii,j,penij->get_max_value());
							max_look_back=(int32_t) (CMath::ceil(penij->get_max_value()));
						}
					}
					else
						look_back.set_element(CMath::min( (int32_t)CMath::ceil(penij->get_max_value()), m_long_transition_threshold ), j, ii) ;

					if (penij->uses_svm_values())
						use_svm=true ;
				}
			}
			/* make sure max_look_back is at least as long as a long transition */
			if (long_transitions)
				max_look_back = CMath::max(m_long_transition_threshold, max_look_back) ;

			/* make sure max_look_back is not longer than the whole string */
			max_look_back = CMath::min(m_genestr.get_dim1(), max_look_back) ;

			int32_t num_long_transitions = 0 ;
			for (int32_t i=0; i<m_N; i++)
				for (int32_t j=0; j<m_N; j++)
				{
					if (look_back.get_element(i,j)==m_long_transition_threshold)
						num_long_transitions++ ;
					if (look_back.get_element(i,j)==INT_MAX)
					{
						if (long_transitions)
							look_back.set_element(m_long_transition_threshold, i, j) ;
						else
							look_back.set_element(max_look_back, i, j) ;
					}
				}
			SG_DEBUG("Using %i long transitions\n", num_long_transitions) ;
		}

		//SG_PRINT("use_svm=%i, genestr_len: \n", use_svm, m_genestr.get_dim1()) ;
		SG_DEBUG("use_svm=%i\n", use_svm) ;

		SG_DEBUG("maxlook: %d m_N: %d nbest: %d \n", max_look_back, m_N, nbest);
		const int32_t look_back_buflen = (max_look_back*m_N+1)*nbest ;
		SG_DEBUG("look_back_buflen=%i\n", look_back_buflen) ;
		/*const float64_t mem_use = (float64_t)(seq_len*m_N*nbest*(sizeof(T_STATES)+sizeof(int16_t)+sizeof(int32_t))+
		  look_back_buflen*(2*sizeof(float64_t)+sizeof(int32_t))+
		  seq_len*(sizeof(T_STATES)+sizeof(int32_t))+
		  m_genestr.get_dim1()*sizeof(bool))/(1024*1024);*/

		//bool is_big = (mem_use>200) || (seq_len>5000) ;

		/*if (is_big)
		  {
		  SG_DEBUG("calling best_path_trans: seq_len=%i, m_N=%i, lookback=%i nbest=%i\n", 
		  seq_len, m_N, max_look_back, nbest) ;
		  SG_DEBUG("allocating %1.2fMB of memory\n", 
		  mem_use) ;
		  }*/
		ASSERT(nbest<32000) ;



		CArray3<float64_t> delta(seq_len, m_N, nbest) ;
		delta.set_name("delta");
		float64_t* delta_array = delta.get_array() ;
		//delta.zero() ;

		CArray3<T_STATES> psi(seq_len, m_N, nbest) ;
		psi.set_name("psi");
		//psi.zero() ;

		CArray3<int16_t> ktable(seq_len, m_N, nbest) ;
		ktable.set_name("ktable");
		//ktable.zero() ;

		CArray3<int32_t> ptable(seq_len, m_N, nbest) ;	
		ptable.set_name("ptable");
		//ptable.zero() ;

		CArray<float64_t> delta_end(nbest) ;
		delta_end.set_name("delta_end");
		//delta_end.zero() ;

		CArray<T_STATES> path_ends(nbest) ;
		path_ends.set_name("path_ends");
		//path_ends.zero() ;

		CArray<int16_t> ktable_end(nbest) ;
		ktable_end.set_name("ktable_end");
		//ktable_end.zero() ;

		float64_t * fixedtempvv=new float64_t[look_back_buflen] ;
		memset(fixedtempvv, 0, look_back_buflen*sizeof(float64_t)) ;
		int32_t * fixedtempii=new int32_t[look_back_buflen] ;
		memset(fixedtempii, 0, look_back_buflen*sizeof(int32_t)) ;

		CArray<float64_t> oldtempvv(look_back_buflen) ;
		oldtempvv.set_name("oldtempvv");
		CArray<float64_t> oldtempvv2(look_back_buflen) ;
		oldtempvv2.set_name("oldtempvv2");
		//oldtempvv.zero() ;
		//oldtempvv.display_size() ;

		CArray<int32_t> oldtempii(look_back_buflen) ;
		oldtempii.set_name("oldtempii");
		CArray<int32_t> oldtempii2(look_back_buflen) ;
		oldtempii2.set_name("oldtempii2");
		//oldtempii.zero() ;

		CArray<T_STATES> state_seq(seq_len) ;
		state_seq.set_name("state_seq");
		//state_seq.zero() ;

		CArray<int32_t> pos_seq(seq_len) ;
		pos_seq.set_name("pos_seq");
		//pos_seq.zero() ;


		m_dict_weights.set_name("dict_weights") ;
		m_word_degree.set_name("word_degree") ;
		m_cum_num_words.set_name("cum_num_words") ;
		m_num_words.set_name("num_words") ;
		//word_used.set_name("word_used") ;
		//svm_values_unnormalized.set_name("svm_values_unnormalized") ;
		m_svm_pos_start.set_name("svm_pos_start") ;
		m_num_unique_words.set_name("num_unique_words") ;

		PEN.set_name("PEN") ;
		seq.set_name("seq") ;

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
		oldtempvv2.set_name("oldtempvv2") ;
		oldtempii.set_name("oldtempii") ;
		oldtempii2.set_name("oldtempii2") ;


		//////////////////////////////////////////////////////////////////////////////// 

#ifdef DYNPROG_DEBUG
		state_seq.display_size() ;
		pos_seq.display_size() ;

		m_dict_weights.display_size() ;
		m_word_degree.display_array() ;
		m_cum_num_words.display_array() ;
		m_num_words.display_array() ;
		//word_used.display_size() ;
		//svm_values_unnormalized.display_size() ;
		m_svm_pos_start.display_array() ;
		m_num_unique_words.display_array() ;

		PEN.display_size() ;
		PEN_state_signals.display_size() ;
		seq.display_size() ;
		m_orf_info.display_size() ;

		//m_genestr_stop.display_size() ;
		delta.display_size() ;
		psi.display_size() ;
		ktable.display_size() ;
		ptable.display_size() ;
		delta_end.display_size() ;
		path_ends.display_size() ;
		ktable_end.display_size() ;

#ifdef USE_TMP_ARRAYCLASS
		fixedtempvv.display_size() ;
		fixedtempii.display_size() ;
#endif

		//oldtempvv.display_size() ;
		//oldtempii.display_size() ;

		state_seq.display_size() ;
		pos_seq.display_size() ;

		//seq.zero() ;

#endif //DYNPROG_DEBUG

		////////////////////////////////////////////////////////////////////////////////



		{
			for (int32_t s=0; s<m_num_svms; s++)
				ASSERT(m_string_words_array[s]<1)  ;
		}


		//CArray2<int32_t*> trans_matrix_svms(m_N,m_N);
		//CArray2<int32_t> trans_matrix_num_svms(m_N,m_N);

		{ // initialization

			for (T_STATES i=0; i<m_N; i++)
			{
				//delta.element(0, i, 0) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
				delta.element(delta_array, 0, i, 0, seq_len, m_N) = get_p(i) + seq.element(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
				psi.element(0,i,0)   = 0 ;
				if (nbest>1)
					ktable.element(0,i,0)  = 0 ;
				ptable.element(0,i,0)  = 0 ;
				for (int16_t k=1; k<nbest; k++)
				{
					int32_t dim1, dim2, dim3 ;
					delta.get_array_size(dim1, dim2, dim3) ;
					//SG_DEBUG("i=%i, k=%i -- %i, %i, %i\n", i, k, dim1, dim2, dim3) ;
					//delta.element(0, i, k)    = -CMath::INFTY ;
					delta.element(delta_array, 0, i, k, seq_len, m_N)    = -CMath::INFTY ;
					psi.element(0,i,0)      = 0 ;                  // <--- what's this for?
					if (nbest>1)
						ktable.element(0,i,k)     = 0 ;
					ptable.element(0,i,k)     = 0 ;
				}
				/*
				   for (T_STATES j=0; j<m_N; j++)
				   {
				   CPlifBase * penalty = PEN.element(i,j) ;
				   int32_t num_current_svms=0;
				   int32_t svm_ids[] = {-8, -7, -6, -5, -4, -3, -2, -1};
				   if (penalty)
				   {
				   SG_PRINT("trans %i -> %i \n",i,j);
				   penalty->get_used_svms(&num_current_svms, svm_ids);
				   trans_matrix_svms.set_element(svm_ids,i,j);
				   for (int32_t l=0;l<num_current_svms;l++)
				   SG_PRINT("svm_ids[%i]: %i \n",l,svm_ids[l]);
				   trans_matrix_num_svms.set_element(num_current_svms,i,j);
				   }
				   }
				   */

			}
		}

		/*struct svm_values_struct svs;
		  svs.num_unique_words = NULL;
		  svs.svm_values = NULL;
		  svs.svm_values_unnormalized = NULL;
		  svs.word_used = NULL;*/

		struct segment_loss_struct loss;
		loss.segments_changed = NULL;
		loss.num_segment_id = NULL;

		SG_DEBUG("START_RECURSION \n\n");

		// recursion
		for (int32_t t=1; t<seq_len; t++)
		{
			//if (is_big && t%(1+(seq_len/1000))==1)
			//	SG_PROGRESS(t, 0, seq_len);
			//SG_PRINT("%i\n", t) ;

			if (with_loss)
			{
				init_segment_loss(loss, seq_len, max_look_back);
				find_segment_loss_till_pos(t, m_segment_ids, m_segment_mask, loss);  
			}

			for (T_STATES j=0; j<m_N; j++)
			{
				if (seq.element(j,t)<=-1e20)
				{ // if we cannot observe the symbol here, then we can omit the rest
					for (int16_t k=0; k<nbest; k++)
					{
						delta.element(delta_array, t, j, k, seq_len, m_N)    = seq.element(j,t) ;
						psi.element(t,j,k)         = 0 ;
						if (nbest>1)
							ktable.element(t,j,k)  = 0 ;
						ptable.element(t,j,k)      = 0 ;
					}
				}
				else
				{
					const T_STATES num_elem   = trans_list_forward_cnt[j] ;
					const T_STATES *elem_list = trans_list_forward[j] ;
					const float64_t *elem_val      = trans_list_forward_val[j] ;
					const int32_t *elem_id      = trans_list_forward_id[j] ;

					int32_t fixed_list_len = 0 ;
					float64_t fixedtempvv_ = CMath::INFTY ;
					int32_t fixedtempii_ = 0 ;
					bool fixedtemplong = false ;

					for (int32_t i=0; i<num_elem; i++)
					{
						T_STATES ii = elem_list[i] ;

						const CPlifBase * penalty = PEN.element(j,ii) ;

						/*int32_t look_back = max_look_back ;
						  if (0)
						  { // find lookback length
						  CPlifBase *pen = (CPlifBase*) penalty ;
						  if (pen!=NULL)
						  look_back=(int32_t) (CMath::ceil(pen->get_max_value()));
						  if (look_back>=1e6)
						  SG_PRINT("%i,%i -> %d from %ld\n", j, ii, look_back, (long)pen) ;
						  ASSERT(look_back<1e6);
						  } */

						int32_t look_back_ = look_back.element(j, ii) ;

						int32_t orf_from = m_orf_info.element(ii,0) ;
						int32_t orf_to   = m_orf_info.element(j,1) ;
						if((orf_from!=-1)!=(orf_to!=-1))
							SG_DEBUG("j=%i  ii=%i  orf_from=%i orf_to=%i p=%1.2f\n", j, ii, orf_from, orf_to, elem_val[i]) ;
						ASSERT((orf_from!=-1)==(orf_to!=-1)) ;

						int32_t orf_target = -1 ;
						if (orf_from!=-1)
						{
							orf_target=orf_to-orf_from ;
							if (orf_target<0) 
								orf_target+=3 ;
							ASSERT(orf_target>=0 && orf_target<3) ;
						}

						int32_t orf_last_pos = m_pos[t] ;
						int32_t loss_last_pos = t ;
						float64_t last_loss = 0.0 ;

#ifdef DYNPROG_TIMING
						MyTime3.start() ;
#endif				
						int32_t num_ok_pos = 0 ;
						float64_t last_mval=0 ;
						int32_t last_ts = 0 ;

						for (int32_t ts=t-1; ts>=0 && m_pos[t]-m_pos[ts]<=look_back_; ts--)
						{
							bool ok ;
							//int32_t plen=t-ts;

							/*for (int32_t s=0; s<m_num_svms; s++)
							  if ((fabs(svs.svm_values[s*svs.seqlen+plen]-svs2.svm_values[s*svs.seqlen+plen])>1e-6) ||
							  (fabs(svs.svm_values[s*svs.seqlen+plen]-svs3.svm_values[s*svs.seqlen+plen])>1e-6))
							  {
							  SG_DEBUG( "s=%i, t=%i, ts=%i, %1.5e, %1.5e, %1.5e\n", s, t, ts, svs.svm_values[s*svs.seqlen+plen], svs2.svm_values[s*svs.seqlen+plen], svs3.svm_values[s*svs.seqlen+plen]);
							  }*/

							if (orf_target==-1)
								ok=true ;
							else if (m_pos[ts]!=-1 && (m_pos[t]-m_pos[ts])%3==orf_target)
								ok=(!use_orf) || extend_orf(orf_from, orf_to, m_pos[ts], orf_last_pos, m_pos[t]) ;
							else
								ok=false ;

							if (ok)
							{

								float64_t segment_loss = 0.0 ;
								if (with_loss)
									segment_loss = extend_segment_loss(loss, elem_id[i], ts, loss_last_pos, last_loss) ;

								////////////////////////////////////////////////////////
								// BEST_PATH_TRANS
								////////////////////////////////////////////////////////

								int32_t frame = m_orf_info.element(ii,0);
								lookup_content_svm_values(ts, t, m_pos[ts], m_pos[t], svm_value, frame);


								//int32_t offset = plen*m_num_svms ;
								//for (int32_t ss=0; ss<m_num_svms; ss++)
								//{
								//	//svm_value[ss]=svs.svm_values[offset+ss];
								//	svm_value[ss]=new_svm_value[ss];
								//	//if (CMath::abs(new_svm_value[ss]-svm_value[ss])>1e-5)
								//	//	SG_PRINT("ts: %i t: %i  precomp: %f old: %f diff: %f \n",ts, t,new_svm_value[ss],svm_value[ss], CMath::abs(new_svm_value[ss]-svm_value[ss]));
								//}

								float64_t pen_val = 0.0 ;
								if (penalty)
								{
#ifdef DYNPROG_TIMING_DETAIL
									MyTime.start() ;
#endif								
									pen_val = penalty->lookup_penalty(m_pos[t]-m_pos[ts], svm_value) ;

#ifdef DYNPROG_TIMING_DETAIL
									MyTime.stop() ;
									content_plifs_time += MyTime.time_diff_sec() ;
#endif
								}

#ifdef DYNPROG_TIMING_DETAIL
								MyTime.start() ;
#endif								
								num_ok_pos++ ;

								if (nbest==1)
								{
									float64_t  val        = elem_val[i] + pen_val ;
									if (with_loss)
										val              += segment_loss ;

									float64_t mval = -(val + delta.element(delta_array, ts, ii, 0, seq_len, m_N)) ;

									if (mval<fixedtempvv_)
									{
										fixedtempvv_ = mval ;
										fixedtempii_ = ii + ts*m_N;
										fixed_list_len = 1 ;
										fixedtemplong = false ;
									}
									last_mval = mval ;
									last_ts = ts ;
								}
								else
								{
									for (int16_t diff=0; diff<nbest; diff++)
									{
										float64_t  val        = elem_val[i]  ;
										val                  += pen_val ;
										if (with_loss)
											val              += segment_loss ;

										float64_t mval = -(val + delta.element(delta_array, ts, ii, diff, seq_len, m_N)) ;

										/* only place -val in fixedtempvv if it is one of the nbest lowest values in there */
										/* fixedtempvv[i], i=0:nbest-1, is sorted so that fixedtempvv[0] <= fixedtempvv[1] <= ...*/
										/* fixed_list_len has the number of elements in fixedtempvv */

										if ((fixed_list_len < nbest) || ((0==fixed_list_len) || (mval < fixedtempvv[fixed_list_len-1])))
										{
											if ( (fixed_list_len<nbest) && ((0==fixed_list_len) || (mval>fixedtempvv[fixed_list_len-1])) )
											{
												fixedtempvv[fixed_list_len] = mval ;
												fixedtempii[fixed_list_len] = ii + diff*m_N + ts*m_N*nbest;
												fixed_list_len++ ;
											}
											else  // must have mval < fixedtempvv[fixed_list_len-1]
											{
												int32_t addhere = fixed_list_len;
												while ((addhere > 0) && (mval < fixedtempvv[addhere-1]))
													addhere--;

												// move everything from addhere+1 one forward 
												for (int32_t jj=fixed_list_len-1; jj>addhere; jj--)
												{
													fixedtempvv[jj] = fixedtempvv[jj-1];
													fixedtempii[jj] = fixedtempii[jj-1];
												}

												fixedtempvv[addhere] = mval;
												fixedtempii[addhere] = ii + diff*m_N + ts*m_N*nbest;

												if (fixed_list_len < nbest)
													fixed_list_len++;
											}
										}
									}
								}
#ifdef DYNPROG_TIMING_DETAIL
								MyTime.stop() ;
								inner_loop_max_time += MyTime.time_diff_sec() ;
#endif
							}
						}
#ifdef DYNPROG_TIMING
						MyTime3.stop() ;
						inner_loop_time += MyTime3.time_diff_sec() ;
#endif

						/* long transition stuff */
						/* only do this, if 
						 * this feature is enabled
						 * this is not a transition with ORF restrictions
						 * the loss is switched off
						 * nbest=1
						 */ 
#ifdef DYNPROG_TIMING
						MyTime3.start() ;
#endif
						if ( long_transitions && orf_target==-1 && look_back_ == m_long_transition_threshold )
						{
							int ts = t ;
							while (ts>0 && m_pos[t]-m_pos[ts-1] < m_long_transition_threshold)
								ts-- ;

							if (ts>0)
							{
								ASSERT((m_pos[t]-m_pos[ts-1] >= m_long_transition_threshold) && (m_pos[t]-m_pos[ts] < m_long_transition_threshold))

									/* only consider this transition, if the right position was found */
									float pen_val = 0.0 ;
								if (penalty)
								{
									int32_t frame = m_orf_info.element(ii,0);
									lookup_content_svm_values(ts, t, m_pos[ts], m_pos[t], svm_value, frame);
									pen_val = penalty->lookup_penalty(m_pos[t]-m_pos[ts], svm_value) ;
								}
								//if (m_pos[ts]==3812)
								//	SG_PRINT("%i,%i,%i: pen_val=%1.5f (t=%i, ts=%i, ts-1=%i, ts+1=%i)\n", m_pos[t], j, ii, pen_val, m_pos[t], m_pos[ts], m_pos[ts-1], m_pos[ts+1]) ;

								float64_t mval = -(long_transition_content_scores.get_element(ii, j) + pen_val*0.5) ;
								/* // incomplete extra check
								   float64_t mval2 = CMath::INFTY ;
								   if (long_transition_content_position.get_element(ii,j)>0)
								   mval2 = -( pen_val/2 + delta.element(delta_array, long_transition_content_position.get_element(ii,j), ii, 0, seq_len, m_N) + elem_val[i]  ) ;
								   if (fabs(mval-mval2)>1e-8)
								   SG_PRINT("!!!  mval=%1.2f  mval2=%1.2f\n", mval, mval2) ;

								   if (long_transition_content_position.get_element(ii,j)>0)
								   mval2 = -( pen_val/2 + delta.element(delta_array, long_transition_content_position.get_element(ii,j), ii, 0, seq_len, m_N) + elem_val[i]  ) ;
								   */


								if ((mval < fixedtempvv_) &&
										(m_pos[t] - m_pos[long_transition_content_position.get_element(ii, j)])<=m_long_transition_max)
								{
									/* then the long transition is better than the short one => replace it */ 
									int32_t fromtjk =  fixedtempii_ ;
									/*SG_PRINT("%i,%i: Long transition (%1.5f=-(%1.5f+%1.5f+%1.5f+%1.5f), %i) to m_pos %i better than short transition (%1.5f,%i) to m_pos %i \n", 
									  m_pos[t], j, 
									  mval, pen_val*0.5, long_transition_content_scores_pen.get_element(ii, j), long_transition_content_scores_elem.get_element(ii, j), long_transition_content_scores_prev.get_element(ii, j), ii, 
									  m_pos[long_transition_content_position.get_element(ii, j)], 
									  fixedtempvv_, (fromtjk%m_N), m_pos[(fromtjk-(fromtjk%(m_N*nbest)))/(m_N*nbest)]) ;*/
									ASSERT((fromtjk-(fromtjk%(m_N*nbest)))/(m_N*nbest)==0 || m_pos[(fromtjk-(fromtjk%(m_N*nbest)))/(m_N*nbest)]>=m_pos[long_transition_content_position.get_element(ii, j)] || fixedtemplong) ;

									fixedtempvv_ = mval ;
									fixedtempii_ = ii + m_N*long_transition_content_position.get_element(ii, j) ;
									fixed_list_len = 1 ;
									fixedtemplong = true ;
								}

								int32_t start = long_transition_content_start.get_element(ii, j) ;
								for (int32_t ts2=start; m_pos[t]-m_pos[ts2] > m_long_transition_threshold ; ts2++)
								{
									int32_t t2 = ts2 ;
									while (t2<=t && m_pos[t2+1]-m_pos[ts2]<m_long_transition_threshold)
										t2++ ;

									ASSERT(m_pos[t2+1]-m_pos[ts2] >= m_long_transition_threshold || t2==t) ;
									ASSERT(m_pos[t2]-m_pos[ts2] < m_long_transition_threshold) ;


									/* recompute penalty, if necessary */
									if (penalty)
									{
										int32_t frame = m_orf_info.element(ii,0);
										lookup_content_svm_values(ts2, t, m_pos[ts2], m_pos[t2], svm_value, frame);
										pen_val = penalty->lookup_penalty(m_pos[t2]-m_pos[ts2], svm_value) ;
									}

									//if (m_pos[ts2]==3812)
									//{
									//	SG_PRINT("%i - %i   vs  %i - %i\n", m_pos[t], m_pos[ts], m_pos[t2], m_pos[ts2]) ;
									//	SG_PRINT("ts=%i  t=%i  ts2=%i  seq_len=%i\n", m_pos[ts], m_pos[t], m_pos[ts2], seq_len) ;
									//}

									float64_t mval_trans = -( elem_val[i] + pen_val*0.5 + delta.element(delta_array, ts2, ii, 0, seq_len, m_N) ) ;
									//float64_t mval_trans = -( elem_val[i] + delta.element(delta_array, ts, ii, 0, seq_len, m_N) ) ; // enable this for the incomplete extra check

									if (m_pos[t2] - m_pos[long_transition_content_position.get_element(ii, j)] > m_long_transition_max)
									{
										long_transition_content_scores.set_element(-CMath::INFTY, ii, j) ;
										long_transition_content_scores_pen.set_element(0, ii, j) ;
										long_transition_content_scores_elem.set_element(0, ii, j) ;
										long_transition_content_scores_prev.set_element(0, ii, j) ;
										long_transition_content_position.set_element(0, ii, j) ;
									}

									if (-long_transition_content_scores.get_element(ii, j) > mval_trans )
									{
										/* then the old long transition is either too far away or worse than the current one */
										long_transition_content_scores.set_element(-mval_trans, ii, j) ;
										long_transition_content_scores_pen.set_element(pen_val*0.5, ii, j) ;
										long_transition_content_scores_elem.set_element(elem_val[i], ii, j) ;
										long_transition_content_scores_prev.set_element(delta.element(delta_array, ts2, ii, 0, seq_len, m_N), ii, j) ;
										long_transition_content_position.set_element(ts2, ii, j) ;
									}

									long_transition_content_start.set_element(ts2, ii, j) ;
								}

								/* // extra check
								   float64_t mval_trans2 = -( elem_val[i] + pen_val + delta.element(delta_array, ts, ii, 0, seq_len, m_N) ) ;
								   if (last_ts==ts && fabs(last_mval-mval_trans2)>1e-5)
								   SG_PRINT("last_mval=%1.2f at m_pos %i vs. mval_trans2=%1.2f at m_pos %i (diff=%f)\n", last_mval, m_pos[last_ts], mval_trans2, m_pos[ts], last_mval-mval_trans2) ;
								   */
							}
						}
					}
#ifdef DYNPROG_TIMING
					MyTime3.stop() ;
					long_transition_time += MyTime3.time_diff_sec() ;
#endif


					int32_t numEnt = fixed_list_len;

					float64_t minusscore;
					int64_t fromtjk;

					for (int16_t k=0; k<nbest; k++)
					{
						if (k<numEnt)
						{
							if (nbest==1)
							{
								minusscore = fixedtempvv_ ;
								fromtjk = fixedtempii_ ;
							}
							else
							{
								minusscore = fixedtempvv[k];
								fromtjk = fixedtempii[k];
							}

							delta.element(delta_array, t, j, k, seq_len, m_N)    = -minusscore + seq.element(j,t);
							psi.element(t,j,k)      = (fromtjk%m_N) ;
							if (nbest>1)
								ktable.element(t,j,k)   = (fromtjk%(m_N*nbest)-psi.element(t,j,k))/m_N ;
							ptable.element(t,j,k)   = (fromtjk-(fromtjk%(m_N*nbest)))/(m_N*nbest) ;
						}
						else
						{
							delta.element(delta_array, t, j, k, seq_len, m_N)    = -CMath::INFTY ;
							psi.element(t,j,k)      = 0 ;
							if (nbest>1)
								ktable.element(t,j,k)     = 0 ;
							ptable.element(t,j,k)     = 0 ;
						}
					}
				}
			}
		}

		//clear_svm_values(svs);
		if (with_loss)
			clear_segment_loss(loss);

		{ //termination
			int32_t list_len = 0 ;
			for (int16_t diff=0; diff<nbest; diff++)
			{
				for (T_STATES i=0; i<m_N; i++)
				{
					oldtempvv[list_len] = -(delta.element(delta_array, (seq_len-1), i, diff, seq_len, m_N)+get_q(i)) ;
					oldtempii[list_len] = i + diff*m_N ;
					list_len++ ;
				}
			}

			CMath::nmin(oldtempvv.get_array(), oldtempii.get_array(), list_len, nbest) ;

			for (int16_t k=0; k<nbest; k++)
			{
				delta_end.element(k) = -oldtempvv[k] ;
				path_ends.element(k) = (oldtempii[k]%m_N) ;
				if (nbest>1)
					ktable_end.element(k) = (oldtempii[k]-path_ends.element(k))/m_N ;
			}
		}

		{ //state sequence backtracking		
			for (int16_t k=0; k<nbest; k++)
			{
				prob_nbest[k]= delta_end.element(k) ;

				int32_t i         = 0 ;
				state_seq[i]  = path_ends.element(k) ;
				int16_t q   = 0 ;
				if (nbest>1)
					q=ktable_end.element(k) ;
				pos_seq[i]    = seq_len-1 ;

				while (pos_seq[i]>0)
				{
					//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
					state_seq[i+1] = psi.element(pos_seq[i], state_seq[i], q);
					pos_seq[i+1]   = ptable.element(pos_seq[i], state_seq[i], q) ;
					if (nbest>1)
						q              = ktable.element(pos_seq[i], state_seq[i], q) ;
					i++ ;
				}
				//SG_DEBUG("s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				int32_t num_states = i+1 ;
				for (i=0; i<num_states;i++)
				{
					my_state_seq[i+k*seq_len] = state_seq[num_states-i-1] ;
					my_pos_seq[i+k*seq_len]   = pos_seq[num_states-i-1] ;
				}
				my_state_seq[num_states+k*seq_len]=-1 ;
				my_pos_seq[num_states+k*seq_len]=-1 ;
			}
		}

		//if (is_big)
		//	SG_PRINT( "DONE.     \n") ;


#ifdef DYNPROG_TIMING
		MyTime2.stop() ;

		//if (is_big)
		SG_PRINT("Timing:  orf=%1.2f s \n Segment_init=%1.2f s Segment_pos=%1.2f s  Segment_extend=%1.2f s Segment_clean=%1.2f s\nsvm_init=%1.2f s  svm_pos=%1.2f  svm_clean=%1.2f\n  content_svm_values_time=%1.2f  content_plifs_time=%1.2f\ninner_loop_max_time=%1.2f inner_loop=%1.2f long_transition_time=%1.2f\n total=%1.2f\n", orf_time, segment_init_time, segment_pos_time, segment_extend_time, segment_clean_time, svm_init_time, svm_pos_time, svm_clean_time, content_svm_values_time, content_plifs_time, inner_loop_max_time, inner_loop_time, long_transition_time, MyTime2.time_diff_sec()) ;
#endif

		delete[] fixedtempvv ;
		delete[] fixedtempii ;
	}


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
	 * @param PLif_matrix Plif matrix
	 * @param Plif_state_signals Plif state signals
	 * @param max_num_signals maximal number of signals
	 */
	void best_path_trans_deriv(
			int32_t* my_state_seq, int32_t *my_pos_seq, float64_t *my_scores,
			float64_t* my_losses, int32_t my_seq_len, const float64_t *seq_array,
			int32_t seq_len, CPlifBase **PLif_matrix,
			CPlifBase **Plif_state_signals, int32_t max_num_signals);

	/// access function for number of states N
	inline T_STATES get_N() const
	  {
	    return m_N ;
	  }
	
	/** access function for probability of end states
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_q(T_STATES offset, float64_t value)
	{
		m_end_state_distribution_q[offset]=value;
	}

	/** access function for probability of first state
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_p(T_STATES offset, float64_t value)
	{
		m_initial_state_distribution_p[offset]=value;
	}

	/** access function for matrix a
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @param value value to be set
	 */
	inline void set_a(T_STATES line_, T_STATES column, float64_t value)
	{
	  m_transition_matrix_a.element(line_,column)=value; // look also best_path!
	}

	/** access function for probability of end states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_q(T_STATES offset) const
	{
		return m_end_state_distribution_q[offset];
	}

	/** access function for derivated probability of end states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_q_deriv(T_STATES offset) const
	{
		return m_end_state_distribution_q_deriv[offset];
	}

	/** access function for probability of initial states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_p(T_STATES offset) const
	{
		return m_initial_state_distribution_p[offset];
	}

	/** access function for derivated probability of initial states
	 *
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline float64_t get_p_deriv(T_STATES offset) const
	{
		return m_initial_state_distribution_p_deriv[offset];
	}
	
	/** create array of precomputed content svm values
	 * Jonas
	 *
	 * @param wordstr word strings
	 * @param pos position
	 * @param num_cand_pos number of cand position
	 * @param dictionary_weights SVM weight vectors for content prediction
	 * @param dict_len number of weight vectors 
	 */
	void precompute_content_values(const int32_t *pos,
		const int32_t num_cand_pos, float64_t *dictionary_weights, int32_t dict_len);


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
	/** set your own array of precomputed linear features like content predictions
	 *  and PLiFed tiling array data
	 * Jonas
	 *
	 * @param p_feat_array array of features
	 * @param p_num_svms number of tracks
	 * @param p_seq_len number of candidate positions
	 */
	inline void set_lin_feat(float64_t* p_lin_feat, int32_t p_num_svms, int32_t p_seq_len) 
	{
 	  m_lin_feat.set_array(p_lin_feat, p_num_svms, p_seq_len, true, true);
	}
	/** create word string from char*
	 * Jonas
	 *
	 * @param wordstr word strings
	 */
	void create_word_string();

	/** precompute stop codons
	 */
	void precompute_stop_codons();

	/** access function for matrix a
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline float64_t get_a(T_STATES line_, T_STATES column) const
	{
	  return m_transition_matrix_a.element(line_, column); // look also best_path()!
	}

	/** access function for matrix a derivated
	 *
	 * @param line_ row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline float64_t get_a_deriv(T_STATES line_, T_STATES column) const
	{
	  return m_transition_matrix_a_deriv.element(line_, column); // look also best_path()!
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
	void lookup_content_svm_values(const int32_t from_state,
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

	/** @brief SVM values */
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

	/** @brief segment loss */
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
		int32_t segment_id, int32_t pos, int32_t& last_pos, float64_t &last_value);

	/** find segment loss till pos
	 *
	 * @param pos position
	 * @param t_end t end
	 * @param segment_ids segment IDs
	 * @param segment_mask segmend mask
	 * @param loss segment loss
	 */
	void find_segment_loss_till_pos(int32_t t_end,
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
	int32_t m_N;

	/// transition matrix
	CArray2<int32_t> m_transition_matrix_a_id;
	CArray2<float64_t> m_transition_matrix_a;
	CArray2<float64_t> m_transition_matrix_a_deriv;

	/// initial distribution of states
	CArray<float64_t> m_initial_state_distribution_p;
	CArray<float64_t> m_initial_state_distribution_p_deriv;

	/// distribution of end-states
	CArray<float64_t> m_end_state_distribution_q;
	CArray<float64_t> m_end_state_distribution_q_deriv;

	//@}
	
	/** number of degress */
	int32_t m_num_degrees;
	/** number of SVMs */
	int32_t m_num_svms;
	/** number of strings */
	int32_t m_num_strings;
	
	/** word degree */
	CArray<int32_t> m_word_degree;
	/** cum num words */
	CArray<int32_t> m_cum_num_words;
	/** cum num words array */
	int32_t * m_cum_num_words_array;
	/** num words */
	CArray<int32_t> m_num_words;
	/** num words array */
	int32_t* m_num_words_array;
	/** mod words */
	CArray2<int32_t> m_mod_words;
	/** mod words array */
	int32_t* m_mod_words_array;
	/** sign words */
	CArray<bool> m_sign_words;
	/** sign words array */
	bool* m_sign_words_array;
	/** string words */
	CArray<int32_t> m_string_words;
	/** string words array */
	int32_t* m_string_words_array;

	/** SVM start position */
	CArray<int32_t> m_svm_pos_start;
	/** number of unique words */
	CArray<int32_t> m_num_unique_words;
	/** SVM arrays clean */
	bool m_svm_arrays_clean;

	/** number of SVMs single */
	int32_t m_num_svms_single;
	/** word degree single */
	int32_t m_word_degree_single;
	/** num words single */
	int32_t m_num_words_single;

	/** word used single */
	CArray<bool> m_word_used_single;
	/** SVM value unnormalised single */
	CArray<float64_t> m_svm_value_unnormalized_single;
	/** number of unique words single */
	int32_t m_num_unique_words_single;

	/** max a id */
	int32_t m_max_a_id;
	
	// input arguments
	/** sequence */
	CArray3<float64_t> m_seq;
	/** position */
	CArray<int32_t> m_pos;
	/** orf info */
	CArray2<int32_t> m_orf_info;
	/** segment sum weights */
	CArray2<float64_t> m_segment_sum_weights;
	/** Plif list */
	CArray<CPlifBase*> m_plif_list;
	/** PEN */
	CArray2<CPlifBase*> m_PEN;
	/** PEN state signals */
	CArray2<CPlifBase*> m_PEN_state_signals;
	/** a single string (to be segmented) */
	CArray<char> m_genestr;
	/** 
	  wordstr is a vector of L n-gram indices, with wordstr(i) representing a number betweeen 0 and 4095 
	  corresponding to the 6-mer in genestr(i-5:i) 
	  pos is a vector of candidate transition positions (it is input to best_path_trans)
	  t_end is some index in pos
	  
	  svs has been initialized by init_svm_values
	  
	  At the end of this procedure, 
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  The SVM weights are precomputed in m_dict_weights
	**/
	uint16_t*** m_wordstr;
	/** dict weights */
	CArray2<float64_t> m_dict_weights;
	/** segment loss */
	CArray3<float64_t> m_segment_loss;
	/** segment IDs */
	CArray<int32_t> m_segment_ids;
	/** segment mask */
	CArray<float64_t> m_segment_mask;
	/** my state seq */
	CArray<int32_t> m_my_state_seq;
	/** my position sequence */
	CArray<int32_t> m_my_pos_seq;
	/** my scores */
	CArray<float64_t> m_my_scores;
	/** my losses */
	CArray<float64_t> m_my_losses;

	// output arguments
	/** scores */
	CArray<float64_t> m_scores;
	/** states */
	CArray2<int32_t> m_states;
	/** positions */
	CArray2<int32_t> m_positions;

	CSparseFeatures<float64_t>* m_seq_sparse1;
	CSparseFeatures<float64_t>* m_seq_sparse2;

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
	float64_t *m_raw_intensities;
	/** probe position */
	int32_t* m_probe_pos;
	/** number of probes */
	int32_t* m_num_probes_cum;
	/** num lin feat plifs cum */
	int32_t* m_num_lin_feat_plifs_cum;
	/** number of additional data tracks like tiling, RNA-Seq, ...*/
	int32_t m_num_raw_data;

	bool m_long_transitions ;
	int32_t m_long_transition_threshold  ;
	int32_t m_long_transition_max ;

};
#endif
