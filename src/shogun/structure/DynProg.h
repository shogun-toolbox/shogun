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

#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/structure/PlifMatrix.h>
#include <shogun/structure/PlifBase.h>
#include <shogun/structure/Plif.h>
#include <shogun/structure/IntronList.h>
#include <shogun/structure/SegmentLoss.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/distributions/Distribution.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/Time.h>

#include <stdio.h>
#include <limits.h>

namespace shogun
{
	template <class T> class CSparseFeatures;
	class CIntronList;
	class CPlifMatrix;
	class CSegmentLoss;

	template <class T> class CDynamicArray;

//#define DYNPROG_TIMING

#ifdef USE_BIGSTATES
typedef uint16_t T_STATES ;
#else
typedef uint8_t T_STATES ;
#endif
typedef T_STATES* P_STATES ;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif

/** @brief Dynamic Programming Class.
 *
 * Structure and Function collection.
 * This Class implements a Dynamic Programming functions.
 */
class CDynProg : public CSGObject
{
public:
	/** constructor
	 *
	 * @param p_num_svms number of SVMs
	 */
	CDynProg(int32_t p_num_svms=8);
	virtual ~CDynProg();

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

	/** init CDynamicArray for precomputed content svm values
	 *  with size seq_len x num_svms
	 *
	 *  @param p_num_svms: number of svm weight vectors for content prediction
	 */
	void init_content_svm_value_array(const int32_t p_num_svms);

	/** init CDynamicArray for precomputed tiling intensitie-plif-values
	 *  with size seq_len x num_svms
	 *
	 *  @param probe_pos local positions of probes
	 *  @param intensities intensities of probes
	 *  @param num_probes number of probes
	 */
	void init_tiling_data(int32_t* probe_pos, float64_t* intensities, const int32_t num_probes);

	/** precompute tiling Plifs
	 *
	 * @param PEN Plif PEN
	 * @param tiling_plif_ids tiling plif id's
	 * @param num_tiling_plifs number of tiling plifs
	 */
	void precompute_tiling_plifs(CPlif** PEN, const int32_t* tiling_plif_ids, const int32_t num_tiling_plifs);

	/** append rows to linear features array
	 *
	 * @param num_new_feat number of new rows to add
	 */
	void resize_lin_feat(int32_t num_new_feat);
	/** set vector p
	 *
	 * @param p new vector p
	 */
	void set_p_vector(SGVector<float64_t> p);

	/** set vector q
	 *
	 * @param q new vector q
	 */
	void set_q_vector(SGVector<float64_t> q);

	/** set matrix a
	 *
	 * @param a new matrix a
	 */
	void set_a(SGMatrix<float64_t> a);

	/** set a id
	 *
	 * @param a new a id
	 */
	void set_a_id(SGMatrix<int32_t> a);

	/** set a transition matrix
	 *
	 * @param a_trans transition matrix a
	 */
	void set_a_trans_matrix(SGMatrix<float64_t> a_trans);

	/** init mod words array
	 *
	 * @param p_mod_words_array new mod words array
	 */
	void init_mod_words_array(SGMatrix<int32_t> p_mod_words_array);

	/** check SVM arrays
	 * call this function to check consistency
	 *
	 * @return whether arrays are ok
	 */
	bool check_svm_arrays();

	/** set best path seq
	 *
	 * @param seq signal features
	 */
	void set_observation_matrix(SGNDArray<float64_t> seq);

	/** get number of positions; the dynamic program is sparse encoded
	 *  and this function gives the number of positions that can actually
	 *  be part of a predicted path
	 *
	 * @return number of positions
	 */
	int32_t get_num_positions();

	/** set an array of length #(candidate positions)
	 *  which specifies the content type of each pos
	 *  and a mask that determines to which extend the
	 *  loss should be applied to this position; this
	 *  is a way to encode label confidence via weights
	 *  between zero and one
	 *
	 * @param seg_path seg path
	 */
	void set_content_type_array(SGMatrix<float64_t> seg_path);

	/** set best path pos
	 *
	 * @param pos the position vector
	 */
	void set_pos(SGVector<int32_t> pos);

	/** set best path orf info
	 * only for compute_nbest_paths
	 *
	 * @param orf_info the orf info
	 */
	void set_orf_info(SGMatrix<int32_t> orf_info);

	/** set best path genesstr
	 *
	 * @param genestr gene string
	 */
	void set_gene_string(SGVector<char> genestr);


	/** set best path dict weights
	 *
	 * @param dictionary_weights dictionary weights
	 */
	void set_dict_weights(SGMatrix<float64_t> dictionary_weights);

	/** set best path segment loss
	 *
	 * @param segment_loss segment loss
	 */
	void best_path_set_segment_loss(SGMatrix<float64_t> segment_loss);

	/** set best path segmend ids mask
	 *
	 * @param segment_ids segment ids
	 * @param segment_mask segment mask
	 * @param m dimension m
	 */
	void best_path_set_segment_ids_mask(int32_t* segment_ids, float64_t* segment_mask, int32_t m);

	/** set sparse feature matrices */
	void set_sparse_features(CSparseFeatures<float64_t>* seq_sparse1, CSparseFeatures<float64_t>* seq_sparse2);

	/** set plif matrices
	 *
	 * @param pm plif matrix object
	 */
	void set_plif_matrices(CPlifMatrix* pm);

	// best_path result retrieval functions
	/** best path get scores
	 *
	 * @return scores scores
	 */
	SGVector<float64_t> get_scores();

	/** best path get states
	 *
	 * @return states states
	 */
	SGMatrix<int32_t> get_states();

	/** best path get positions
	 *
	 * @return positions positions
	 */
	SGMatrix<int32_t> get_positions();


	/** run the viterbi algorithm to compute the n best viterbi paths
	 *
	 * @param max_num_signals maximal number of signals for a single state
	 * @param use_orf whether orf shall be used
	 * @param nbest number of best paths (n)
	 * @param with_loss use loss
	 * @param with_multiple_sequences !!!not functional set to false!!!
	 */
	void compute_nbest_paths(int32_t max_num_signals,
						 bool use_orf, int16_t nbest, bool with_loss, bool with_multiple_sequences);

////////////////////////////////////////////////////////////////////////////////

	/** given a path though the state model and the corresponding
	 *  positions compute the features. This can be seen as the derivative
	 *  of the score (output of dynamic program) with respect to the
	 *  parameters
	 *
	 * @param my_state_seq state sequence of the path
	 * @param my_pos_seq sequence of positions
	 * @param my_seq_len length of state and position sequences
	 * @param seq_array array of features
	 * @param max_num_signals maximal number of signals
	 */
	void best_path_trans_deriv(
			int32_t* my_state_seq, int32_t *my_pos_seq,
			int32_t my_seq_len, const float64_t *seq_array, int32_t max_num_signals);

	// additional best_path_trans_deriv functions
	/** set best path my state sequence
	 *
	 * @param my_state_seq my state sequence
	 */
	void set_my_state_seq(int32_t* my_state_seq);

	/** set best path my position sequence
	 *
	 * @param my_pos_seq my position sequence
	 */
	void set_my_pos_seq(int32_t* my_pos_seq);

	/** get path scores
	 *
	 * best_path_trans_deriv result retrieval functions
	 *
	 * @param my_scores scores
	 * @param seq_len length of sequence
	 */
	void get_path_scores(float64_t** my_scores, int32_t* seq_len);

	/** get path losses
	 *
	 * best_path_trans_deriv result retrieval functions
	 *
	 * @param my_losses my losses
	 * @param seq_len length of sequence
	 */
	void get_path_losses(float64_t** my_losses, int32_t* seq_len);


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
	 *
	 */
	void precompute_content_values();

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
	 * @param p_lin_feat array of features
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
	/** set intron list
	 *
	 * @param intron_list
	 * @param num_plifs number of intron plifs
	 */
	void set_intron_list(CIntronList* intron_list, int32_t num_plifs);

	/** get the segment loss object */
	CSegmentLoss* get_segment_loss_object()
	{
		return m_seg_loss_obj;
	}

	/** settings for long transition handling
	 *
	 *  @param use_long_transitions use the long transition approximation
	 *  @param threshold use long transition for segments larger than
	 *  @param max_len allow transitions up to
	 *  */
	void long_transition_settings(bool use_long_transitions, int32_t threshold, int32_t max_len)
	{
		m_long_transitions = use_long_transitions;
		m_long_transition_threshold = threshold;
		SG_DEBUG("ignoring max_len\n")
		//m_long_transition_max = max_len;
	}

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

#ifndef DOXYGEN_SHOULD_SKIP_THIS
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
#endif // DOXYGEN_SHOULD_SKIP_THIS

	/** extend orf
	 *
	 * @param orf_from orf from
	 * @param orf_to orf to
	 * @param start start
	 * @param last_pos last position
	 * @param to to
	 */
	bool extend_orf(int32_t orf_from, int32_t orf_to, int32_t start, int32_t &last_pos, int32_t to);

	/** @return object name */
	virtual const char* get_name() const { return "DynProg"; }

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


protected:
	/**@name model specific variables.
	 * these are p,q,a,b,N,M etc
	 */
	//@{
	/// number of states
	int32_t m_N;

	/// transition matrix
	CDynamicArray<int32_t> m_transition_matrix_a_id; // 2d
	CDynamicArray<float64_t> m_transition_matrix_a; // 2d
	CDynamicArray<float64_t> m_transition_matrix_a_deriv; // 2d

	/// initial distribution of states
	CDynamicArray<float64_t> m_initial_state_distribution_p;
	CDynamicArray<float64_t> m_initial_state_distribution_p_deriv;

	/// distribution of end-states
	CDynamicArray<float64_t> m_end_state_distribution_q;
	CDynamicArray<float64_t> m_end_state_distribution_q_deriv;

	//@}

	/** number of degress */
	int32_t m_num_degrees;
	/** number of SVMs */
	int32_t m_num_svms;

	/** word degree */
	CDynamicArray<int32_t> m_word_degree;
	/** cum num words */
	CDynamicArray<int32_t> m_cum_num_words;
	/** cum num words array */
	int32_t * m_cum_num_words_array;
	/** num words */
	CDynamicArray<int32_t> m_num_words;
	/** num words array */
	int32_t* m_num_words_array;
	/** mod words */
	CDynamicArray<int32_t> m_mod_words; // 2d
	/** mod words array */
	int32_t* m_mod_words_array;
	/** sign words */
	CDynamicArray<bool> m_sign_words;
	/** sign words array */
	bool* m_sign_words_array;
	/** string words */
	CDynamicArray<int32_t> m_string_words;
	/** string words array */
	int32_t* m_string_words_array;

	/** SVM start position */
//	CDynamicArray<int32_t> m_svm_pos_start;
	/** number of unique words */
	CDynamicArray<int32_t> m_num_unique_words;
	/** SVM arrays clean */
	bool m_svm_arrays_clean;
	/** max a id */
	int32_t m_max_a_id;

	// input arguments
	/** sequence */
	CDynamicArray<float64_t> m_observation_matrix; //3d
	/** candidate position */
	CDynamicArray<int32_t> m_pos;
	/** number of candidate positions */
	int32_t m_seq_len;
	/** orf info */
	CDynamicArray<int32_t> m_orf_info; // 2d
	/** segment sum weights */
	CDynamicArray<float64_t> m_segment_sum_weights; // 2d
	/** Plif list */
	CDynamicObjectArray m_plif_list; // CPlifBase*
	/** a single string (to be segmented) */
	CDynamicArray<char> m_genestr;
	/**
	  wordstr is a vector of L n-gram indices, with wordstr(i) representing a number betweeen 0 and 4095
	  corresponding to the 6-mer in genestr(i-5:i)
	  pos is a vector of candidate transition positions (it is input to compute_nbest_paths)
	  t_end is some index in pos

	  svs has been initialized by init_svm_values

	  At the end of this procedure,
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end))
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback

	  The SVM weights are precomputed in m_dict_weights
	**/
	uint16_t*** m_wordstr;
	/** dict weights */
	CDynamicArray<float64_t> m_dict_weights; // 2d
	/** segment loss */
	CDynamicArray<float64_t> m_segment_loss; // 3d
	/** segment IDs */
	CDynamicArray<int32_t> m_segment_ids;
	/** segment mask */
	CDynamicArray<float64_t> m_segment_mask;
	/** my state seq */
	CDynamicArray<int32_t> m_my_state_seq;
	/** my position sequence */
	CDynamicArray<int32_t> m_my_pos_seq;
	/** my scores */
	CDynamicArray<float64_t> m_my_scores;
	/** my losses */
	CDynamicArray<float64_t> m_my_losses;

	/** segment loss object containing the functions
	 *  to compute the segment loss*/
	CSegmentLoss* m_seg_loss_obj;

	// output arguments
	/** scores */
	CDynamicArray<float64_t> m_scores;
	/** states */
	CDynamicArray<int32_t> m_states; // 2d
	/** positions */
	CDynamicArray<int32_t> m_positions; // 2d

	/** sparse feature matrix dim1*/
	CSparseFeatures<float64_t>* m_seq_sparse1;
	/** sparse feature matrix dim2*/
	CSparseFeatures<float64_t>* m_seq_sparse2;
	/** plif matrices*/
	CPlifMatrix* m_plif_matrices;

	/** storeage of stop codons
	 *  array of size length(sequence)
	 */
	CDynamicArray<bool> m_genestr_stop;

	/** administers a list of introns and quality scores
	 *  and provides functions for fast access */
	CIntronList* m_intron_list;

	/** number of intron features and plifs*/
	int32_t m_num_intron_plifs;

	/**
	 *  array for storage of precomputed linear features linge content svm values or pliffed tiling data
	 * Jonas
	 */
	CDynamicArray<float64_t> m_lin_feat; // 2d

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

	/** use long transition approximation*/
	bool m_long_transitions ;
	/** threshold for transitions that are computed
	 *  the traditional way*/
	int32_t m_long_transition_threshold  ;
	/** maximal length of a long transition
	 *  Note: is ignored in the current implementation
	 *        => arbitrarily long transitions can be decoded
	 */
	//int32_t m_long_transition_max ;

	/**default values defining the k-mer degrees
	 * used for content type prediction
	 */
	static int32_t word_degree_default[4];

	/**default values storing the cumulative sum
	 * of the number of kmers that exist for the
	 * different degrees e.g. matlab spoken: cumsum(4.^[3 4 5 6])*/
	static int32_t cum_num_words_default[5];

	/**default values defining which of the plif are the
	 * frame specific plifs*/
	static int32_t frame_plifs[3];

	/**default values like cum_num_words_default
	 * but not cumsumed: e.g. 4.^[3 4 5 6]*/
	static int32_t num_words_default[4];

	/**default values*/
	static int32_t mod_words_default[32];

	/**default values*/
	static bool sign_words_default[16];

	/**default values*/
	static int32_t string_words_default[16];
};
}
#endif
