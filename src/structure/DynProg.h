
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
#include "lib/DynArray2.h"

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
	
	/** Train definitions.
	 * Encapsulates Modelparameters that are constant/shall be learned.
	 * Consists of structures and access functions for learning only defined transitions and constants.
	 */
	class CModel
		{
		public:
			/// Constructor - initializes all variables/structures
			CModel();
			
			/// Destructor - cleans up
			virtual ~CModel();
			
			/// sorts learn_a matrix
			inline void sort_learn_a()
				{
					CMath::sort(learn_a,2) ;
				}
			
			/// sorts learn_b matrix
			inline void sort_learn_b()
				{
					CMath::sort(learn_b,2) ;
				}
			
			/**@name read access functions.
			 * For learn arrays and const arrays
			 */
			//@{
			/// get entry out of learn_a matrix
			inline INT get_learn_a(INT line, INT column) const
				{
					return learn_a[line*2 + column];
				}
			
			/// get entry out of learn_b matrix
			inline INT get_learn_b(INT line, INT column) const 
				{
					return learn_b[line*2 + column];
				}
			
			/// get entry out of learn_p vector
			inline INT get_learn_p(INT offset) const 
				{
					return learn_p[offset];
				}
			
			/// get entry out of learn_q vector
			inline INT get_learn_q(INT offset) const 
				{
					return learn_q[offset];
				}
			
			/// get entry out of const_a matrix
			inline INT get_const_a(INT line, INT column) const
				{
					return const_a[line*2 + column];
				}
			
			/// get entry out of const_b matrix
			inline INT get_const_b(INT line, INT column) const 
				{
					return const_b[line*2 + column];
				}
			
			/// get entry out of const_p vector
			inline INT get_const_p(INT offset) const 
				{
					return const_p[offset];
				}
			
			/// get entry out of const_q vector
			inline INT get_const_q(INT offset) const
				{
					return const_q[offset];
				}
			
			/// get value out of const_a_val vector
			inline DREAL get_const_a_val(INT line) const
				{
					return const_a_val[line];
				}
			
			/// get value out of const_b_val vector
			inline DREAL get_const_b_val(INT line) const 
				{
					return const_b_val[line];
				}
			
			/// get value out of const_p_val vector
			inline DREAL get_const_p_val(INT offset) const 
				{
					return const_p_val[offset];
				}
			
			/// get value out of const_q_val vector
			inline DREAL get_const_q_val(INT offset) const
				{
					return const_q_val[offset];
				}
			//@}
			
			/**@name write access functions
			 * For learn and const arrays
			 */
			//@{
			/// set value in learn_a matrix
			inline void set_learn_a(INT offset, INT value)
				{
					learn_a[offset]=value;
				}
			
			/// set value in learn_b matrix
			inline void set_learn_b(INT offset, INT value)
				{
					learn_b[offset]=value;
				}
			
			/// set value in learn_p vector
			inline void set_learn_p(INT offset, INT value)
				{
					learn_p[offset]=value;
				}
			
			/// set value in learn_q vector
			inline void set_learn_q(INT offset, INT value)
				{
					learn_q[offset]=value;
				}
			
			/// set value in const_a matrix
			inline void set_const_a(INT offset, INT value)
				{
					const_a[offset]=value;
				}
			
			/// set value in const_b matrix
			inline void set_const_b(INT offset, INT value)
				{
					const_b[offset]=value;
				}
			
			/// set value in const_p vector
			inline void set_const_p(INT offset, INT value)
				{
					const_p[offset]=value;
				}
			
			/// set value in const_q vector
			inline void set_const_q(INT offset, INT value)
				{
					const_q[offset]=value;
				}
			
			/// set value in const_a_val vector
			inline void set_const_a_val(INT offset, DREAL value)
				{
					const_a_val[offset]=value;
				}
			
			/// set value in const_b_val vector
			inline void set_const_b_val(INT offset, DREAL value)
				{
					const_b_val[offset]=value;
				}
			
			/// set value in const_p_val vector
			inline void set_const_p_val(INT offset, DREAL value)
				{
					const_p_val[offset]=value;
				}
			
			/// set value in const_q_val vector
			inline void set_const_q_val(INT offset, DREAL value)
				{
					const_q_val[offset]=value;
				}
			//@}
			
		protected:
			/**@name learn arrays.
			 * Everything that is to be learned is enumerated here.
			 * All values will be inititialized with random values
			 * and normalized to satisfy stochasticity.
			 */
			//@{
			/// transitions to be learned 
			INT* learn_a;
			
			/// emissions to be learned
			INT* learn_b;
			
			/// start states to be learned
			INT* learn_p;
			
			/// end states to be learned
			INT* learn_q;
			//@}
			
			/**@name constant arrays.
			 * These arrays hold constant fields. All values that
			 * are not constant and will not be learned are initialized
			 * with 0.
			 */
			//@{
			/// transitions that have constant probability
			INT* const_a;
			
			/// emissions that have constant probability
			INT* const_b;
			
			/// start states that have constant probability
			INT* const_p;
			
			/// end states that have constant probability
			INT* const_q;		
			
			
			/// values for transitions that have constant probability
			DREAL* const_a_val;
			
			/// values for emissions that have constant probability
			DREAL* const_b_val;
			
			/// values for start states that have constant probability
			DREAL* const_p_val;
			
			/// values for end states that have constant probability
			DREAL* const_q_val;		
			
			//@}
		};
	

public:
	/**@name Constructor/Destructor and helper function
	 */
	//@{
	/** Constructor
	 * @param N number of states
	 * @param M number of emissions
	 * @param model model which holds definitions of states to be learned + consts
	 */
	CDynProg(INT N, INT M,	CModel* model);
	CDynProg(INT N, double* p, double* q, double* a) ;
	CDynProg(INT N, double* p, double* q, int num_trans, double* a_trans) ;

	/// Constructor - Clone model h
	CDynProg(CDynProg* h);

	/// Destructor - Cleanup
	virtual ~CDynProg();
	
	/** initialization function - gets called by constructors.
	 * @param model model which holds definitions of states to be learned + consts
	 */
	bool initialize(CModel* model);
	//@}
	
	/// allocates memory that depends on N
	bool alloc_state_dependend_arrays();

	/// free memory that depends on N
	void free_state_dependend_arrays();

	inline T_STATES get_psi(INT time, T_STATES state) const
	{
#ifdef HMM_DEBUG
	  if ((time>=p_observations->get_max_vector_length())||(state>N))
	    CIO::message(stderr,"index out of range in get_psi(%i,%i) [%i,%i]\n",time,state,p_observations->get_max_vector_length(),N) ;
#endif
	  return states_per_observation_psi[time*N+state];
	}
	inline void set_psi(INT time, T_STATES state, T_STATES value)
	{
#ifdef HMM_DEBUG
	  if ((time>=p_observations->get_max_vector_length())||(state>N))
	    CIO::message(stderr,"index out of range in set_psi(%i,%i,.) [%i,%i]\n",time,state,p_observations->get_max_vector_length(),N) ;
#endif
	  states_per_observation_psi[time*N+state]=value;
	}

	void translate_from_single_order(WORD* obs, INT sequence_length, 
									 INT start, INT order, 
									 INT max_val=2/*DNA->2bits*/) ;

	/** calculates probability of best state sequence s_0,...,s_T-1 AND path itself using viterbi algorithm.
	 * The path can be found in the array PATH(dimension)[0..T-1] afterwards
	 * @param dimension dimension of observation for which the most probable path is calculated (observations are a matrix, where a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1} 
	 */
	DREAL best_path_no_b(INT max_iter, INT & best_iter, INT *my_path) ;
	void best_path_no_b_trans(INT max_iter, INT & max_best_iter, short int nbest, DREAL *prob_nbest, INT *my_paths) ;
	//void best_path_no_b_trans1(INT max_iter, INT & max_best_iter, DREAL *prob_nbest, INT *my_paths) ;
	void model_prob_no_b_trans(INT max_iter, DREAL *prob_iter) ;
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
	
	/// access function for number of observations M
	inline INT get_M() const
	  {
	    return M ;
	  }
	
	/** access function for probability of end states
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_q(T_STATES offset, DREAL value)
	{
#ifdef HMM_DEBUG
	  if (offset>=N)
	    CIO::message(stderr,"index out of range in set_q(%i,%e) [%i]\n", offset,value,N) ;
#endif
		end_state_distribution_q[offset]=value;
	}

	/** access function for probability of first state
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_p(T_STATES offset, DREAL value)
	{
#ifdef HMM_DEBUG
	  if (offset>=N)
	    CIO::message(stderr,"index out of range in set_p(%i,.) [%i]\n", offset,N) ;
#endif
		initial_state_distribution_p[offset]=value;
	}

	/** access function for matrix A
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @param value value to be set
	 */
	inline void set_A(T_STATES line_, T_STATES column, DREAL value)
	{
#ifdef HMM_DEBUG
	  if ((line_>N)||(column>N))
	    CIO::message(stderr,"index out of range in set_A(%i,%i,.) [%i,%i]\n",line_,column,N,N) ;
#endif
		transition_matrix_A[line_+column*N]=value;
	}

	/** access function for matrix a 
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @param value value to be set
	 */
	inline void set_a(T_STATES line_, T_STATES column, DREAL value)
	{
#ifdef HMM_DEBUG
	  if ((line_>N)||(column>N))
	    CIO::message(stderr,"index out of range in set_a(%i,%i,.) [%i,%i]\n",line_,column,N,N) ;
#endif
	  transition_matrix_a[line_+column*N]=value; // look also best_path!
	}

	/** access function for matrix B
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...M-1
	 * @param value value to be set
	 */
	inline void set_B(T_STATES line_, WORD column, DREAL value)
	{
#ifdef HMM_DEBUG
	  if ((line_>=N)||(column>=M))
	    CIO::message(stderr,"index out of range in set_B(%i,%i) [%i,%i]\n", line_, column,N,M) ;
#endif
	  observation_matrix_B[line_*M+column]=value;
	}

	/** access function for matrix b
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...M-1
	 * @param value value to be set
	 */
	inline void set_b(T_STATES line_, WORD column, DREAL value)
	{
#ifdef HMM_DEBUG
	  if ((line_>=N)||(column>=M))
	    CIO::message(stderr,"index out of range in set_b(%i,%i) [%i,%i]\n", line_, column,N,M) ;
#endif
		observation_matrix_b[line_*M+column]=value;
	}

	/** access function for probability of end states
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_q(T_STATES offset) const 
	{
#ifdef HMM_DEBUG
	  if (offset>=N)
	    CIO::message(stderr,"index out of range in %e=get_q(%i) [%i]\n", end_state_distribution_q[offset],offset,N) ;
#endif
		return end_state_distribution_q[offset];
	}

	/** access function for probability of initial states
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline DREAL get_p(T_STATES offset) const 
	{
#ifdef HMM_DEBUG
	  if (offset>=N)
	    CIO::message(stderr,"index out of range in get_p(%i,.) [%i]\n", offset,N) ;
#endif
		return initial_state_distribution_p[offset];
	}

	/** access function for matrix A
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline DREAL get_A(T_STATES line_, T_STATES column) const
	{
#ifdef HMM_DEBUG
	  if ((line_>N)||(column>N))
	    CIO::message(stderr,"index out of range in get_A(%i,%i) [%i,%i]\n",line_,column,N,N) ;
#endif
		return transition_matrix_A[line_+column*N];
	}

	/** access function for matrix a
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...N-1
	 * @return value at position line colum
	 */
	inline DREAL get_a(T_STATES line_, T_STATES column) const
	{
#ifdef HMM_DEBUG
	  if ((line_>N)||(column>N))
	    CIO::message(stderr,"index out of range in get_a(%i,%i) [%i,%i]\n",line_,column,N,N) ;
#endif
	  return transition_matrix_a[line_+column*N]; // look also best_path()!
	}

	/** access function for matrix B
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...M-1
	 * @return value at position line colum
	 */
	inline DREAL get_B(T_STATES line_, WORD column) const
	{
#ifdef HMM_DEBUG
	  if ((line_>=N)||(column>=M))
	    CIO::message(stderr,"index out of range in get_B(%i,%i) [%i,%i]\n", line_, column,N,M) ;
#endif
		return observation_matrix_B[line_*M+column];
	}

	/** access function for matrix b
	 * @param line row in matrix 0...N-1
	 * @param column column in matrix 0...M-1
	 * @return value at position line colum
	 */
	inline DREAL get_b(T_STATES line_, WORD column) const 
	{
#ifdef HMM_DEBUG
	  if ((line_>=N)||(column>=M))
	    CIO::message(stderr,"index out of range in get_b(%i,%i) [%i,%i]\n", line_, column,N,M) ;
#endif
	  return observation_matrix_b[line_*M+column];
	}

	//@}
protected:

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
	bool extend_orf(const CDynamicArray<bool>& genestr_stop, INT orf_from, INT orf_to, INT start, INT &last_pos, INT to) ;

	/**@name model specific variables.
	 * these are p,q,a,b,N,M etc 
	 */
	//@{
	/// number of observation symbols eg. ACGT -> 0123
	INT M;

	/// number of states
	INT N;

	//train definition for HMM
	CModel* model;

	/// matrix  of absolute counts of transitions 
	DREAL* transition_matrix_A;

	/// matrix of absolute counts of observations within each state
	DREAL* observation_matrix_B;

	/// transition matrix 
	DREAL* transition_matrix_a;

	/// initial distribution of states
	DREAL* initial_state_distribution_p;

	/// distribution of end-states
	DREAL* end_state_distribution_q;		

	/// distribution of observations within each state
	DREAL* observation_matrix_b;	

	T_STATES* states_per_observation_psi ;
	
	/// probability of model
	DREAL mod_prob;	

	/// true if model probability is up to date
	bool mod_prob_updated;	
	
	// true if model is using log likelihood
	bool loglikelihood;		

	// true->ok, false->error
	bool status;			

	// true->stolen from other HMMs, false->got own
	bool reused_caches;
	//@}

	static const INT num_degrees ;
	static const INT num_svms  ;
	
	static const INT word_degree[] ;
	static const INT cum_num_words[] ;
	static const INT num_words[] ;

	static CDynamicArray2<bool> word_used ;
	static CDynamicArray2<DREAL> svm_values_unnormalized ;
	static DREAL *dict_weights ;
	static INT svm_pos_start[] ;
	static INT num_unique_words[] ;

	static const INT num_svms_single  ;
	static const INT word_degree_single ;
	static const INT cum_num_words_single ;
	static const INT num_words_single ;

	static bool word_used_single[] ;
	static DREAL svm_value_unnormalized_single[] ;
	static INT num_unique_words_single ;
};
#endif
