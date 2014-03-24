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

#ifndef __CHMM_H__
#define __CHMM_H__


#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/distributions/Distribution.h>

#include <stdio.h>

#ifdef USE_HMMPARALLEL
#define USE_HMMPARALLEL_STRUCTURES 1
#endif

namespace shogun
{
	class CFeatures;
	template <class ST> class CStringFeatures;
/**@name HMM specific types*/
//@{

/// type for alpha/beta caching table
typedef float64_t T_ALPHA_BETA_TABLE;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
/// type for alpha/beta table
struct T_ALPHA_BETA
{
	/// dimension for that alpha/beta table was generated
	int32_t dimension;

	/// perversely huge alpha/beta cache table
	T_ALPHA_BETA_TABLE* table;

	/// true if table is valid
	bool updated;

	/// sum over all paths == model_probability for this dimension
	float64_t sum;
};
#endif // DOXYGEN_SHOULD_SKIP_THIS

/** type that is used for states.
 * Probably uint8_t is enough if you have at most 256 states,
 * however uint16_t/long/... is also possible although you might quickly run into memory problems
 */
#ifdef USE_BIGSTATES
typedef uint16_t T_STATES ;
#else
typedef uint8_t T_STATES ;
#endif
typedef T_STATES* P_STATES ;

//@}

/** Training type */
enum BaumWelchViterbiType
{
	/// standard baum welch
	BW_NORMAL,
	/// baum welch only for specified transitions
	BW_TRANS,
	/// baum welch only for defined transitions/observations
	BW_DEFINED,
	/// standard viterbi
	VIT_NORMAL,
	/// viterbi only for defined transitions/observations
	VIT_DEFINED
};


/** @brief class Model */
class Model
{
	public:
		/// Constructor - initializes all variables/structures
		Model();

		/// Destructor - cleans up
		virtual ~Model();

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
		inline int32_t get_learn_a(int32_t line, int32_t column) const
		{
			return learn_a[line*2 + column];
		}

		/// get entry out of learn_b matrix
		inline int32_t get_learn_b(int32_t line, int32_t column) const
		{
			return learn_b[line*2 + column];
		}

		/// get entry out of learn_p vector
		inline int32_t get_learn_p(int32_t offset) const
		{
			return learn_p[offset];
		}

		/// get entry out of learn_q vector
		inline int32_t get_learn_q(int32_t offset) const
		{
			return learn_q[offset];
		}

		/// get entry out of const_a matrix
		inline int32_t get_const_a(int32_t line, int32_t column) const
		{
			return const_a[line*2 + column];
		}

		/// get entry out of const_b matrix
		inline int32_t get_const_b(int32_t line, int32_t column) const
		{
			return const_b[line*2 + column];
		}

		/// get entry out of const_p vector
		inline int32_t get_const_p(int32_t offset) const
		{
			return const_p[offset];
		}

		/// get entry out of const_q vector
		inline int32_t get_const_q(int32_t offset) const
		{
			return const_q[offset];
		}

		/// get value out of const_a_val vector
		inline float64_t get_const_a_val(int32_t line) const
		{
			return const_a_val[line];
		}

		/// get value out of const_b_val vector
		inline float64_t get_const_b_val(int32_t line) const
		{
			return const_b_val[line];
		}

		/// get value out of const_p_val vector
		inline float64_t get_const_p_val(int32_t offset) const
		{
			return const_p_val[offset];
		}

		/// get value out of const_q_val vector
		inline float64_t get_const_q_val(int32_t offset) const
		{
			return const_q_val[offset];
		}
#ifdef FIX_POS
		/// get value out of fix_pos_state array
		inline char get_fix_pos_state(int32_t pos, T_STATES state, T_STATES num_states)
		{
#ifdef HMM_DEBUG
			if ((pos<0)||(pos*num_states+state>65336))
				SG_DEBUG("index out of range in get_fix_pos_state(%i,%i,%i) \n", pos,state,num_states)
#endif
			return fix_pos_state[pos*num_states+state] ;
		}
#endif
		//@}

		/**@name write access functions
		 * For learn and const arrays
		 */
		//@{
		/// set value in learn_a matrix
		inline void set_learn_a(int32_t offset, int32_t value)
		{
			learn_a[offset]=value;
		}

		/// set value in learn_b matrix
		inline void set_learn_b(int32_t offset, int32_t value)
		{
			learn_b[offset]=value;
		}

		/// set value in learn_p vector
		inline void set_learn_p(int32_t offset, int32_t value)
		{
			learn_p[offset]=value;
		}

		/// set value in learn_q vector
		inline void set_learn_q(int32_t offset, int32_t value)
		{
			learn_q[offset]=value;
		}

		/// set value in const_a matrix
		inline void set_const_a(int32_t offset, int32_t value)
		{
			const_a[offset]=value;
		}

		/// set value in const_b matrix
		inline void set_const_b(int32_t offset, int32_t value)
		{
			const_b[offset]=value;
		}

		/// set value in const_p vector
		inline void set_const_p(int32_t offset, int32_t value)
		{
			const_p[offset]=value;
		}

		/// set value in const_q vector
		inline void set_const_q(int32_t offset, int32_t value)
		{
			const_q[offset]=value;
		}

		/// set value in const_a_val vector
		inline void set_const_a_val(int32_t offset, float64_t value)
		{
			const_a_val[offset]=value;
		}

		/// set value in const_b_val vector
		inline void set_const_b_val(int32_t offset, float64_t value)
		{
			const_b_val[offset]=value;
		}

		/// set value in const_p_val vector
		inline void set_const_p_val(int32_t offset, float64_t value)
		{
			const_p_val[offset]=value;
		}

		/// set value in const_q_val vector
		inline void set_const_q_val(int32_t offset, float64_t value)
		{
			const_q_val[offset]=value;
		}
#ifdef FIX_POS
		/// set value in fix_pos_state vector
		inline void set_fix_pos_state(
			int32_t pos, T_STATES state, T_STATES num_states, char value)
		{
#ifdef HMM_DEBUG
			if ((pos<0)||(pos*num_states+state>65336))
				SG_DEBUG("index out of range in set_fix_pos_state(%i,%i,%i,%i) [%i]\n", pos,state,num_states,(int)value, pos*num_states+state)
#endif
			fix_pos_state[pos*num_states+state]=value;
			if (value==FIX_ALLOWED)
				for (int32_t i=0; i<num_states; i++)
					if (get_fix_pos_state(pos,i,num_states)==FIX_DEFAULT)
						set_fix_pos_state(pos,i,num_states,FIX_DISALLOWED) ;
		}
		//@}

		/// FIX_DISALLOWED - state is forbidden and will be penalized with DISALLOWED_PENALTY
		const static char FIX_DISALLOWED ;

		/// FIX_ALLOWED - state is allowed
		const static char FIX_ALLOWED ;

		/// FIX_DEFAULT - default value
		const static char FIX_DEFAULT ;

		/// DISALLOWED_PENALTY - states in FIX_DISALLOWED will be penalized with this value
		const static float64_t DISALLOWED_PENALTY ;
#endif
	protected:
		/**@name learn arrays.
		 * Everything that is to be learned is enumerated here.
		 * All values will be inititialized with random values
		 * and normalized to satisfy stochasticity.
		 */
		//@{
		/// transitions to be learned
		int32_t* learn_a;

		/// emissions to be learned
		int32_t* learn_b;

		/// start states to be learned
		int32_t* learn_p;

		/// end states to be learned
		int32_t* learn_q;
		//@}

		/**@name constant arrays.
		 * These arrays hold constant fields. All values that
		 * are not constant and will not be learned are initialized
		 * with 0.
		 */
		//@{
		/// transitions that have constant probability
		int32_t* const_a;

		/// emissions that have constant probability
		int32_t* const_b;

		/// start states that have constant probability
		int32_t* const_p;

		/// end states that have constant probability
		int32_t* const_q;


		/// values for transitions that have constant probability
		float64_t* const_a_val;

		/// values for emissions that have constant probability
		float64_t* const_b_val;

		/// values for start states that have constant probability
		float64_t* const_p_val;

		/// values for end states that have constant probability
		float64_t* const_q_val;

#ifdef FIX_POS
		/** states in whose the model has to be at specific times/states which the model has to avoid.
		 * only used in viterbi
		 */
		char* fix_pos_state;
#endif
		//@}
};


/** @brief Hidden Markov Model.
 *
 * Structure and Function collection.
 * This Class implements a Hidden Markov Model.
 * For a tutorial on HMMs see Rabiner et.al A Tutorial on Hidden Markov Models
 * and Selected Applications in Speech Recognition, 1989
 *
 * Several functions for tasks such as training,reading/writing models, reading observations,
 * calculation of derivatives are supplied.
 */
class CHMM : public CDistribution
{
	private:

		T_STATES trans_list_len ;
		T_STATES **trans_list_forward  ;
		T_STATES *trans_list_forward_cnt  ;
		float64_t **trans_list_forward_val ;
		T_STATES **trans_list_backward  ;
		T_STATES *trans_list_backward_cnt  ;
		bool mem_initialized ;

#ifdef USE_HMMPARALLEL_STRUCTURES

		/// Datatype that is used in parrallel computation of viterbi
		struct S_DIM_THREAD_PARAM
		{
			CHMM* hmm;
			int32_t dim;
			float64_t prob_sum;
		};

		/// Datatype that is used in parrallel baum welch model estimation
		struct S_BW_THREAD_PARAM
		{
			CHMM* hmm;
			int32_t dim_start;
			int32_t dim_stop;

			float64_t ret;

			float64_t* p_buf;
			float64_t* q_buf;
			float64_t* a_buf;
			float64_t* b_buf;
		};

		inline T_ALPHA_BETA & ALPHA_CACHE(int32_t dim) {
			return alpha_cache[dim%parallel->get_num_threads()] ; } ;
		inline T_ALPHA_BETA & BETA_CACHE(int32_t dim) {
			return beta_cache[dim%parallel->get_num_threads()] ; } ;
#ifdef USE_LOGSUMARRAY
		inline float64_t* ARRAYS(int32_t dim) {
			return arrayS[dim%parallel->get_num_threads()] ; } ;
#endif
		inline float64_t* ARRAYN1(int32_t dim) {
			return arrayN1[dim%parallel->get_num_threads()] ; } ;
		inline float64_t* ARRAYN2(int32_t dim) {
			return arrayN2[dim%parallel->get_num_threads()] ; } ;
		inline T_STATES* STATES_PER_OBSERVATION_PSI(int32_t dim) {
			return states_per_observation_psi[dim%parallel->get_num_threads()] ; } ;
		inline const T_STATES* STATES_PER_OBSERVATION_PSI(int32_t dim) const {
			return states_per_observation_psi[dim%parallel->get_num_threads()] ; } ;
		inline T_STATES* PATH(int32_t dim) {
			return path[dim%parallel->get_num_threads()] ; } ;
		inline bool & PATH_PROB_UPDATED(int32_t dim) {
			return path_prob_updated[dim%parallel->get_num_threads()] ; } ;
		inline int32_t & PATH_PROB_DIMENSION(int32_t dim) {
			return path_prob_dimension[dim%parallel->get_num_threads()] ; } ;
#else
		inline T_ALPHA_BETA & ALPHA_CACHE(int32_t /*dim*/) {
			return alpha_cache ; } ;
		inline T_ALPHA_BETA & BETA_CACHE(int32_t /*dim*/) {
			return beta_cache ; } ;
#ifdef USE_LOGSUMARRAY
		inline float64_t* ARRAYS(int32_t dim) {
			return arrayS ; } ;
#endif
		inline float64_t* ARRAYN1(int32_t /*dim*/) {
			return arrayN1 ; } ;
		inline float64_t* ARRAYN2(int32_t /*dim*/) {
			return arrayN2 ; } ;
		inline T_STATES* STATES_PER_OBSERVATION_PSI(int32_t /*dim*/) {
			return states_per_observation_psi ; } ;
		inline const T_STATES* STATES_PER_OBSERVATION_PSI(int32_t /*dim*/) const {
			return states_per_observation_psi ; } ;
		inline T_STATES* PATH(int32_t /*dim*/) {
			return path ; } ;
		inline bool & PATH_PROB_UPDATED(int32_t /*dim*/) {
			return path_prob_updated ; } ;
		inline int32_t & PATH_PROB_DIMENSION(int32_t /*dim*/) {
			return path_prob_dimension ; } ;
#endif

		/** Determines if algorithm has converged
		 * @param x value to check against y
		 * @param y value to check against x
		 */
		bool converged(float64_t x, float64_t y);

		/** Train definitions.
		 * Encapsulates Modelparameters that are constant/shall be learned.
		 * Consists of structures and access functions for learning only defined transitions and constants.
		 */

	public:
		/** default constructor  */
		CHMM();

		/**@name Constructor/Destructor and helper function
		*/
		//@{
		/** Constructor
		 * @param N number of states
		 * @param M number of emissions
		 * @param model model which holds definitions of states to be learned + consts
		 * @param PSEUDO Pseudo Value
		 */

		CHMM(
			int32_t N, int32_t M, Model* model, float64_t PSEUDO);
		CHMM(
			CStringFeatures<uint16_t>* obs, int32_t N, int32_t M,
			float64_t PSEUDO);
		CHMM(
			int32_t N, float64_t* p, float64_t* q, float64_t* a);
		CHMM(
			int32_t N, float64_t* p, float64_t* q, int32_t num_trans,
			float64_t* a_trans);

		/** Constructor - Initialization from model file.
		 * @param model_file Filehandle to a hmm model file (*.mod)
		 * @param PSEUDO Pseudo Value
		 */
		CHMM(FILE* model_file, float64_t PSEUDO);

		/// Constructor - Clone model h
		CHMM(CHMM* h);

		/// Destructor - Cleanup
		virtual ~CHMM();

		/** learn distribution
		 *
		 * @param data training data (parameter can be avoided if distance or
		 * kernel-based classifiers are used and distance/kernels are
		 * initialized with train data)
		 *
		 * @return whether training was successful
		 */
		virtual bool train(CFeatures* data=NULL);
		virtual int32_t get_num_model_parameters() { return N*(N+M+2); }
		virtual float64_t get_log_model_parameter(int32_t num_param);
		virtual float64_t get_log_derivative(int32_t num_param, int32_t num_example);
		virtual float64_t get_log_likelihood_example(int32_t num_example)
		{
			return model_probability(num_example);
		}

		/** initialization function - gets called by constructors.
		 * @param model model which holds definitions of states to be learned + consts
		 * @param PSEUDO Pseudo Value
		 * @param model_file Filehandle to a hmm model file (*.mod)
		 */
		bool initialize(Model* model, float64_t PSEUDO, FILE* model_file=NULL);
		//@}

		/// allocates memory that depends on N
		bool alloc_state_dependend_arrays();

		/// free memory that depends on N
		void free_state_dependend_arrays();

		/**@name probability functions.
		 * forward/backward/viterbi algorithm
		 */
		//@{
		/** forward algorithm.
		 * calculates Pr[O_0,O_1, ..., O_t, q_time=S_i| lambda] for 0<= time <= T-1
		 * Pr[O|lambda] for time > T
		 * @param time t
		 * @param state i
		 * @param dimension dimension of observation (observations are a matrix, where a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1}
		 */
		float64_t forward_comp(int32_t time, int32_t state, int32_t dimension);
		float64_t forward_comp_old(
			int32_t time, int32_t state, int32_t dimension);

		/** backward algorithm.
		 * calculates Pr[O_t+1,O_t+2, ..., O_T-1| q_time=S_i, lambda] for 0<= time <= T-1
		 * Pr[O|lambda] for time >= T
		 * @param time t
		 * @param state i
		 * @param dimension dimension of observation (observations are a matrix, where a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1}
		 */
		float64_t backward_comp(int32_t time, int32_t state, int32_t dimension);
		float64_t backward_comp_old(
			int32_t time, int32_t state, int32_t dimension);

		/** calculates probability of best state sequence s_0,...,s_T-1 AND path itself using viterbi algorithm.
		 * The path can be found in the array PATH(dimension)[0..T-1] afterwards
		 * @param dimension dimension of observation for which the most probable path is calculated (observations are a matrix, where a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1}
		 */
		float64_t best_path(int32_t dimension);
		inline uint16_t get_best_path_state(int32_t dim, int32_t t)
		{
			ASSERT(PATH(dim))
			return PATH(dim)[t];
		}

		/// calculates probability that observations were generated
		/// by the model using forward algorithm.
		float64_t model_probability_comp() ;

		/// inline proxy for model probability.
		inline float64_t model_probability(int32_t dimension=-1)
		{
			//for faster calculation cache model probability
			if (dimension==-1)
			{
				if (mod_prob_updated)
					return mod_prob/p_observations->get_num_vectors();
				else
					return model_probability_comp()/p_observations->get_num_vectors();
			}
			else
				return forward(p_observations->get_vector_length(dimension), 0, dimension);
		}

		/** calculates likelihood for linear model
		 * on observations in MEMORY
		 * @param dimension dimension for which probability is calculated
		 * @return model probability
		 */
		inline float64_t linear_model_probability(int32_t dimension)
		{
			float64_t lik=0;
			int32_t len=0;
			bool free_vec;
			uint16_t* o=p_observations->get_feature_vector(dimension, len, free_vec);
			float64_t* obs_b=observation_matrix_b;

			ASSERT(N==len)

			for (int32_t i=0; i<N; i++)
			{
				lik+=obs_b[*o++];
				obs_b+=M;
			}
			p_observations->free_feature_vector(o, dimension, free_vec);
			return lik;

			// sorry, the above code is the speed optimized version of :
			/*	float64_t lik=0;

				for (int32_t i=0; i<N; i++)
				lik+=get_b(i, p_observations->get_feature(dimension, i));
				return lik;
				*/
			// : that
		}

		//@}

		/**@name convergence criteria
		 */
		inline bool set_iterations(int32_t num) { iterations=num; return true; }
		inline int32_t get_iterations() { return iterations; }
		inline bool set_epsilon (float64_t eps) { epsilon=eps; return true; }
		inline float64_t get_epsilon() { return epsilon; }

		/** interface for e.g. GUIHMM to run BaumWelch or Viterbi training
		 * @param type type of BaumWelch/Viterbi training
		 */
		bool baum_welch_viterbi_train(BaumWelchViterbiType type);

		/**@name model training
		*/
		//@{
		/** uses baum-welch-algorithm to train a fully connected HMM.
		 * @param train model from which the new model is estimated
		 */
		void estimate_model_baum_welch(CHMM* train);
		void estimate_model_baum_welch_trans(CHMM* train);

#ifdef USE_HMMPARALLEL_STRUCTURES
		void ab_buf_comp(
			float64_t* p_buf, float64_t* q_buf, float64_t* a_buf,
			float64_t* b_buf, int32_t dim) ;
#else
		void estimate_model_baum_welch_old(CHMM* train);
#endif

		/** uses baum-welch-algorithm to train the defined transitions etc.
		 * @param train model from which the new model is estimated
		 */
		void estimate_model_baum_welch_defined(CHMM* train);

		/** uses viterbi training to train a fully connected HMM
		 * @param train model from which the new model is estimated
		 */
		void estimate_model_viterbi(CHMM* train);

		/** uses viterbi training to train the defined transitions etc.
		 * @param train model from which the new model is estimated
		 */
		void estimate_model_viterbi_defined(CHMM* train);

		//@}

		/// estimates linear model from observations.
		bool linear_train(bool right_align=false);

		/// compute permutation entropy
		bool permutation_entropy(int32_t window_width, int32_t sequence_number);

		/**@name output functions.*/
		//@{
		/** prints the model parameters on screen.
		 * @param verbose when false only the model probability will be printed
		 * when true the whole model will be printed additionally
		 */
		void output_model(bool verbose=false);

		/// performs output_model only for the defined transitions etc
		void output_model_defined(bool verbose=false);
		//@}


		/**@name model helper functions.*/
		//@{

		/// normalize the model to satisfy stochasticity
		void normalize(bool keep_dead_states=false);

		/// increases the number of states by num_states
		/// the new a/b/p/q values are given the value default_val
		/// where 0<=default_val<=1
		void add_states(int32_t num_states, float64_t default_val=0);

		/// appends the append_model to the current hmm, i.e.
		/// two extra states are created. one is the end state of
		/// the current hmm with outputs cur_out (of size M) and
		/// the other state is the start state of the append_model.
		/// transition probability from state 1 to states 1 is 1
		bool append_model(
			CHMM* append_model, float64_t* cur_out, float64_t* app_out);

		/// appends the append_model to the current hmm, here
		/// no extra states are created. former q_i are multiplied by q_ji
		/// to give the a_ij from the current hmm to the append_model
		bool append_model(CHMM* append_model);

		/// set any model parameter with probability smaller than value to ZERO
		void chop(float64_t value);

		/// convert model to log probabilities
		void convert_to_log();

		/// init model with random values
		void init_model_random();

		/** init model according to const_x, learn_x.
		 * first model is initialized with 0 for all parameters
		 * then parameters in learn_x are initialized with random values
		 * finally const_x parameters are set and model is normalized.
		 */
		void init_model_defined();

		/// initializes model with log(PSEUDO)
		void clear_model();

		/// initializes only parameters in learn_x with log(PSEUDO)
		void clear_model_defined();

		/// copies the the modelparameters from l
		void copy_model(CHMM* l);

		/** invalidates all caches.
		 * this function has to be called when direct changes to the model have been made.
		 * this is necessary for the forward/backward/viterbi algorithms to not work with old tables
		 */
		void invalidate_model();

		/** get status
		 * @return true if everything is ok, else false
		 */
		inline bool get_status() const
		{
			return status;
		}

		/// returns current pseudo value
		inline float64_t get_pseudo() const
		{
			return PSEUDO ;
		}

		/// sets current pseudo value
		inline void set_pseudo(float64_t pseudo)
		{
			PSEUDO=pseudo ;
		}

#ifdef USE_HMMPARALLEL_STRUCTURES
		static void* bw_dim_prefetch(void * params);
		static void* bw_single_dim_prefetch(void * params);
		static void* vit_dim_prefetch(void * params);
#endif

#ifdef FIX_POS
		/** access function to set value in fix_pos_state vector in underlying model
		 * @see Model
		 */
		inline bool set_fix_pos_state(int32_t pos, T_STATES state, char value)
		{
			if (!model)
				return false ;
			model->set_fix_pos_state(pos, state, N, value) ;
			return true ;
		} ;
#endif
		//@}

		/** observation functions
		 * set/get observation matrix
		 */
		//@{
		/** set new observations
		 * sets the observation pointer and initializes observation-dependent caches
		 * if hmm is given, then the caches of the model hmm are used
		 */
		void set_observations(CStringFeatures<uint16_t>* obs, CHMM* hmm=NULL);

		/** set new observations
		 * only set the observation pointer and drop caches if there were any
		 */
		void set_observation_nocache(CStringFeatures<uint16_t>* obs);

		/// return observation pointer
		inline CStringFeatures<uint16_t>* get_observations()
		{
			SG_REF(p_observations);
			return p_observations;
		}
		//@}

		/**@name load/save functions.
		 * for observations/model/traindefinitions
		 */
		//@{
		/** read definitions file (learn_x,const_x) used for training.
		 * -format specs: definition_file (train.def)
		 % HMM-TRAIN - specification
		 % learn_a - elements in state_transition_matrix to be learned
		 % learn_b - elements in oberservation_per_state_matrix to be learned
		 %			note: each line stands for
		 %				state, observation(0), observation(1)...observation(NOW)
		 % learn_p - elements in initial distribution to be learned
		 % learn_q - elements in the end-state distribution to be learned
		 %
		 % const_x - specifies initial values of elements
		 %				rest is assumed to be 0.0
		 %
		 %	NOTE: IMPLICIT DEFINES:
		 %		define A 0
		 %		define C 1
		 %		define G 2
		 %		define T 3

		 learn_a=[ [int32_t,int32_t];
		 [int32_t,int32_t];
		 [int32_t,int32_t];
		 ........
		 [int32_t,int32_t];
		 [-1,-1];
		 ];

		 learn_b=[ [int32_t,int32_t,int32_t,...,int32_t];
		 [int32_t,int32_t,int32_t,...,int32_t];
		 [int32_t,int32_t,int32_t,...,int32_t];
		 ........
		 [int32_t,int32_t,int32_t,...,int32_t];
		 [-1,-1];
		 ];

		 learn_p= [ int32_t, ... , int32_t, -1 ];

		 learn_q= [ int32_t, ... , int32_t, -1 ];


		 const_a=[ [int32_t,int32_t,float64_t];
		 [int32_t,int32_t,float64_t];
		 [int32_t,int32_t,float64_t];
		 ........
		 [int32_t,int32_t,float64_t];
		 [-1,-1,-1];
		 ];

		 const_b=[ [int32_t,int32_t,int32_t,...,int32_t,float64_t];
		 [int32_t,int32_t,int32_t,...,int32_t,float64_t];
		 [int32_t,int32_t,int32_t,...,int32_t,<DOUBLE];
		 ........
		 [int32_t,int32_t,int32_t,...,int32_t,float64_t];
		 [-1,-1,-1];
		 ];

		 const_p[]=[ [int32_t, float64_t], ... , [int32_t,float64_t], [-1,-1] ];
		 const_q[]=[ [int32_t, float64_t], ... , [int32_t,float64_t], [-1,-1] ];

		 * @param file filehandle to definitions file
		 * @param verbose true for verbose messages
		 * @param initialize true to initialize to underlying HMM
		 */
		bool load_definitions(FILE* file, bool verbose, bool initialize=true);

		/** read model from file.
		 -format specs: model_file (model.hmm)
		 % HMM - specification
		 % N  - number of states
		 % M  - number of observation_tokens
		 % a is state_transition_matrix
		 % size(a)= [N,N]
		 %
		 % b is observation_per_state_matrix
		 % size(b)= [N,M]
		 %
		 % p is initial distribution
		 % size(p)= [1, N]

		 N=int32_t;
		 M=int32_t;

		 p=[float64_t,float64_t...float64_t];
		 q=[float64_t,float64_t...float64_t];

		 a=[ [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 ];

		 b=[ [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 [float64_t,float64_t...float64_t];
		 ];
		 * @param file filehandle to model file
		 */
		bool load_model(FILE* file);

		/** save model to file.
		 * @param file filehandle to model file
		 */
		bool save_model(FILE* file);

		/** save model derivatives to file in ascii format.
		 * @param file filehandle
		 */
		bool save_model_derivatives(FILE* file);

		/** save model derivatives to file in binary format.
		 * @param file filehandle
		 */
		bool save_model_derivatives_bin(FILE* file);

		/** save model in binary format.
		 * @param file filehandle
		 */
		bool save_model_bin(FILE* file);

		/// numerically check whether derivates were calculated right
		bool check_model_derivatives() ;
		bool check_model_derivatives_combined() ;

		/** get viterbi path and path probability
		 * @param dim dimension for which to obtain best path
		 * @param prob likelihood of path
		 * @return viterbi path
		 */
		T_STATES* get_path(int32_t dim, float64_t& prob);

		/** save viterbi path in ascii format
		 * @param file filehandle
		 */
		bool save_path(FILE* file);

		/** save viterbi path in ascii format
		 * @param file filehandle
		 */
		bool save_path_derivatives(FILE* file);

		/** save viterbi path in binary format
		 * @param file filehandle
		 */
		bool save_path_derivatives_bin(FILE* file);

#ifdef USE_HMMDEBUG
		/// numerically check whether derivates were calculated right
		bool check_path_derivatives() ;
#endif //USE_HMMDEBUG

		/** save model probability in binary format
		 * @param file filehandle
		 */
		bool save_likelihood_bin(FILE* file);

		/** save model probability in ascii format
		 * @param file filehandle
		 */
		bool save_likelihood(FILE* file);
		//@}

		/**@name access functions for model parameters
		 * for all the arrays a,b,p,q,A,B,psi
		 * and scalar model parameters like N,M
		 */
		//@{

		/// access function for number of states N
		inline T_STATES get_N() const { return N ; }

		/// access function for number of observations M
		inline int32_t get_M() const { return M ; }

		/** access function for probability of end states
		 * @param offset index 0...N-1
		 * @param value value to be set
		 */
		inline void set_q(T_STATES offset, float64_t value)
		{
#ifdef HMM_DEBUG
			if (offset>=N)
				SG_DEBUG("index out of range in set_q(%i,%e) [%i]\n", offset,value,N)
#endif
			end_state_distribution_q[offset]=value;
		}

		/** access function for probability of first state
		 * @param offset index 0...N-1
		 * @param value value to be set
		 */
		inline void set_p(T_STATES offset, float64_t value)
		{
#ifdef HMM_DEBUG
			if (offset>=N)
				SG_DEBUG("index out of range in set_p(%i,.) [%i]\n", offset,N)
#endif
			initial_state_distribution_p[offset]=value;
		}

		/** access function for matrix A
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...N-1
		 * @param value value to be set
		 */
		inline void set_A(T_STATES line_, T_STATES column, float64_t value)
		{
#ifdef HMM_DEBUG
			if ((line_>N)||(column>N))
				SG_DEBUG("index out of range in set_A(%i,%i,.) [%i,%i]\n",line_,column,N,N)
#endif
			transition_matrix_A[line_+column*N]=value;
		}

		/** access function for matrix a
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...N-1
		 * @param value value to be set
		 */
		inline void set_a(T_STATES line_, T_STATES column, float64_t value)
		{
#ifdef HMM_DEBUG
			if ((line_>N)||(column>N))
				SG_DEBUG("index out of range in set_a(%i,%i,.) [%i,%i]\n",line_,column,N,N)
#endif
			transition_matrix_a[line_+column*N]=value; // look also best_path!
		}

		/** access function for matrix B
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...M-1
		 * @param value value to be set
		 */
		inline void set_B(T_STATES line_, uint16_t column, float64_t value)
		{
#ifdef HMM_DEBUG
			if ((line_>=N)||(column>=M))
				SG_DEBUG("index out of range in set_B(%i,%i) [%i,%i]\n", line_, column,N,M)
#endif
			observation_matrix_B[line_*M+column]=value;
		}

		/** access function for matrix b
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...M-1
		 * @param value value to be set
		 */
		inline void set_b(T_STATES line_, uint16_t column, float64_t value)
		{
#ifdef HMM_DEBUG
			if ((line_>=N)||(column>=M))
				SG_DEBUG("index out of range in set_b(%i,%i) [%i,%i]\n", line_, column,N,M)
#endif
			observation_matrix_b[line_*M+column]=value;
		}

		/** access function for backtracking table psi
		 * @param time time 0...T-1
		 * @param state state 0...N-1
		 * @param value value to be set
		 * @param dimension dimension of observations 0...DIMENSION-1
		 */
		inline void set_psi(
			int32_t time, T_STATES state, T_STATES value, int32_t dimension)
		{
#ifdef HMM_DEBUG
			if ((time>=p_observations->get_max_vector_length())||(state>N))
				SG_DEBUG("index out of range in set_psi(%i,%i,.) [%i,%i]\n",time,state,p_observations->get_max_vector_length(),N)
#endif
			STATES_PER_OBSERVATION_PSI(dimension)[time*N+state]=value;
		}

		/** access function for probability of end states
		 * @param offset index 0...N-1
		 * @return value at offset
		 */
		inline float64_t get_q(T_STATES offset) const
		{
#ifdef HMM_DEBUG
			if (offset>=N)
				SG_DEBUG("index out of range in %e=get_q(%i) [%i]\n", end_state_distribution_q[offset],offset,N)
#endif
			return end_state_distribution_q[offset];
		}

		/** access function for probability of initial states
		 * @param offset index 0...N-1
		 * @return value at offset
		 */
		inline float64_t get_p(T_STATES offset) const
		{
#ifdef HMM_DEBUG
			if (offset>=N)
				SG_DEBUG("index out of range in get_p(%i,.) [%i]\n", offset,N)
#endif
			return initial_state_distribution_p[offset];
		}

		/** access function for matrix A
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...N-1
		 * @return value at position line colum
		 */
		inline float64_t get_A(T_STATES line_, T_STATES column) const
		{
#ifdef HMM_DEBUG
			if ((line_>N)||(column>N))
				SG_DEBUG("index out of range in get_A(%i,%i) [%i,%i]\n",line_,column,N,N)
#endif
			return transition_matrix_A[line_+column*N];
		}

		/** access function for matrix a
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...N-1
		 * @return value at position line colum
		 */
		inline float64_t get_a(T_STATES line_, T_STATES column) const
		{
#ifdef HMM_DEBUG
			if ((line_>N)||(column>N))
				SG_DEBUG("index out of range in get_a(%i,%i) [%i,%i]\n",line_,column,N,N)
#endif
			return transition_matrix_a[line_+column*N]; // look also best_path()!
		}

		/** access function for matrix B
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...M-1
		 * @return value at position line colum
		 */
		inline float64_t get_B(T_STATES line_, uint16_t column) const
		{
#ifdef HMM_DEBUG
			if ((line_>=N)||(column>=M))
				SG_DEBUG("index out of range in get_B(%i,%i) [%i,%i]\n", line_, column,N,M)
#endif
			return observation_matrix_B[line_*M+column];
		}

		/** access function for matrix b
		 * @param line_ row in matrix 0...N-1
		 * @param column column in matrix 0...M-1
		 * @return value at position line colum
		 */
		inline float64_t get_b(T_STATES line_, uint16_t column) const
		{
#ifdef HMM_DEBUG
			if ((line_>=N)||(column>=M))
				SG_DEBUG("index out of range in get_b(%i,%i) [%i,%i]\n", line_, column,N,M)
#endif
			//SG_PRINT("idx %d\n", line_*M+column)
			return observation_matrix_b[line_*M+column];
		}

		/** access function for backtracking table psi
		 * @param time time 0...T-1
		 * @param state state 0...N-1
		 * @param dimension dimension of observations 0...DIMENSION-1
		 * @return state at specified time and position
		 */
		inline T_STATES get_psi(
			int32_t time, T_STATES state, int32_t dimension) const
		{
#ifdef HMM_DEBUG
			if ((time>=p_observations->get_max_vector_length())||(state>N))
				SG_DEBUG("index out of range in get_psi(%i,%i) [%i,%i]\n",time,state,p_observations->get_max_vector_length(),N)
#endif
			return STATES_PER_OBSERVATION_PSI(dimension)[time*N+state];
		}

		//@}

		/** @return object name */
		virtual const char* get_name() const { return "HMM"; }

	protected:
		/**@name model specific variables.
		 * these are p,q,a,b,N,M etc
		 */
		//@{
		/// number of observation symbols eg. ACGT -> 0123
		int32_t M;

		/// number of states
		int32_t N;

		/// define pseudocounts against overfitting
		float64_t PSEUDO;

		// line number during processing input files
		int32_t line;

		/// observation matrix
		CStringFeatures<uint16_t>* p_observations;

		//train definition for HMM
		Model* model;

		/// matrix  of absolute counts of transitions
		float64_t* transition_matrix_A;

		/// matrix of absolute counts of observations within each state
		float64_t* observation_matrix_B;

		/// transition matrix
		float64_t* transition_matrix_a;

		/// initial distribution of states
		float64_t* initial_state_distribution_p;

		/// distribution of end-states
		float64_t* end_state_distribution_q;

		/// distribution of observations within each state
		float64_t* observation_matrix_b;

		/// convergence criterion iterations
		int32_t iterations;
		int32_t iteration_count;

		/// convergence criterion epsilon
		float64_t epsilon;
		int32_t conv_it;

		/// probability of best path
		float64_t all_pat_prob;

		/// probability of best path
		float64_t pat_prob;

		/// probability of model
		float64_t mod_prob;

		/// true if model probability is up to date
		bool mod_prob_updated;

		/// true if path probability is up to date
		bool all_path_prob_updated;

		/// dimension for which path_deriv was calculated
		int32_t path_deriv_dimension;

		/// true if path derivative is up to date
		bool path_deriv_updated;

		// true if model is using log likelihood
		bool loglikelihood;

		// true->ok, false->error
		bool status;

		// true->stolen from other HMMs, false->got own
		bool reused_caches;
		//@}

#ifdef USE_HMMPARALLEL_STRUCTURES
		/** array of size N*parallel.get_num_threads() for temporary calculations */
		float64_t** arrayN1 /*[parallel.get_num_threads()]*/ ;
		/** array of size N*parallel.get_num_threads() for temporary calculations */
		float64_t** arrayN2 /*[parallel.get_num_threads()]*/ ;
#else //USE_HMMPARALLEL_STRUCTURES
		/** array of size N for temporary calculations */
		float64_t* arrayN1;
		/** array of size N for temporary calculations */
		float64_t* arrayN2;
#endif //USE_HMMPARALLEL_STRUCTURES

#ifdef USE_LOGSUMARRAY
#ifdef USE_HMMPARALLEL_STRUCTURES
		/** array for for temporary calculations of log_sum */
		float64_t** arrayS /*[parallel.get_num_threads()]*/;
#else
		/** array for for temporary calculations of log_sum */
		float64_t* arrayS;
#endif // USE_HMMPARALLEL_STRUCTURES
#endif // USE_LOGSUMARRAY

#ifdef USE_HMMPARALLEL_STRUCTURES

		/// cache for forward variables can be terrible HUGE O(T*N)
		T_ALPHA_BETA* alpha_cache /*[parallel.get_num_threads()]*/ ;
		/// cache for backward variables can be terrible HUGE O(T*N)
		T_ALPHA_BETA* beta_cache /*[parallel.get_num_threads()]*/ ;

		/// backtracking table for viterbi can be terrible HUGE O(T*N)
		T_STATES** states_per_observation_psi /*[parallel.get_num_threads()]*/ ;

		/// best path (=state sequence) through model
		T_STATES** path /*[parallel.get_num_threads()]*/ ;

		/// true if path probability is up to date
		bool* path_prob_updated /*[parallel.get_num_threads()]*/;

		/// dimension for which path_prob was calculated
		int32_t* path_prob_dimension /*[parallel.get_num_threads()]*/ ;

#else //USE_HMMPARALLEL_STRUCTURES
		/// cache for forward variables can be terrible HUGE O(T*N)
		T_ALPHA_BETA alpha_cache;
		/// cache for backward variables can be terrible HUGE O(T*N)
		T_ALPHA_BETA beta_cache;

		/// backtracking table for viterbi can be terrible HUGE O(T*N)
		T_STATES* states_per_observation_psi;

		/// best path (=state sequence) through model
		T_STATES* path;

		/// true if path probability is up to date
		bool path_prob_updated;

		/// dimension for which path_prob was calculated
		int32_t path_prob_dimension;

#endif //USE_HMMPARALLEL_STRUCTURES
		//@}

		/** GOTN */
		static const int32_t GOTN;
		/** GOTM */
		static const int32_t GOTM;
		/** GOTO */
		static const int32_t GOTO;
		/** GOTa */
		static const int32_t GOTa;
		/** GOTb */
		static const int32_t GOTb;
		/** GOTp */
		static const int32_t GOTp;
		/** GOTq */
		static const int32_t GOTq;

		/** GOTlearn_a */
		static const int32_t GOTlearn_a;
		/** GOTlearn_b */
		static const int32_t GOTlearn_b;
		/** GOTlearn_p */
		static const int32_t GOTlearn_p;
		/** GOTlearn_q */
		static const int32_t GOTlearn_q;
		/** GOTconst_a */
		static const int32_t GOTconst_a;
		/** GOTconst_b */
		static const int32_t GOTconst_b;
		/** GOTconst_p */
		static const int32_t GOTconst_p;
		/** GOTconst_q */
		static const int32_t GOTconst_q;

		public:
		/**@name functions for observations
		 * management and access functions for observation matrix
		 */
		//@{

		/// calculates probability of being in state i at time t for dimension
inline float64_t state_probability(
	int32_t time, int32_t state, int32_t dimension)
{
	return forward(time, state, dimension) + backward(time, state, dimension) - model_probability(dimension);
}

/// calculates probability of being in state i at time t and state j at time t+1 for dimension
inline float64_t transition_probability(
	int32_t time, int32_t state_i, int32_t state_j, int32_t dimension)
{
	return forward(time, state_i, dimension) +
		backward(time+1, state_j, dimension) +
		get_a(state_i,state_j) + get_b(state_j,p_observations->get_feature(dimension ,time+1)) - model_probability(dimension);
}

/**@name derivatives of model probabilities.
 * computes log dp(lambda)/d lambda_i
 * @param dimension dimension for that derivatives are calculated
 * @param i,j parameter specific
 */
//@{

/** computes log dp(lambda)/d b_ij for linear model
*/
inline float64_t linear_model_derivative(
	T_STATES i, uint16_t j, int32_t dimension)
{
	float64_t der=0;

	for (int32_t k=0; k<N; k++)
	{
		if (k!=i || p_observations->get_feature(dimension, k) != j)
			der+=get_b(k, p_observations->get_feature(dimension, k));
	}

	return der;
}

/** computes log dp(lambda)/d p_i.
 * backward path downto time 0 multiplied by observing first symbol in path at state i
 */
inline float64_t model_derivative_p(T_STATES i, int32_t dimension)
{
	return backward(0,i,dimension)+get_b(i, p_observations->get_feature(dimension, 0));
}

/** computes log dp(lambda)/d q_i.
 * forward path upto time T-1
 */
inline float64_t model_derivative_q(T_STATES i, int32_t dimension)
{
	return forward(p_observations->get_vector_length(dimension)-1,i,dimension) ;
}

/// computes log dp(lambda)/d a_ij.
inline float64_t model_derivative_a(T_STATES i, T_STATES j, int32_t dimension)
{
	float64_t sum=-CMath::INFTY;
	for (int32_t t=0; t<p_observations->get_vector_length(dimension)-1; t++)
		sum= CMath::logarithmic_sum(sum, forward(t, i, dimension) + backward(t+1, j, dimension) + get_b(j, p_observations->get_feature(dimension,t+1)));

	return sum;
}


/// computes log dp(lambda)/d b_ij.
inline float64_t model_derivative_b(T_STATES i, uint16_t j, int32_t dimension)
{
	float64_t sum=-CMath::INFTY;
	for (int32_t t=0; t<p_observations->get_vector_length(dimension); t++)
	{
		if (p_observations->get_feature(dimension,t)==j)
			sum= CMath::logarithmic_sum(sum, forward(t,i,dimension)+backward(t,i,dimension)-get_b(i,p_observations->get_feature(dimension,t)));
	}
	//if (sum==-CMath::INFTY)
	// SG_DEBUG("log derivative is -inf: dim=%i, state=%i, obs=%i\n",dimension, i, j)
	return sum;
}
//@}

/**@name derivatives of path probabilities.
 * computes d log p(lambda,best_path)/d lambda_i
 * @param dimension dimension for that derivatives are calculated
 * @param i,j parameter specific
 */
//@{

///computes d log p(lambda,best_path)/d p_i
inline float64_t path_derivative_p(T_STATES i, int32_t dimension)
{
	best_path(dimension);
	return (i==PATH(dimension)[0]) ? (exp(-get_p(PATH(dimension)[0]))) : (0) ;
}

/// computes d log p(lambda,best_path)/d q_i
inline float64_t path_derivative_q(T_STATES i, int32_t dimension)
{
	best_path(dimension);
	return (i==PATH(dimension)[p_observations->get_vector_length(dimension)-1]) ? (exp(-get_q(PATH(dimension)[p_observations->get_vector_length(dimension)-1]))) : 0 ;
}

/// computes d log p(lambda,best_path)/d a_ij
inline float64_t path_derivative_a(T_STATES i, T_STATES j, int32_t dimension)
{
	prepare_path_derivative(dimension) ;
	return (get_A(i,j)==0) ? (0) : (get_A(i,j)*exp(-get_a(i,j))) ;
}

/// computes d log p(lambda,best_path)/d b_ij
inline float64_t path_derivative_b(T_STATES i, uint16_t j, int32_t dimension)
{
	prepare_path_derivative(dimension) ;
	return (get_B(i,j)==0) ? (0) : (get_B(i,j)*exp(-get_b(i,j))) ;
}

//@}


protected:
	/**@name input helper functions.
	 * for reading model/definition/observation files
	 */
	//@{
	/// put a sequence of numbers into the buffer
	bool get_numbuffer(FILE* file, char* buffer, int32_t length);

	/// expect open bracket.
	void open_bracket(FILE* file);

	/// expect closing bracket
	void close_bracket(FILE* file);

	/// expect comma or space.
	bool comma_or_space(FILE* file);

	/// parse error messages
	inline void error(int32_t p_line, const char* str)
	{
		if (p_line)
			SG_ERROR("error in line %d %s\n", p_line, str)
		else
			SG_ERROR("error %s\n", str)
	}
	//@}

	/// initialization function that is called before path_derivatives are calculated
	inline void prepare_path_derivative(int32_t dim)
	{
		if (path_deriv_updated && (path_deriv_dimension==dim))
			return ;
		int32_t i,j,t ;
		best_path(dim);
		//initialize with zeros
		for (i=0; i<N; i++)
		{
			for (j=0; j<N; j++)
				set_A(i,j, 0);
			for (j=0; j<M; j++)
				set_B(i,j, 0);
		}

		//counting occurences for A and B
		for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
		{
			set_A(PATH(dim)[t], PATH(dim)[t+1], get_A(PATH(dim)[t], PATH(dim)[t+1])+1);
			set_B(PATH(dim)[t], p_observations->get_feature(dim,t),  get_B(PATH(dim)[t], p_observations->get_feature(dim,t))+1);
		}
		set_B(PATH(dim)[p_observations->get_vector_length(dim)-1], p_observations->get_feature(dim,p_observations->get_vector_length(dim)-1),  get_B(PATH(dim)[p_observations->get_vector_length(dim)-1], p_observations->get_feature(dim,p_observations->get_vector_length(dim)-1)) + 1);
		path_deriv_dimension=dim ;
		path_deriv_updated=true ;
	} ;
	//@}

	/// inline proxies for forward pass
	inline float64_t forward(int32_t time, int32_t state, int32_t dimension)
	{
		if (time<1)
			time=0;

		if (ALPHA_CACHE(dimension).table && (dimension==ALPHA_CACHE(dimension).dimension) && ALPHA_CACHE(dimension).updated)
		{
			if (time<p_observations->get_vector_length(dimension))
				return ALPHA_CACHE(dimension).table[time*N+state];
			else
				return ALPHA_CACHE(dimension).sum;
		}
		else
			return forward_comp(time, state, dimension) ;
	}

	/// inline proxies for backward pass
	inline float64_t backward(int32_t time, int32_t state, int32_t dimension)
	{
		if (BETA_CACHE(dimension).table && (dimension==BETA_CACHE(dimension).dimension) && (BETA_CACHE(dimension).updated))
		{
			if (time<0)
				return BETA_CACHE(dimension).sum;
			if (time<p_observations->get_vector_length(dimension))
				return BETA_CACHE(dimension).table[time*N+state];
			else
				return -CMath::INFTY;
		}
		else
			return backward_comp(time, state, dimension) ;
	}

};
}
#endif
