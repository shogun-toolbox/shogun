// HMM.h: interface for the CHMM class.
//
//////////////////////////////////////////////////////////////////////

#ifndef __CHMM_H__
#define __CHMM_H__

#include "lib/Mathmatics.h"
#include "lib/common.h"
#include "lib/io.h"
#include "features/StringFeatures.h"
#include "distributions/Distribution.h"

#include <math.h>
#include <stdio.h>
#include <assert.h>

#ifdef PARALLEL
#define PARALLEL_STRUCTURES 1
#endif

class CHMM;
/**@name HMM specific types*/
//@{

/// type for alpha/beta caching table
typedef REAL T_ALPHA_BETA_TABLE;

/// type for alpha/beta table
struct T_ALPHA_BETA
{
  /// dimension for that alpha/beta table was generated
  INT dimension;
	
  /// perversely huge alpha/beta cache table
  T_ALPHA_BETA_TABLE* table;

  /// true if table is valid
  bool updated;				
  
  /// sum over all paths == model_probability for this dimension
  REAL sum;
};

/** type that is used for states.
 * Probably BYTE is enough if you have at most 256 states,
 * however WORD/long/... is also possible although you might quickly run into memory problems
 */
#ifdef BIGSTATES
typedef WORD T_STATES ;
#else
typedef BYTE T_STATES ;
#endif
typedef T_STATES* P_STATES ;

//@}

/** Hidden Markov Model.
 * Structure and Function collection.
 * This Class implements a Hidden Markov Model.
 * Several functions for tasks such as training,reading/writing models, reading observations,
 * calculation of derivatives are supplied.
 */
class CHMM : private CDistribution
{
    private:

  INT trans_list_len ;
  INT **trans_list_forward  ;
  INT *trans_list_forward_cnt  ;
  REAL **trans_list_forward_val ;
  INT **trans_list_backward  ;
  INT *trans_list_backward_cnt  ;
  bool mem_initialized ;

#ifdef PARALLEL_STRUCTURES

  INT NUM_PARALLEL ;

	/// Datatype that is used in parrallel computation of model probability
	struct S_MODEL_PROB_THREAD_PARAM
	{
	    CHMM * hmm;
	    INT dim_start;
	    INT dim_stop;

	    REAL prob_sum;
	};

	/// Datatype that is used in parrallel baum welch model estimation
	struct S_BW_THREAD_PARAM
	{
	    CHMM * hmm;
	    INT dim ;
	    INT dim_start;
	    INT dim_stop;

	    REAL ret;
	    REAL prob;

	    REAL* p_buf;
	    REAL* q_buf;
	    REAL* a_buf;
	    REAL* b_buf;
	};

	inline T_ALPHA_BETA & ALPHA_CACHE(INT dim) {
	    return alpha_cache[dim%NUM_PARALLEL] ; } ;
	inline T_ALPHA_BETA & BETA_CACHE(INT dim) {
	    return beta_cache[dim%NUM_PARALLEL] ; } ;
#ifdef LOG_SUM_ARRAY 
	inline REAL* ARRAYS(INT dim) {
	    return arrayS[dim%NUM_PARALLEL] ; } ;
#endif
	inline REAL* ARRAYN1(INT dim) {
	    return arrayN1[dim%NUM_PARALLEL] ; } ;
	inline REAL* ARRAYN2(INT dim) {
	    return arrayN2[dim%NUM_PARALLEL] ; } ;
	inline T_STATES* STATES_PER_OBSERVATION_PSI(INT dim) {
	    return states_per_observation_psi[dim%NUM_PARALLEL] ; } ;
	inline const T_STATES* STATES_PER_OBSERVATION_PSI(INT dim) const {
	    return states_per_observation_psi[dim%NUM_PARALLEL] ; } ;
	inline T_STATES* PATH(INT dim) {
	    return path[dim%NUM_PARALLEL] ; } ;
	inline bool & PATH_PROB_UPDATED(INT dim) {
	    return path_prob_updated[dim%NUM_PARALLEL] ; } ;
	inline INT & PATH_PROB_DIMENSION(INT dim) {
	    return path_prob_dimension[dim%NUM_PARALLEL] ; } ;
#else
	inline T_ALPHA_BETA & ALPHA_CACHE(INT /*dim*/) {
	    return alpha_cache ; } ;
	inline T_ALPHA_BETA & BETA_CACHE(INT /*dim*/) {
	    return beta_cache ; } ;
#ifdef LOG_SUM_ARRAY
	inline REAL* ARRAYS(INT dim) {
	    return arrayS ; } ;
#endif
	inline REAL* ARRAYN1(INT /*dim*/) {
	    return arrayN1 ; } ;
	inline REAL* ARRAYN2(INT /*dim*/) {
	    return arrayN2 ; } ;
	inline T_STATES* STATES_PER_OBSERVATION_PSI(INT /*dim*/) {
	    return states_per_observation_psi ; } ;
	inline const T_STATES* STATES_PER_OBSERVATION_PSI(INT /*dim*/) const {
	    return states_per_observation_psi ; } ;
	inline T_STATES* PATH(INT /*dim*/) {
	    return path ; } ;
	inline bool & PATH_PROB_UPDATED(INT /*dim*/) {
	    return path_prob_updated ; } ;
	inline INT & PATH_PROB_DIMENSION(INT /*dim*/) {
	    return path_prob_dimension ; } ;
#endif


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
		  math.sort(learn_a,2) ;
		}
		
		/// sorts learn_b matrix
		inline void sort_learn_b()
		{
		  math.sort(learn_b,2) ;
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
		inline REAL get_const_a_val(INT line) const
		{
			return const_a_val[line];
		}

		/// get value out of const_b_val vector
		inline REAL get_const_b_val(INT line) const 
		{
			return const_b_val[line];
		}

		/// get value out of const_p_val vector
		inline REAL get_const_p_val(INT offset) const 
		{
			return const_p_val[offset];
		}

		/// get value out of const_q_val vector
		inline REAL get_const_q_val(INT offset) const
		{
			return const_q_val[offset];
		}
#ifdef FIX_POS
		/// get value out of fix_pos_state array
		inline CHAR get_fix_pos_state(INT pos, T_STATES state, T_STATES num_states)
		{
#ifdef DEBUG
		  if ((pos<0)||(pos*num_states+state>65336))
		    CIO::message(stderr,"index out of range in get_fix_pos_state(%i,%i,%i) \n", pos,state,num_states) ;
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
		inline void set_const_a_val(INT offset, REAL value)
		{
			const_a_val[offset]=value;
		}

		/// set value in const_b_val vector
		inline void set_const_b_val(INT offset, REAL value)
		{
			const_b_val[offset]=value;
		}

		/// set value in const_p_val vector
		inline void set_const_p_val(INT offset, REAL value)
		{
			const_p_val[offset]=value;
		}

		/// set value in const_q_val vector
		inline void set_const_q_val(INT offset, REAL value)
		{
			const_q_val[offset]=value;
		}
#ifdef FIX_POS
		/// set value in fix_pos_state vector
		inline void set_fix_pos_state(INT pos, T_STATES state, T_STATES num_states, CHAR value)
		{
#ifdef DEBUG
       		  if ((pos<0)||(pos*num_states+state>65336))
		    CIO::message(stderr,"index out of range in set_fix_pos_state(%i,%i,%i,%i) [%i]\n", pos,state,num_states,(int)value, pos*num_states+state) ;
#endif
		  fix_pos_state[pos*num_states+state]=value;
		  if (value==FIX_ALLOWED)
		    for (INT i=0; i<num_states; i++)
		      if (get_fix_pos_state(pos,i,num_states)==FIX_DEFAULT)
			set_fix_pos_state(pos,i,num_states,FIX_DISALLOWED) ;
		}
		//@}

		/// FIX_DISALLOWED - state is forbidden and will be penalized with DISALLOWED_PENALTY
		const static CHAR FIX_DISALLOWED ;

		/// FIX_ALLOWED - state is allowed
		const static CHAR FIX_ALLOWED ;

		/// FIX_DEFAULT - default value 
		const static CHAR FIX_DEFAULT ;

		/// DISALLOWED_PENALTY - states in FIX_DISALLOWED will be penalized with this value
		const static REAL DISALLOWED_PENALTY ;
#endif
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
		REAL* const_a_val;

		/// values for emissions that have constant probability
		REAL* const_b_val;

		/// values for start states that have constant probability
		REAL* const_p_val;

		/// values for end states that have constant probability
		REAL* const_q_val;		

#ifdef FIX_POS
		/** states in whose the model has to be at specific times/states which the model has to avoid.
		 * only used in viterbi
		 */
		CHAR* fix_pos_state;
#endif
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
	 * @param PSEUDO Pseudo Value
	 */
	CHMM(INT N, INT M,	CModel* model, REAL PSEUDO, INT num_parallel);
	CHMM(INT N, double *p, double*q, double* a) ;
	CHMM(INT N, double *p, double*q, int num_trans, double* a_trans) ;

	/** Constructor - Initialization from model file.
	 * @param model_file Filehandle to a hmm model file (*.mod)
	 * @param PSEUDO Pseudo Value
	 */
	CHMM(FILE* model_file, REAL PSEUDO, INT num_parallel);

	/// Constructor - Clone model h
	CHMM(CHMM* h, INT num_parallel);

	/// Destructor - Cleanup
	virtual ~CHMM();
	
	virtual inline bool train()
	{
		return false;
	}

	virtual inline INT get_num_model_parameters() { return N*(N+M+2); }

	virtual REAL get_log_model_parameter(INT param_num)
	{
		if (param_num<N)
			return get_p(param_num);
		else if (param_num<2*N)
			return get_q(param_num-N);
		else if (param_num<N*(N+2))
			return transition_matrix_a[param_num-2*N];
		else if (param_num<N*(N+2+M))
			return observation_matrix_b[param_num-N*(N+2)];

		assert(false);
		return -1;
	}

	virtual REAL get_log_derivative(INT param_num, INT num_example)
	{
		if (param_num<N)
			return model_derivative_p(param_num, num_example);
		else if (param_num<2*N)
			return model_derivative_q(param_num-N, num_example);
		else if (param_num<N*(N+2))
		{
			INT k=param_num-2*N;
			INT i=(k/N)*N;
			INT j=N*N-i;
			return model_derivative_a(i,j, k);
		}
		else if (param_num<N*(N+2+M))
		{
			INT k=param_num-N*(N+2);
			INT i=(k/N)*M;
			INT j=N*M-i;
			return model_derivative_b(i,j, k);
		}

		assert(false);
		return -1;
	}

	virtual REAL get_log_likelihood_example(INT num_example)
	{
		return 0;
	}

	/** initialization function - gets called by constructors.
	 * @param N number of states
	 * @param M number of emissions
	 * @param model model which holds definitions of states to be learned + consts
	 * @param PSEUDO Pseudo Value
	 * @param model_file Filehandle to a hmm model file (*.mod)
	 */
	bool initialize(CModel* model, REAL PSEUDO, FILE* model_file=NULL);
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
	REAL forward_comp(INT time, INT state, INT dimension);
	REAL forward_comp_old(INT time, INT state, INT dimension);

	/** backward algorithm.
	 * calculates Pr[O_t+1,O_t+2, ..., O_T-1| q_time=S_i, lambda] for 0<= time <= T-1
	 * Pr[O|lambda] for time >= T
	 * @param time t
	 * @param state i
	 * @param dimension dimension of observation (observations are a matrix, where a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1} 
	 */
	REAL backward_comp(INT time, INT state, INT dimension);
	REAL backward_comp_old(INT time, INT state, INT dimension);

#ifndef NOVIT
	/** calculates probability of best state sequence s_0,...,s_T-1 AND path itself using viterbi algorithm.
	 * The path can be found in the array PATH(dimension)[0..T-1] afterwards
	 * @param dimension dimension of observation for which the most probable path is calculated (observations are a matrix, where a row stands for one dimension i.e. 0_0,O_1,...,O_{T-1} 
	 */
	REAL best_path(INT dimension);
	REAL best_path_no_b(INT max_iter, INT & best_iter, INT *my_path) ;
	void best_path_no_b_trans(INT max_iter, INT & max_best_iter, INT nbest, REAL *prob_nbest, INT *my_paths) ;
	void model_prob_no_b_trans(INT max_iter, REAL *prob_iter) ;
	
#endif

	/// calculates probability that observations were generated 
	/// by the model using forward algorithm.
	REAL model_probability_comp() ;
	
	/// inline proxy for model probability.
	inline REAL model_probability(INT dimension=-1)
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
	 * efficient implementation (for larger files >1MB)
	 * @param file filehandle to observation data
	 * @param WIDTH number of characters in a line (including <CR>)
	 * @param UPTO number of columns we are interested in
	 * @param singleline only one line is read, probability for that line is returned
 	 * @return model probability
	 */
	REAL linear_likelihood(FILE* file, INT WIDTH, INT UPTO, bool singleline=false);

	/** calculates likelihood for linear model
	 * on observations in MEMORY
	 * @param dimension dimension for which probability is calculated
 	 * @return model probability
	 */
	inline REAL linear_model_probability(INT dimension)
	{
	    REAL lik=0;
		INT len=0;
	    WORD* o=p_observations->get_feature_vector(dimension, len);
	    REAL* obs_b=observation_matrix_b;

		assert(N==len);

	    for (INT i=0; i<N; i++)
		{
			lik+=obs_b[*o++];
			obs_b+=M;
		}
	    return lik;

	  // sorry, the above code is the speed optimized version of :
	  /*	REAL lik=0;

		for (INT i=0; i<N; i++)
		    lik+=get_b(i, p_observations->get_feature(dimension, i));
		return lik;
		*/
	  // : that
	}

	//@}
	
	/**@name model training
	 */
	//@{
	/** uses baum-welch-algorithm to train a fully connected HMM.
	 * @param train model from which the new model is estimated
	 */
	void estimate_model_baum_welch(CHMM* train);
	void estimate_model_baum_welch_old(CHMM* train);
	void estimate_model_baum_welch_trans(CHMM* train);

#ifdef PARALLEL_STRUCTURES
	void ab_buf_comp(REAL* p_buf, REAL* q_buf, REAL* a_buf, REAL* b_buf, INT dim) ;
#endif
	
	/** uses baum-welch-algorithm to train the {\bf defined} transitions etc.
	 * @param train model from which the new model is estimated
	 */
	void estimate_model_baum_welch_defined(CHMM* train);

	/** uses viterbi training to train a fully connected HMM
	 * @param train model from which the new model is estimated
	 */
	void estimate_model_viterbi(CHMM* train);

	/** uses viterbi training to train the {\bf defined} transitions etc.
	 * @param train model from which the new model is estimated
	 */
	void estimate_model_viterbi_defined(CHMM* train);
	
	/** estimates linear model
	 * efficient implementation (for larger files >1MB)
	 * @param file filehandle to observation data
	 * @param WIDTH number of characters in a line (including <CR>)
	 * @param UPTO number of columns we are interested in
	 */
	bool linear_train(FILE* file, const INT WIDTH, const INT UPTO);
	//@}

	/// estimates linear model from observations.
	bool linear_train(bool right_align=false);
	
	/// compute permutation entropy
	bool permutation_entropy(INT window_width, INT sequence_number);

	/**@name output functions.*/
	//@{
	/** prints the model parameters on screen.
	 * @param verbose when false only the model probability will be printed
	 * when true the whole model will be printed additionally
	 */
	void output_model(bool verbose=false);

	/// performs output_model only for the defined transitions etc
	void output_model_defined(bool verbose=false);
#ifndef NOVIT
	/** prints the state sequence and the symbols that were most likely in each state.
	 * @param verbose when false only probability of viterbi path is printed
	 * @param from start dimension
	 * @param to end dimension
	 */
	void output_model_sequence(bool verbose=false,INT from=0,INT to=10);

	/// does not produce senseful output at the moment.
	void output_gene_positions(bool verbose=false);
#endif //NOVIT
	//@}


	/**@name model helper functions.*/
	//@{
	
	/// normalize the model to satisfy stochasticity
	void normalize(bool keep_dead_states=false);

	/// increases the number of states by num_states
	/// the new a/b/p/q values are given the value default_val
	/// where 0<=default_val<=1
	void add_states(INT num_states, REAL default_val=0);

	/// appends the append_model to the current hmm, i.e.
	/// two extra states are created. one is the end state of
	/// the current hmm with outputs cur_out (of size M) and
	/// the other state is the start state of the append_model.
	/// transition probability from state 1 to states 1 is 1
	bool append_model(CHMM* append_model, REAL* cur_out, REAL* app_out);

	/// appends the append_model to the current hmm, here
	/// no extra states are created. former q_i are multiplied by q_ji
	/// to give the a_ij from the current hmm to the append_model
	bool append_model(CHMM* append_model);

	/// set any model parameter with probability smaller than value to ZERO
	void chop(REAL value);

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
	inline REAL get_pseudo() const
	{
	    return PSEUDO ;
	}

	/// sets current pseudo value
	inline void set_pseudo(REAL pseudo) 
	{
	    PSEUDO=pseudo ;
	}

#ifdef PARALLEL_STRUCTURES
	
	static void* bw_dim_prefetch(void * params);
#ifndef NOVIT
	static void* vit_dim_prefetch(void * params);
#endif
	/** function that gets called from thread start routines.
	 * @param dim dimension of observation
	 * @param bw true for model_probability false for best_path
	 */
	REAL prefetch(INT dim, bool bw, REAL* p_buff=NULL, REAL* q_buf=NULL, REAL* a_buf=NULL, REAL* b_buf=NULL) ;
#endif

#ifdef FIX_POS
	/** access function to set value in fix_pos_state vector in underlying model 
	 * @see CModel
	 */
	inline bool set_fix_pos_state(INT pos, T_STATES state, CHAR value)
	  {
	    if (!model)
	      return false ;
	    model->set_fix_pos_state(pos, state, N, value) ;
	    return true ;
	  } ;
#endif	
	//@}

	/**@observation functions
	 * set/get observation matrix
	 */
	//@{
	/** set new observations
	 * sets the observation pointer and initializes observation-dependent caches
	 * if lambda is given, then the caches of the model lambda are used
	 */
	void set_observations(CStringFeatures<WORD>* obs, CHMM* lambda=NULL);

	/** set new observations
	 * only set the observation pointer and drop caches if there were any
	 */
	void set_observation_nocache(CStringFeatures<WORD>* obs);

	/// return observation pointer
	inline CStringFeatures<WORD>* get_observations()
	{
	    return p_observations;
	}
	//@}
	
	/**@name load/save functions.
	 * for observations/model/traindefinitions
	 */
	//@{
	/** read definitions file (learn_x,const_x) used for training.
	 * \begin{verbatim}
	   -format specs: definition_file (train.def)
		% HMM-TRAIN - specification
		% learn_a - elements in state_transition_matrix to be learned
		% learn_b - elements in oberservation_per_state_matrix to be learned
		%			note: each line stands for 
		%				<state>, <observation(0)>, observation(1)...observation(NOW)>
		% learn_p - elements in initial distribution to be learned
		% learn_q - elements in the end-state distribution to be learned
		%
		% const_x - specifies initial values of elements
		%				rest is assumed to be 0.0
		%
		%	NOTE: IMPLICIT DEFINES:
		%		#define A 0
		%		#define C 1
		%		#define G 2
		%		#define T 3
		%

		learn_a=[ [<INT>,<INT>]; 
			  [<INT>,<INT>]; 
			  [<INT>,<INT>]; 
		 	    ........
			  [<INT>,<INT>]; 
			  [-1,-1];
			];

		learn_b=[ [<INT>,<INT>,<INT>,...,<INT>]; 
			  [<INT>,<INT>,<INT>,...,<INT>]; 
			  [<INT>,<INT>,<INT>,...,<INT>]; 
				........
			  [<INT>,<INT>,<INT>,...,<INT>]; 
			  [-1,-1];
			];

		learn_p= [ <INT>, ... , <INT>, -1 ];

		learn_q= [ <INT>, ... , <INT>, -1 ];


		const_a=[ [<INT>,<INT>,<DOUBLE>]; 
			  [<INT>,<INT>,<DOUBLE>]; 
			  [<INT>,<INT>,<DOUBLE>]; 
				........
			  [<INT>,<INT>,<DOUBLE>]; 
			  [-1,-1,-1];
			];

		const_b=[ [<INT>,<INT>,<INT>,...,<INT>,<DOUBLE>]; 
			  [<INT>,<INT>,<INT>,...,<INT>,<DOUBLE>]; 
			  [<INT>,<INT>,<INT>,...,<INT>,<DOUBLE]; 
				........
			  [<INT>,<INT>,<INT>,...,<INT>,<DOUBLE>]; 
			  [-1,-1,-1];
			];

		const_p[]=[ [<INT>, <DOUBLE>], ... , [<INT>,<DOUBLE>], [-1,-1] ];
		const_q[]=[ [<INT>, <DOUBLE>], ... , [<INT>,<DOUBLE>], [-1,-1] ];
	\end{verbatim}	
	 * @param file filehandle to definitions file
	 * @param verbose true for verbose messages
	 * @param initialize true to initialize to underlying HMM
	 */
	bool load_definitions(FILE* file, bool verbose, bool initialize=true);

	/** read model from file.
	 * \begin{verbatim}
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

		N=<INT>;	
		M=<INT>;

		p=[<REAL>,<REAL>...<DOUBLE>];
		q=[<DOUBLE>,<DOUBLE>...<DOUBLE>];

		a=[ [<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
		  ];

		b=[ [<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
			[<DOUBLE>,<DOUBLE>...<DOUBLE>];
		  ];
	\end{verbatim}
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
#ifndef NOVIT
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
	
	/// numerically check whether derivates were calculated right
	bool check_path_derivatives() ;
#endif //NOVIT

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
	inline void set_q(T_STATES offset, REAL value)
	{
#ifdef DEBUG
	  if (offset>=N)
	    CIO::message(stderr,"index out of range in set_q(%i,%e) [%i]\n", offset,value,N) ;
#endif
		end_state_distribution_q[offset]=value;
	}

	/** access function for probability of first state
	 * @param offset index 0...N-1
	 * @param value value to be set
	 */
	inline void set_p(T_STATES offset, REAL value)
	{
#ifdef DEBUG
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
	inline void set_A(T_STATES line_, T_STATES column, REAL value)
	{
#ifdef DEBUG
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
	inline void set_a(T_STATES line_, T_STATES column, REAL value)
	{
#ifdef DEBUG
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
	inline void set_B(T_STATES line_, WORD column, REAL value)
	{
#ifdef DEBUG
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
	inline void set_b(T_STATES line_, WORD column, REAL value)
	{
#ifdef DEBUG
	  if ((line_>=N)||(column>=M))
	    CIO::message(stderr,"index out of range in set_b(%i,%i) [%i,%i]\n", line_, column,N,M) ;
#endif
		observation_matrix_b[line_*M+column]=value;
	}

#ifndef NOVIT
	/** access function for backtracking table psi
	 * @param time time 0...T-1
	 * @param state state 0...N-1
	 * @param value value to be set
	 * @param dimension dimension of observations 0...DIMENSION-1
	 */
	inline void set_psi(INT time, T_STATES state, T_STATES value, INT dimension)
	{
#ifdef DEBUG
	  if ((time>=p_observations->get_max_vector_length())||(state>N))
	    CIO::message(stderr,"index out of range in set_psi(%i,%i,.) [%i,%i]\n",time,state,p_observations->get_max_vector_length(),N) ;
#endif
	  STATES_PER_OBSERVATION_PSI(dimension)[time*N+state]=value;
	}
#endif // NOVIT

	/** access function for probability of end states
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline REAL get_q(T_STATES offset) const 
	{
#ifdef DEBUG
	  if (offset>=N)
	    CIO::message(stderr,"index out of range in %e=get_q(%i) [%i]\n", end_state_distribution_q[offset],offset,N) ;
#endif
		return end_state_distribution_q[offset];
	}

	/** access function for probability of initial states
	 * @param offset index 0...N-1
	 * @return value at offset
	 */
	inline REAL get_p(T_STATES offset) const 
	{
#ifdef DEBUG
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
	inline REAL get_A(T_STATES line_, T_STATES column) const
	{
#ifdef DEBUG
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
	inline REAL get_a(T_STATES line_, T_STATES column) const
	{
#ifdef DEBUG
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
	inline REAL get_B(T_STATES line_, WORD column) const
	{
#ifdef DEBUG
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
	inline REAL get_b(T_STATES line_, WORD column) const 
	{
#ifdef DEBUG
	  if ((line_>=N)||(column>=M))
	    CIO::message(stderr,"index out of range in get_b(%i,%i) [%i,%i]\n", line_, column,N,M) ;
#endif
	  return observation_matrix_b[line_*M+column];
	}

#ifndef NOVIT
	/** access function for backtracking table psi
	 * @param time time 0...T-1
	 * @param state state 0...N-1
	 * @param dimension dimension of observations 0...DIMENSION-1
	 * @return state at specified time and position
	 */
	inline T_STATES get_psi(INT time, T_STATES state, INT dimension) const
	{
#ifdef DEBUG
	  if ((time>=p_observations->get_max_vector_length())||(state>N))
	    CIO::message(stderr,"index out of range in get_psi(%i,%i) [%i,%i]\n",time,state,p_observations->get_max_vector_length(),N) ;
#endif
	  return STATES_PER_OBSERVATION_PSI(dimension)[time*N+state];
	}
#endif //NOVIT
	//@}
protected:
	/**@name model specific variables.
	 * these are p,q,a,b,N,M etc 
	 */
	//@{
	/// number of observation symbols eg. ACGT -> 0123
	INT M;

	/// number of states
	INT N;

	/// define pseudocounts against overfitting
	REAL PSEUDO;

	// line number during processing input files
	INT line;
	
	/// observation matrix
	CStringFeatures<WORD>* p_observations;

	//train definition for HMM
	CModel* model;

	/// matrix  of absolute counts of transitions 
	REAL* transition_matrix_A;

	/// matrix of absolute counts of observations within each state
	REAL* observation_matrix_B;

	/// transition matrix 
	REAL* transition_matrix_a;

	/// initial distribution of states
	REAL* initial_state_distribution_p;

	/// distribution of end-states
	REAL* end_state_distribution_q;		

	/// distribution of observations within each state
	REAL* observation_matrix_b;	

#ifndef NOVIT		
	/// probability of best path
	REAL all_pat_prob; 

	/// probability of best path
	REAL pat_prob;	
#endif // NOVIT
	/// probability of model
	REAL mod_prob;	

	/// true if model probability is up to date
	bool mod_prob_updated;	
#ifndef NOVIT
	/// true if path probability is up to date
	bool all_path_prob_updated;	
	
	/// dimension for which path_deriv was calculated
	INT path_deriv_dimension;
	
	/// true if path derivative is up to date
	bool path_deriv_updated;
#endif // NOVIT
	
	// true if model is using log likelihood
	bool loglikelihood;		

	// true->ok, false->error
	bool status;			

	// true->stolen from other HMMs, false->got own
	bool reused_caches;
	//@}
	
#ifdef PARALLEL_STRUCTURES
	// array of size N*NUM_PARALLEL for temporary calculations
	REAL** arrayN1 /*[NUM_PARALLEL]*/ ;
	// array of size N*NUM_PARRALEL for temporary calculations
	REAL** arrayN2 /*[NUM_PARALLEL]*/ ;
#else //PARALLEL_STRUCTURES
	// array of size N for temporary calculations
	REAL* arrayN1;
	// array of size N for temporary calculations
	REAL* arrayN2;
#endif //PARALLEL_STRUCTURES

#ifdef LOG_SUM_ARRAY
#ifdef PARALLEL_STRUCTURES
	// array for for temporary calculations of log_sum
	REAL** arrayS /*[NUM_PARALLEL]*/;
#else
	// array for for temporary calculations of log_sum
	REAL* arrayS;
#endif // PARALLEL_STRUCTURES
#endif // LOG_SUM_ARRAY

#ifdef PARALLEL_STRUCTURES

	/// cache for forward variables can be terrible HUGE O(T*N)
	T_ALPHA_BETA *alpha_cache /*[NUM_PARALLEL]*/ ;
	/// cache for backward variables can be terrible HUGE O(T*N)
	T_ALPHA_BETA *beta_cache /*[NUM_PARALLEL]*/ ;

#ifndef NOVIT
	/// backtracking table for viterbi can be terrible HUGE O(T*N)
	T_STATES** states_per_observation_psi /*[NUM_PARALLEL]*/ ;

	/// best path (=state sequence) through model
	T_STATES** path /*[NUM_PARALLEL]*/ ;
	
	/// true if path probability is up to date
	bool* path_prob_updated /*[NUM_PARALLEL]*/;
	
	/// dimension for which path_prob was calculated
	INT* path_prob_dimension /*[NUM_PARALLEL]*/ ;	
#endif //NOVIT

#else //PARALLEL_STRUCTURES
	/// cache for forward variables can be terrible HUGE O(T*N)
	T_ALPHA_BETA alpha_cache;
	/// cache for backward variables can be terrible HUGE O(T*N)
	T_ALPHA_BETA beta_cache;

#ifndef NOVIT
	/// backtracking table for viterbi can be terrible HUGE O(T*N)
	T_STATES* states_per_observation_psi;

	/// best path (=state sequence) through model
	T_STATES* path;

	/// true if path probability is up to date
	bool path_prob_updated;

	/// dimension for which path_prob was calculated
	INT path_prob_dimension;
#endif // NOVIT

#endif //PARALLEL_STRUCTURES
	//@}

	static const INT GOTN;
	static const INT GOTM;
	static const INT GOTO;
	static const INT GOTa;
	static const INT GOTb;
	static const INT GOTp;
	static const INT GOTq;

	static const INT GOTlearn_a;
	static const INT GOTlearn_b;
	static const INT GOTlearn_p;
	static const INT GOTlearn_q;
	static const INT GOTconst_a;
	static const INT GOTconst_b;
	static const INT GOTconst_p;
	static const INT GOTconst_q;

public:
	/**@name functions for observations
	 * management and access functions for observation matrix
	 */
	//@{

	/// calculates probability of being in state i at time t for dimension
	inline REAL state_probability(INT time, INT state, INT dimension)
	  {
	    return forward(time, state, dimension) + backward(time, state, dimension) - model_probability(dimension);
	  }
	
	/// calculates probability of being in state i at time t and state j at time t+1 for dimension
	inline REAL transition_probability(INT time, INT state_i, INT state_j, INT dimension)
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
	inline REAL linear_model_derivative(T_STATES i, WORD j, INT dimension)
	{
		REAL der=0;

		for (INT k=0; k<N; k++)
		{
			if (k!=i || p_observations->get_feature(dimension, k) != j)
				der+=get_b(k, p_observations->get_feature(dimension, k));
		}

		return der;
	}

	/** computes log dp(lambda)/d p_i. 
	 * backward path downto time 0 multiplied by observing first symbol in path at state i
	 */
	inline REAL model_derivative_p(T_STATES i, INT dimension)
	  {
	    return backward(0,i,dimension)+get_b(i, p_observations->get_feature(dimension, 0));		
	  }
	
	/** computes log dp(lambda)/d q_i. 
	 * forward path upto time T-1
	 */
	inline REAL model_derivative_q(T_STATES i, INT dimension)
	  {
	    return forward(p_observations->get_vector_length(dimension)-1,i,dimension) ;
	  }
	
	/// computes log dp(lambda)/d a_ij. 
	inline REAL model_derivative_a(T_STATES i, T_STATES j, INT dimension)
	  {
	    REAL sum=-math.INFTY;
	    for (INT t=0; t<p_observations->get_vector_length(dimension)-1; t++)
	      sum= math.logarithmic_sum(sum, forward(t, i, dimension) + backward(t+1, j, dimension) + get_b(j, p_observations->get_feature(dimension,t+1)));
	    
	    return sum;
	  }

/// computes log dp(lambda)/d b_ij. 
inline REAL model_derivative_b(T_STATES i, WORD j, INT dimension)
{
	REAL sum=-math.INFTY;
	for (INT t=0; t<p_observations->get_vector_length(dimension); t++)
	{
		if (p_observations->get_feature(dimension,t)==j)
			sum= math.logarithmic_sum(sum, forward(t,i,dimension)+backward(t,i,dimension)-get_b(i,p_observations->get_feature(dimension,t)));
	}
	//if (sum==-math.INFTY)
	// CIO::message("log derivative is -inf: dim=%i, state=%i, obs=%i\n",dimension, i, j) ;
	return sum;
} 
//@}

#ifndef NOVIT
	/**@name derivatives of path probabilities.
	 * computes d log p(lambda,best_path)/d lambda_i
	 * @param dimension dimension for that derivatives are calculated
	 * @param i,j parameter specific
	 */ 
	//@{
	
	///computes d log p(lambda,best_path)/d p_i
	inline REAL path_derivative_p(T_STATES i, INT dimension)
	  {
	    best_path(dimension);
	    return (i==PATH(dimension)[0]) ? (exp(-get_p(PATH(dimension)[0]))) : (0) ;
	  }
	
	/// computes d log p(lambda,best_path)/d q_i
	inline REAL path_derivative_q(T_STATES i, INT dimension)
	  {
	    best_path(dimension);
	    return (i==PATH(dimension)[p_observations->get_vector_length(dimension)-1]) ? (exp(-get_q(PATH(dimension)[p_observations->get_vector_length(dimension)-1]))) : 0 ;
	  }
	
	/// computes d log p(lambda,best_path)/d a_ij
	inline REAL path_derivative_a(T_STATES i, T_STATES j, INT dimension)
	  {
	    prepare_path_derivative(dimension) ;
	    return (get_A(i,j)==0) ? (0) : (get_A(i,j)*exp(-get_a(i,j))) ;
	  }

	 /// computes d log p(lambda,best_path)/d b_ij
	inline REAL path_derivative_b(T_STATES i, WORD j, INT dimension)
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
	bool get_numbuffer(FILE* file, CHAR* buffer, INT length);

	/// expect open bracket. 
	void open_bracket(FILE* file);
	
	/// expect closing bracket
	void close_bracket(FILE* file);
	
	/// expect comma or space.
	bool comma_or_space(FILE* file);

	/// parse error messages
	inline void error(INT line, CHAR* str)
	{
	    if (line)
		CIO::message(stderr,"error in line %d %s\n", line, str);
	    else
		CIO::message(stderr,"error %s\n", str);
	}
	//@}

	/// initialization function that is called before path_derivatives are calculated
	inline void prepare_path_derivative(INT dim)
	  {
	    if (path_deriv_updated && (path_deriv_dimension==dim))
	      return ;
	    INT i,j,t ;
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
#endif // NOVIT
	//@}
	
	/// inline proxies for forward pass
	inline REAL forward(INT time, INT state, INT dimension)
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
	      {
		/*printf("forward cache failed for %i: old entry: dim=%i, %i\n", dimension, ALPHA_CACHE(dimension).dimension, ALPHA_CACHE(dimension).updated) ;*/
		return forward_comp(time, state, dimension) ;
	      } ;
	  } ;

	/// inline proxies for backward pass
	inline REAL backward(INT time, INT state, INT dimension)
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
	      {
		/*printf("backward cache failed for %i: old entry: dim=%i, %i\n", dimension, ALPHA_CACHE(dimension).dimension, BETA_CACHE(dimension).updated) ;*/
		return backward_comp(time, state, dimension) ;
	      } ;
	  } ;

};

#endif
