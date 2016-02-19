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
#include <shogun/distributions/HMM.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parallel.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/Alphabet.h>

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>

#define VAL_MACRO log((default_value == 0) ? (CMath::random(MIN_RAND, MAX_RAND)) : default_value)
#define ARRAY_SIZE 65336

using namespace shogun;

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

const int32_t CHMM::GOTN= (1<<1);
const int32_t CHMM::GOTM= (1<<2);
const int32_t CHMM::GOTO= (1<<3);
const int32_t CHMM::GOTa= (1<<4);
const int32_t CHMM::GOTb= (1<<5);
const int32_t CHMM::GOTp= (1<<6);
const int32_t CHMM::GOTq= (1<<7);

const int32_t CHMM::GOTlearn_a= (1<<1);
const int32_t CHMM::GOTlearn_b= (1<<2);
const int32_t CHMM::GOTlearn_p= (1<<3);
const int32_t CHMM::GOTlearn_q= (1<<4);
const int32_t CHMM::GOTconst_a= (1<<5);
const int32_t CHMM::GOTconst_b= (1<<6);
const int32_t CHMM::GOTconst_p= (1<<7);
const int32_t CHMM::GOTconst_q= (1<<8);

enum E_STATE
{
	INITIAL,
	ARRAYs,
	GET_N,
	GET_M,
	GET_a,
	GET_b,
	GET_p,
	GET_q,
	GET_learn_a,
	GET_learn_b,
	GET_learn_p,
	GET_learn_q,
	GET_const_a,
	GET_const_b,
	GET_const_p,
	GET_const_q,
	COMMENT,
	END
};


#ifdef FIX_POS
const char Model::FIX_DISALLOWED=0 ;
const char Model::FIX_ALLOWED=1 ;
const char Model::FIX_DEFAULT=-1 ;
const float64_t Model::DISALLOWED_PENALTY=CMath::ALMOST_NEG_INFTY ;
#endif

Model::Model()
{
	const_a=SG_MALLOC(int, ARRAY_SIZE);				///////static fixme
	const_b=SG_MALLOC(int, ARRAY_SIZE);
	const_p=SG_MALLOC(int, ARRAY_SIZE);
	const_q=SG_MALLOC(int, ARRAY_SIZE);
	const_a_val=SG_MALLOC(float64_t, ARRAY_SIZE);			///////static fixme
	const_b_val=SG_MALLOC(float64_t, ARRAY_SIZE);
	const_p_val=SG_MALLOC(float64_t, ARRAY_SIZE);
	const_q_val=SG_MALLOC(float64_t, ARRAY_SIZE);


	learn_a=SG_MALLOC(int, ARRAY_SIZE);
	learn_b=SG_MALLOC(int, ARRAY_SIZE);
	learn_p=SG_MALLOC(int, ARRAY_SIZE);
	learn_q=SG_MALLOC(int, ARRAY_SIZE);

#ifdef FIX_POS
	fix_pos_state = SG_MALLOC(char, ARRAY_SIZE);
#endif
	for (int32_t i=0; i<ARRAY_SIZE; i++)
	{
		const_a[i]=-1 ;
		const_b[i]=-1 ;
		const_p[i]=-1 ;
		const_q[i]=-1 ;
		const_a_val[i]=1.0 ;
		const_b_val[i]=1.0 ;
		const_p_val[i]=1.0 ;
		const_q_val[i]=1.0 ;
		learn_a[i]=-1 ;
		learn_b[i]=-1 ;
		learn_p[i]=-1 ;
		learn_q[i]=-1 ;
#ifdef FIX_POS
		fix_pos_state[i] = FIX_DEFAULT ;
#endif
	} ;
}

Model::~Model()
{
	SG_FREE(const_a);
	SG_FREE(const_b);
	SG_FREE(const_p);
	SG_FREE(const_q);
	SG_FREE(const_a_val);
	SG_FREE(const_b_val);
	SG_FREE(const_p_val);
	SG_FREE(const_q_val);

	SG_FREE(learn_a);
	SG_FREE(learn_b);
	SG_FREE(learn_p);
	SG_FREE(learn_q);

#ifdef FIX_POS
	SG_FREE(fix_pos_state);
#endif

}

CHMM::CHMM()
{
	N=0;
	M=0;
	model=NULL;
	status=false;
	p_observations=NULL;
	trans_list_forward_cnt=NULL;
	trans_list_backward_cnt=NULL;
	trans_list_forward=NULL;
	trans_list_backward=NULL;
	trans_list_forward_val=NULL;
	iterations=150;
	epsilon=1e-4;
	conv_it=5;
	path=NULL;
	arrayN1=NULL;
	arrayN2=NULL;
	reused_caches=false;
	transition_matrix_a=NULL;
	observation_matrix_b=NULL;
	initial_state_distribution_p=NULL;
	end_state_distribution_q=NULL;
#ifdef USE_LOGSUMARRAY
	arrayS = NULL;
#endif
#ifdef USE_HMMPARALLEL_STRUCTURES
	this->alpha_cache=NULL;
	this->beta_cache=NULL;
	path_prob_updated = NULL;
	path_prob_dimension = NULL;
#else
	this->alpha_cache.table=NULL;
	this->beta_cache.table=NULL;
	this->alpha_cache.dimension=0;
	this->beta_cache.dimension=0;
#endif
	states_per_observation_psi=NULL;
	mem_initialized = false;
}

CHMM::CHMM(CHMM* h)
: CDistribution(), iterations(150), epsilon(1e-4), conv_it(5)
{
#ifdef USE_HMMPARALLEL_STRUCTURES
	SG_INFO("hmm is using %i separate tables\n",  parallel->get_num_threads())
#endif

	this->N=h->get_N();
	this->M=h->get_M();
	status=initialize_hmm(NULL, h->get_pseudo());
	this->copy_model(h);
	set_observations(h->p_observations);
}

CHMM::CHMM(int32_t p_N, int32_t p_M, Model* p_model, float64_t p_PSEUDO)
: CDistribution(), iterations(150), epsilon(1e-4), conv_it(5)
{
	this->N=p_N;
	this->M=p_M;
	model=NULL ;

#ifdef USE_HMMPARALLEL_STRUCTURES
	SG_INFO("hmm is using %i separate tables\n",  parallel->get_num_threads())
#endif

	status=initialize_hmm(p_model, p_PSEUDO);
}

CHMM::CHMM(
	CStringFeatures<uint16_t>* obs, int32_t p_N, int32_t p_M,
	float64_t p_PSEUDO)
: CDistribution(), iterations(150), epsilon(1e-4), conv_it(5)
{
	this->N=p_N;
	this->M=p_M;
	model=NULL ;

#ifdef USE_HMMPARALLEL_STRUCTURES
	SG_INFO("hmm is using %i separate tables\n",  parallel->get_num_threads())
#endif

	initialize_hmm(model, p_PSEUDO);
	set_observations(obs);
}

CHMM::CHMM(int32_t p_N, float64_t* p, float64_t* q, float64_t* a)
: CDistribution(), iterations(150), epsilon(1e-4), conv_it(5)
{
	this->N=p_N;
	this->M=0;
	model=NULL ;

	trans_list_forward = NULL ;
	trans_list_forward_cnt = NULL ;
	trans_list_forward_val = NULL ;
	trans_list_backward = NULL ;
	trans_list_backward_cnt = NULL ;
	trans_list_len = 0 ;
	mem_initialized = false ;

	this->transition_matrix_a=NULL;
	this->observation_matrix_b=NULL;
	this->initial_state_distribution_p=NULL;
	this->end_state_distribution_q=NULL;
	this->p_observations=NULL;
	this->reused_caches=false;

#ifdef USE_HMMPARALLEL_STRUCTURES
	this->alpha_cache=NULL;
	this->beta_cache=NULL;
#else
	this->alpha_cache.table=NULL;
	this->beta_cache.table=NULL;
	this->alpha_cache.dimension=0;
	this->beta_cache.dimension=0;
#endif

	this->states_per_observation_psi=NULL ;
	this->path=NULL;
	arrayN1=NULL ;
	arrayN2=NULL ;

	this->loglikelihood=false;
	mem_initialized = true ;

	transition_matrix_a=a ;
	observation_matrix_b=NULL ;
	initial_state_distribution_p=p ;
	end_state_distribution_q=q ;
	transition_matrix_A=NULL ;
	observation_matrix_B=NULL ;

//	this->invalidate_model();
}

CHMM::CHMM(
	int32_t p_N, float64_t* p, float64_t* q, int32_t num_trans,
	float64_t* a_trans)
: CDistribution(), iterations(150), epsilon(1e-4), conv_it(5)
{
	model=NULL ;

	this->N=p_N;
	this->M=0;

	trans_list_forward = NULL ;
	trans_list_forward_cnt = NULL ;
	trans_list_forward_val = NULL ;
	trans_list_backward = NULL ;
	trans_list_backward_cnt = NULL ;
	trans_list_len = 0 ;
	mem_initialized = false ;

	this->transition_matrix_a=NULL;
	this->observation_matrix_b=NULL;
	this->initial_state_distribution_p=NULL;
	this->end_state_distribution_q=NULL;
	this->p_observations=NULL;
	this->reused_caches=false;

#ifdef USE_HMMPARALLEL_STRUCTURES
	this->alpha_cache=NULL;
	this->beta_cache=NULL;
#else
	this->alpha_cache.table=NULL;
	this->beta_cache.table=NULL;
	this->alpha_cache.dimension=0;
	this->beta_cache.dimension=0;
#endif

	this->states_per_observation_psi=NULL ;
	this->path=NULL;
	arrayN1=NULL ;
	arrayN2=NULL ;

	this->loglikelihood=false;
	mem_initialized = true ;

	trans_list_forward_cnt=NULL ;
	trans_list_len = N ;
	trans_list_forward = SG_MALLOC(T_STATES*, N);
	trans_list_forward_val = SG_MALLOC(float64_t*, N);
	trans_list_forward_cnt = SG_MALLOC(T_STATES, N);

	int32_t start_idx=0;
	for (int32_t j=0; j<N; j++)
	{
		int32_t old_start_idx=start_idx;

		while (start_idx<num_trans && a_trans[start_idx+num_trans]==j)
		{
			start_idx++;

			if (start_idx>1 && start_idx<num_trans)
				ASSERT(a_trans[start_idx+num_trans-1]<=
					a_trans[start_idx+num_trans]);
		}

		if (start_idx>1 && start_idx<num_trans)
			ASSERT(a_trans[start_idx+num_trans-1]<=
				a_trans[start_idx+num_trans]);

		int32_t len=start_idx-old_start_idx;
		ASSERT(len>=0)

		trans_list_forward_cnt[j] = 0 ;

		if (len>0)
		{
			trans_list_forward[j]     = SG_MALLOC(T_STATES, len);
			trans_list_forward_val[j] = SG_MALLOC(float64_t, len);
		}
		else
		{
			trans_list_forward[j]     = NULL;
			trans_list_forward_val[j] = NULL;
		}
	}

	for (int32_t i=0; i<num_trans; i++)
	{
		int32_t from = (int32_t)a_trans[i+num_trans] ;
		int32_t to   = (int32_t)a_trans[i] ;
		float64_t val = a_trans[i+num_trans*2] ;

		ASSERT(from>=0 && from<N)
		ASSERT(to>=0 && to<N)

		trans_list_forward[from][trans_list_forward_cnt[from]]=to ;
		trans_list_forward_val[from][trans_list_forward_cnt[from]]=val ;
		trans_list_forward_cnt[from]++ ;
		//ASSERT(trans_list_forward_cnt[from]<3000)
	} ;

	transition_matrix_a=NULL ;
	observation_matrix_b=NULL ;
	initial_state_distribution_p=p ;
	end_state_distribution_q=q ;
	transition_matrix_A=NULL ;
	observation_matrix_B=NULL ;

//	this->invalidate_model();
}


CHMM::CHMM(FILE* model_file, float64_t p_PSEUDO)
: CDistribution(), iterations(150), epsilon(1e-4), conv_it(5)
{
#ifdef USE_HMMPARALLEL_STRUCTURES
	SG_INFO("hmm is using %i separate tables\n",  parallel->get_num_threads())
#endif

	status=initialize_hmm(NULL, p_PSEUDO, model_file);
}

CHMM::~CHMM()
{
	SG_UNREF(p_observations);

	if (trans_list_forward_cnt)
	  SG_FREE(trans_list_forward_cnt);
	if (trans_list_backward_cnt)
		SG_FREE(trans_list_backward_cnt);
	if (trans_list_forward)
	{
	    for (int32_t i=0; i<trans_list_len; i++)
			if (trans_list_forward[i])
				SG_FREE(trans_list_forward[i]);
	    SG_FREE(trans_list_forward);
	}
	if (trans_list_forward_val)
	{
	    for (int32_t i=0; i<trans_list_len; i++)
			if (trans_list_forward_val[i])
				SG_FREE(trans_list_forward_val[i]);
	    SG_FREE(trans_list_forward_val);
	}
	if (trans_list_backward)
	  {
	    for (int32_t i=0; i<trans_list_len; i++)
	      if (trans_list_backward[i])
		SG_FREE(trans_list_backward[i]);
	    SG_FREE(trans_list_backward);
	  } ;

	free_state_dependend_arrays();

	if (!reused_caches)
	{
#ifdef USE_HMMPARALLEL_STRUCTURES
		if (mem_initialized)
		{
			for (int32_t i=0; i<parallel->get_num_threads(); i++)
			{
				SG_FREE(alpha_cache[i].table);
				SG_FREE(beta_cache[i].table);
				alpha_cache[i].table=NULL;
				beta_cache[i].table=NULL;
			}
		}
		SG_FREE(alpha_cache);
		SG_FREE(beta_cache);
		alpha_cache=NULL;
		beta_cache=NULL;
#else // USE_HMMPARALLEL_STRUCTURES
		SG_FREE(alpha_cache.table);
		SG_FREE(beta_cache.table);
		alpha_cache.table=NULL;
		beta_cache.table=NULL;
#endif // USE_HMMPARALLEL_STRUCTURES

		SG_FREE(states_per_observation_psi);
		states_per_observation_psi=NULL;
	}

#ifdef USE_LOGSUMARRAY
#ifdef USE_HMMPARALLEL_STRUCTURES
	{
		if (mem_initialized)
		{
			for (int32_t i=0; i<parallel->get_num_threads(); i++)
				SG_FREE(arrayS[i]);
		}
		SG_FREE(arrayS);
	} ;
#else //USE_HMMPARALLEL_STRUCTURES
	SG_FREE(arrayS);
#endif //USE_HMMPARALLEL_STRUCTURES
#endif //USE_LOGSUMARRAY

	if (!reused_caches)
	{
#ifdef USE_HMMPARALLEL_STRUCTURES
		if (mem_initialized)
		{
			SG_FREE(path_prob_updated);
			SG_FREE(path_prob_dimension);
			for (int32_t i=0; i<parallel->get_num_threads(); i++)
				SG_FREE(path[i]);
		}
#endif //USE_HMMPARALLEL_STRUCTURES
		SG_FREE(path);
	}
}

bool CHMM::train(CFeatures* data)
{
	if (data)
	{
		if (data->get_feature_class() != C_STRING ||
				data->get_feature_type() != F_WORD)
		{
			SG_ERROR("Expected features of class string type word\n")
		}
		set_observations((CStringFeatures<uint16_t>*) data);
	}
	return baum_welch_viterbi_train(BW_NORMAL);
}

bool CHMM::alloc_state_dependend_arrays()
{

	if (!transition_matrix_a && !observation_matrix_b &&
		!initial_state_distribution_p && !end_state_distribution_q)
	{
		transition_matrix_a=SG_MALLOC(float64_t, N*N);
		observation_matrix_b=SG_MALLOC(float64_t, N*M);
		initial_state_distribution_p=SG_MALLOC(float64_t, N);
		end_state_distribution_q=SG_MALLOC(float64_t, N);
		init_model_random();
		convert_to_log();
	}

#ifdef USE_HMMPARALLEL_STRUCTURES
	for (int32_t i=0; i<parallel->get_num_threads(); i++)
	{
		arrayN1[i]=SG_MALLOC(float64_t, N);
		arrayN2[i]=SG_MALLOC(float64_t, N);
	}
#else //USE_HMMPARALLEL_STRUCTURES
	arrayN1=SG_MALLOC(float64_t, N);
	arrayN2=SG_MALLOC(float64_t, N);
#endif //USE_HMMPARALLEL_STRUCTURES

#ifdef LOG_SUMARRAY
#ifdef USE_HMMPARALLEL_STRUCTURES
	for (int32_t i=0; i<parallel->get_num_threads(); i++)
		arrayS[i]=SG_MALLOC(float64_t, (int32_t)(this->N/2+1));
#else //USE_HMMPARALLEL_STRUCTURES
	arrayS=SG_MALLOC(float64_t, (int32_t)(this->N/2+1));
#endif //USE_HMMPARALLEL_STRUCTURES
#endif //LOG_SUMARRAY
	transition_matrix_A=SG_MALLOC(float64_t, this->N*this->N);
	observation_matrix_B=SG_MALLOC(float64_t, this->N*this->M);

	if (p_observations)
	{
#ifdef USE_HMMPARALLEL_STRUCTURES
		if (alpha_cache[0].table!=NULL)
#else //USE_HMMPARALLEL_STRUCTURES
		if (alpha_cache.table!=NULL)
#endif //USE_HMMPARALLEL_STRUCTURES
			set_observations(p_observations);
		else
			set_observation_nocache(p_observations);
		SG_UNREF(p_observations);
	}

	this->invalidate_model();

	return ((transition_matrix_A != NULL) && (observation_matrix_B != NULL) &&
			(transition_matrix_a != NULL) && (observation_matrix_b != NULL) &&
			(initial_state_distribution_p != NULL) &&
			(end_state_distribution_q != NULL));
}

void CHMM::free_state_dependend_arrays()
{
#ifdef USE_HMMPARALLEL_STRUCTURES
	if (arrayN1 && arrayN2)
	{
		for (int32_t i=0; i<parallel->get_num_threads(); i++)
		{
			SG_FREE(arrayN1[i]);
			SG_FREE(arrayN2[i]);

			arrayN1[i]=NULL;
			arrayN2[i]=NULL;
		}
	}
#endif
	SG_FREE(arrayN1);
	SG_FREE(arrayN2);
	arrayN1=NULL;
	arrayN2=NULL;

	if (observation_matrix_b)
	{
		SG_FREE(transition_matrix_A);
		SG_FREE(observation_matrix_B);
		SG_FREE(transition_matrix_a);
		SG_FREE(observation_matrix_b);
		SG_FREE(initial_state_distribution_p);
		SG_FREE(end_state_distribution_q);
	} ;

	transition_matrix_A=NULL;
	observation_matrix_B=NULL;
	transition_matrix_a=NULL;
	observation_matrix_b=NULL;
	initial_state_distribution_p=NULL;
	end_state_distribution_q=NULL;
}

bool CHMM::initialize_hmm(Model* m, float64_t pseudo, FILE* modelfile)
{
	//yes optimistic
	bool files_ok=true;

	trans_list_forward = NULL ;
	trans_list_forward_cnt = NULL ;
	trans_list_forward_val = NULL ;
	trans_list_backward = NULL ;
	trans_list_backward_cnt = NULL ;
	trans_list_len = 0 ;
	mem_initialized = false ;

	this->transition_matrix_a=NULL;
	this->observation_matrix_b=NULL;
	this->initial_state_distribution_p=NULL;
	this->end_state_distribution_q=NULL;
	this->PSEUDO= pseudo;
	this->model= m;
	this->p_observations=NULL;
	this->reused_caches=false;

#ifdef USE_HMMPARALLEL_STRUCTURES
	alpha_cache=SG_MALLOC(T_ALPHA_BETA, parallel->get_num_threads());
	beta_cache=SG_MALLOC(T_ALPHA_BETA, parallel->get_num_threads());
	states_per_observation_psi=SG_MALLOC(P_STATES, parallel->get_num_threads());

	for (int32_t i=0; i<parallel->get_num_threads(); i++)
	{
		this->alpha_cache[i].table=NULL;
		this->beta_cache[i].table=NULL;
		this->alpha_cache[i].dimension=0;
		this->beta_cache[i].dimension=0;
		this->states_per_observation_psi[i]=NULL ;
	}

#else // USE_HMMPARALLEL_STRUCTURES
	this->alpha_cache.table=NULL;
	this->beta_cache.table=NULL;
	this->alpha_cache.dimension=0;
	this->beta_cache.dimension=0;
	this->states_per_observation_psi=NULL ;
#endif //USE_HMMPARALLEL_STRUCTURES

	if (modelfile)
		files_ok= files_ok && load_model(modelfile);

#ifdef USE_HMMPARALLEL_STRUCTURES
	path_prob_updated=SG_MALLOC(bool, parallel->get_num_threads());
	path_prob_dimension=SG_MALLOC(int, parallel->get_num_threads());

	path=SG_MALLOC(P_STATES, parallel->get_num_threads());

	for (int32_t i=0; i<parallel->get_num_threads(); i++)
		this->path[i]=NULL;

#else // USE_HMMPARALLEL_STRUCTURES
	this->path=NULL;

#endif //USE_HMMPARALLEL_STRUCTURES

#ifdef USE_HMMPARALLEL_STRUCTURES
	arrayN1=SG_MALLOC(float64_t*, parallel->get_num_threads());
	arrayN2=SG_MALLOC(float64_t*, parallel->get_num_threads());
#endif //USE_HMMPARALLEL_STRUCTURES

#ifdef LOG_SUMARRAY
#ifdef USE_HMMPARALLEL_STRUCTURES
	arrayS=SG_MALLOC(float64_t*, parallel->get_num_threads());
#endif // USE_HMMPARALLEL_STRUCTURES
#endif //LOG_SUMARRAY

	alloc_state_dependend_arrays();

	this->loglikelihood=false;
	mem_initialized = true ;
	this->invalidate_model();

	return	((files_ok) &&
			(transition_matrix_A != NULL) && (observation_matrix_B != NULL) &&
			(transition_matrix_a != NULL) && (observation_matrix_b != NULL) && (initial_state_distribution_p != NULL) &&
			(end_state_distribution_q != NULL));
}

//------------------------------------------------------------------------------------//

//forward algorithm
//calculates Pr[O_0,O_1, ..., O_t, q_time=S_i| lambda] for 0<= time <= T-1
//Pr[O|lambda] for time > T
float64_t CHMM::forward_comp(int32_t time, int32_t state, int32_t dimension)
{
	T_ALPHA_BETA_TABLE* alpha_new;
	T_ALPHA_BETA_TABLE* alpha;
	T_ALPHA_BETA_TABLE* dummy;
	if (time<1)
		time=0;


	int32_t wanted_time=time;

	if (ALPHA_CACHE(dimension).table)
	{
		alpha=&ALPHA_CACHE(dimension).table[0];
		alpha_new=&ALPHA_CACHE(dimension).table[N];
		time=p_observations->get_vector_length(dimension)+1;
	}
	else
	{
		alpha_new=(T_ALPHA_BETA_TABLE*)ARRAYN1(dimension);
		alpha=(T_ALPHA_BETA_TABLE*)ARRAYN2(dimension);
	}

	if (time<1)
		return get_p(state) + get_b(state, p_observations->get_feature(dimension,0));
	else
	{
		//initialization	alpha_1(i)=p_i*b_i(O_1)
		for (int32_t i=0; i<N; i++)
			alpha[i] = get_p(i) + get_b(i, p_observations->get_feature(dimension,0)) ;

		//induction		alpha_t+1(j) = (sum_i=1^N alpha_t(i)a_ij) b_j(O_t+1)
		for (register int32_t t=1; t<time && t < p_observations->get_vector_length(dimension); t++)
		{

			for (int32_t j=0; j<N; j++)
			{
				register int32_t i, num = trans_list_forward_cnt[j] ;
				float64_t sum=-CMath::INFTY;
				for (i=0; i < num; i++)
				{
					int32_t ii = trans_list_forward[j][i] ;
					sum = CMath::logarithmic_sum(sum, alpha[ii] + get_a(ii,j));
				} ;

				alpha_new[j]= sum + get_b(j, p_observations->get_feature(dimension,t));
			}

			if (!ALPHA_CACHE(dimension).table)
			{
				dummy=alpha;
				alpha=alpha_new;
				alpha_new=dummy;	//switch alpha/alpha_new
			}
			else
			{
				alpha=alpha_new;
				alpha_new+=N;		//perversely pointer arithmetic
			}
		}


		if (time<p_observations->get_vector_length(dimension))
		{
			register int32_t i, num=trans_list_forward_cnt[state];
			register float64_t sum=-CMath::INFTY;
			for (i=0; i<num; i++)
			{
				int32_t ii = trans_list_forward[state][i] ;
				sum= CMath::logarithmic_sum(sum, alpha[ii] + get_a(ii, state));
			} ;

			return sum + get_b(state, p_observations->get_feature(dimension,time));
		}
		else
		{
			// termination
			register int32_t i ;
			float64_t sum ;
			sum=-CMath::INFTY;
			for (i=0; i<N; i++)						//sum over all paths
				sum=CMath::logarithmic_sum(sum, alpha[i] + get_q(i));	//to get model probability

			if (!ALPHA_CACHE(dimension).table)
				return sum;
			else
			{
				ALPHA_CACHE(dimension).dimension=dimension;
				ALPHA_CACHE(dimension).updated=true;
				ALPHA_CACHE(dimension).sum=sum;

				if (wanted_time<p_observations->get_vector_length(dimension))
					return ALPHA_CACHE(dimension).table[wanted_time*N+state];
				else
					return ALPHA_CACHE(dimension).sum;
			}
		}
	}
}


//forward algorithm
//calculates Pr[O_0,O_1, ..., O_t, q_time=S_i| lambda] for 0<= time <= T-1
//Pr[O|lambda] for time > T
float64_t CHMM::forward_comp_old(int32_t time, int32_t state, int32_t dimension)
{
	T_ALPHA_BETA_TABLE* alpha_new;
	T_ALPHA_BETA_TABLE* alpha;
	T_ALPHA_BETA_TABLE* dummy;
	if (time<1)
		time=0;

	int32_t wanted_time=time;

	if (ALPHA_CACHE(dimension).table)
	{
		alpha=&ALPHA_CACHE(dimension).table[0];
		alpha_new=&ALPHA_CACHE(dimension).table[N];
		time=p_observations->get_vector_length(dimension)+1;
	}
	else
	{
		alpha_new=(T_ALPHA_BETA_TABLE*)ARRAYN1(dimension);
		alpha=(T_ALPHA_BETA_TABLE*)ARRAYN2(dimension);
	}

	if (time<1)
		return get_p(state) + get_b(state, p_observations->get_feature(dimension,0));
	else
	{
		//initialization	alpha_1(i)=p_i*b_i(O_1)
		for (int32_t i=0; i<N; i++)
			alpha[i] = get_p(i) + get_b(i, p_observations->get_feature(dimension,0)) ;

		//induction		alpha_t+1(j) = (sum_i=1^N alpha_t(i)a_ij) b_j(O_t+1)
		for (register int32_t t=1; t<time && t < p_observations->get_vector_length(dimension); t++)
		{

			for (int32_t j=0; j<N; j++)
			{
				register int32_t i ;
#ifdef USE_LOGSUMARRAY
				for (i=0; i<(N>>1); i++)
					ARRAYS(dimension)[i]=CMath::logarithmic_sum(alpha[i<<1] + get_a(i<<1,j),
							alpha[(i<<1)+1] + get_a((i<<1)+1,j));
				if (N%2==1)
					alpha_new[j]=get_b(j, p_observations->get_feature(dimension,t))+
						CMath::logarithmic_sum(alpha[N-1]+get_a(N-1,j),
								CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1)) ;
				else
					alpha_new[j]=get_b(j, p_observations->get_feature(dimension,t))+CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1) ;
#else //USE_LOGSUMARRAY
				float64_t sum=-CMath::INFTY;
				for (i=0; i<N; i++)
					sum= CMath::logarithmic_sum(sum, alpha[i] + get_a(i,j));

				alpha_new[j]= sum + get_b(j, p_observations->get_feature(dimension,t));
#endif //USE_LOGSUMARRAY
			}

			if (!ALPHA_CACHE(dimension).table)
			{
				dummy=alpha;
				alpha=alpha_new;
				alpha_new=dummy;	//switch alpha/alpha_new
			}
			else
			{
				alpha=alpha_new;
				alpha_new+=N;		//perversely pointer arithmetic
			}
		}


		if (time<p_observations->get_vector_length(dimension))
		{
			register int32_t i;
#ifdef USE_LOGSUMARRAY
			for (i=0; i<(N>>1); i++)
				ARRAYS(dimension)[i]=CMath::logarithmic_sum(alpha[i<<1] + get_a(i<<1,state),
						alpha[(i<<1)+1] + get_a((i<<1)+1,state));
			if (N%2==1)
				return get_b(state, p_observations->get_feature(dimension,time))+
					CMath::logarithmic_sum(alpha[N-1]+get_a(N-1,state),
							CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1)) ;
			else
				return get_b(state, p_observations->get_feature(dimension,time))+CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1) ;
#else //USE_LOGSUMARRAY
			register float64_t sum=-CMath::INFTY;
			for (i=0; i<N; i++)
				sum= CMath::logarithmic_sum(sum, alpha[i] + get_a(i, state));

			return sum + get_b(state, p_observations->get_feature(dimension,time));
#endif //USE_LOGSUMARRAY
		}
		else
		{
			// termination
			register int32_t i ;
			float64_t sum ;
#ifdef USE_LOGSUMARRAY
			for (i=0; i<(N>>1); i++)
				ARRAYS(dimension)[i]=CMath::logarithmic_sum(alpha[i<<1] + get_q(i<<1),
						alpha[(i<<1)+1] + get_q((i<<1)+1));
			if (N%2==1)
				sum=CMath::logarithmic_sum(alpha[N-1]+get_q(N-1),
						CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1)) ;
			else
				sum=CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1) ;
#else //USE_LOGSUMARRAY
			sum=-CMath::INFTY;
			for (i=0; i<N; i++)			                      //sum over all paths
				sum=CMath::logarithmic_sum(sum, alpha[i] + get_q(i));     //to get model probability
#endif //USE_LOGSUMARRAY

			if (!ALPHA_CACHE(dimension).table)
				return sum;
			else
			{
				ALPHA_CACHE(dimension).dimension=dimension;
				ALPHA_CACHE(dimension).updated=true;
				ALPHA_CACHE(dimension).sum=sum;

				if (wanted_time<p_observations->get_vector_length(dimension))
					return ALPHA_CACHE(dimension).table[wanted_time*N+state];
				else
					return ALPHA_CACHE(dimension).sum;
			}
		}
	}
}


//backward algorithm
//calculates Pr[O_t+1,O_t+2, ..., O_T| q_time=S_i, lambda] for 0<= time <= T-1
//Pr[O|lambda] for time >= T
float64_t CHMM::backward_comp(int32_t time, int32_t state, int32_t dimension)
{
  T_ALPHA_BETA_TABLE* beta_new;
  T_ALPHA_BETA_TABLE* beta;
  T_ALPHA_BETA_TABLE* dummy;
  int32_t wanted_time=time;

  if (time<0)
    forward(time, state, dimension);

  if (BETA_CACHE(dimension).table)
    {
      beta=&BETA_CACHE(dimension).table[N*(p_observations->get_vector_length(dimension)-1)];
      beta_new=&BETA_CACHE(dimension).table[N*(p_observations->get_vector_length(dimension)-2)];
      time=-1;
    }
  else
    {
      beta_new=(T_ALPHA_BETA_TABLE*)ARRAYN1(dimension);
      beta=(T_ALPHA_BETA_TABLE*)ARRAYN2(dimension);
    }

  if (time>=p_observations->get_vector_length(dimension)-1)
    //	  return 0;
    //	else if (time==p_observations->get_vector_length(dimension)-1)
    return get_q(state);
  else
    {
      //initialization	beta_T(i)=q(i)
      for (register int32_t i=0; i<N; i++)
	beta[i]=get_q(i);

      //induction		beta_t(i) = (sum_j=1^N a_ij*b_j(O_t+1)*beta_t+1(j)
      for (register int32_t t=p_observations->get_vector_length(dimension)-1; t>time+1 && t>0; t--)
	{
	  for (register int32_t i=0; i<N; i++)
	    {
	      register int32_t j, num=trans_list_backward_cnt[i] ;
	      float64_t sum=-CMath::INFTY;
	      for (j=0; j<num; j++)
		{
		  int32_t jj = trans_list_backward[i][j] ;
		  sum= CMath::logarithmic_sum(sum, get_a(i, jj) + get_b(jj, p_observations->get_feature(dimension,t)) + beta[jj]);
		} ;
	      beta_new[i]=sum;
	    }

	  if (!BETA_CACHE(dimension).table)
	    {
	      dummy=beta;
	      beta=beta_new;
	      beta_new=dummy;	//switch beta/beta_new
	    }
	  else
	    {
	      beta=beta_new;
	      beta_new-=N;		//perversely pointer arithmetic
	    }
	}

      if (time>=0)
	{
	  register int32_t j, num=trans_list_backward_cnt[state] ;
	  float64_t sum=-CMath::INFTY;
	  for (j=0; j<num; j++)
	    {
	      int32_t jj = trans_list_backward[state][j] ;
	      sum= CMath::logarithmic_sum(sum, get_a(state, jj) + get_b(jj, p_observations->get_feature(dimension,time+1))+beta[jj]);
	    } ;
	  return sum;
	}
      else // time<0
	{
	  if (BETA_CACHE(dimension).table)
	    {
	      float64_t sum=-CMath::INFTY;
	      for (register int32_t j=0; j<N; j++)
		sum= CMath::logarithmic_sum(sum, get_p(j) + get_b(j, p_observations->get_feature(dimension,0))+beta[j]);
	      BETA_CACHE(dimension).sum=sum;
	      BETA_CACHE(dimension).dimension=dimension;
	      BETA_CACHE(dimension).updated=true;

	      if (wanted_time<p_observations->get_vector_length(dimension))
		return BETA_CACHE(dimension).table[wanted_time*N+state];
	      else
		return BETA_CACHE(dimension).sum;
	    }
	  else
	    {
	      float64_t sum=-CMath::INFTY; // apply LOG_SUM_ARRAY -- no cache ... does not make very sense anyway...
	      for (register int32_t j=0; j<N; j++)
		sum= CMath::logarithmic_sum(sum, get_p(j) + get_b(j, p_observations->get_feature(dimension,0))+beta[j]);
	      return sum;
	    }
	}
    }
}


float64_t CHMM::backward_comp_old(
	int32_t time, int32_t state, int32_t dimension)
{
	T_ALPHA_BETA_TABLE* beta_new;
	T_ALPHA_BETA_TABLE* beta;
	T_ALPHA_BETA_TABLE* dummy;
	int32_t wanted_time=time;

	if (time<0)
		forward(time, state, dimension);

	if (BETA_CACHE(dimension).table)
	{
		beta=&BETA_CACHE(dimension).table[N*(p_observations->get_vector_length(dimension)-1)];
		beta_new=&BETA_CACHE(dimension).table[N*(p_observations->get_vector_length(dimension)-2)];
		time=-1;
	}
	else
	{
		beta_new=(T_ALPHA_BETA_TABLE*)ARRAYN1(dimension);
		beta=(T_ALPHA_BETA_TABLE*)ARRAYN2(dimension);
	}

	if (time>=p_observations->get_vector_length(dimension)-1)
		//	  return 0;
		//	else if (time==p_observations->get_vector_length(dimension)-1)
		return get_q(state);
	else
	{
		//initialization	beta_T(i)=q(i)
		for (register int32_t i=0; i<N; i++)
			beta[i]=get_q(i);

		//induction		beta_t(i) = (sum_j=1^N a_ij*b_j(O_t+1)*beta_t+1(j)
		for (register int32_t t=p_observations->get_vector_length(dimension)-1; t>time+1 && t>0; t--)
		{
			for (register int32_t i=0; i<N; i++)
			{
				register int32_t j ;
#ifdef USE_LOGSUMARRAY
				for (j=0; j<(N>>1); j++)
					ARRAYS(dimension)[j]=CMath::logarithmic_sum(
							get_a(i, j<<1) + get_b(j<<1, p_observations->get_feature(dimension,t)) + beta[j<<1],
							get_a(i, (j<<1)+1) + get_b((j<<1)+1, p_observations->get_feature(dimension,t)) + beta[(j<<1)+1]);
				if (N%2==1)
					beta_new[i]=CMath::logarithmic_sum(get_a(i, N-1) + get_b(N-1, p_observations->get_feature(dimension,t)) + beta[N-1],
							CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1)) ;
				else
					beta_new[i]=CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1) ;
#else //USE_LOGSUMARRAY
				float64_t sum=-CMath::INFTY;
				for (j=0; j<N; j++)
					sum= CMath::logarithmic_sum(sum, get_a(i, j) + get_b(j, p_observations->get_feature(dimension,t)) + beta[j]);

				beta_new[i]=sum;
#endif //USE_LOGSUMARRAY
			}

			if (!BETA_CACHE(dimension).table)
			{
				dummy=beta;
				beta=beta_new;
				beta_new=dummy;	//switch beta/beta_new
			}
			else
			{
				beta=beta_new;
				beta_new-=N;		//perversely pointer arithmetic
			}
		}

		if (time>=0)
		{
			register int32_t j ;
#ifdef USE_LOGSUMARRAY
			for (j=0; j<(N>>1); j++)
				ARRAYS(dimension)[j]=CMath::logarithmic_sum(
						get_a(state, j<<1) + get_b(j<<1, p_observations->get_feature(dimension,time+1)) + beta[j<<1],
						get_a(state, (j<<1)+1) + get_b((j<<1)+1, p_observations->get_feature(dimension,time+1)) + beta[(j<<1)+1]);
			if (N%2==1)
				return CMath::logarithmic_sum(get_a(state, N-1) + get_b(N-1, p_observations->get_feature(dimension,time+1)) + beta[N-1],
						CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1)) ;
			else
				return CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1) ;
#else //USE_LOGSUMARRAY
			float64_t sum=-CMath::INFTY;
			for (j=0; j<N; j++)
				sum= CMath::logarithmic_sum(sum, get_a(state, j) + get_b(j, p_observations->get_feature(dimension,time+1))+beta[j]);

			return sum;
#endif //USE_LOGSUMARRAY
		}
		else // time<0
		{
			if (BETA_CACHE(dimension).table)
			{
#ifdef USE_LOGSUMARRAY//AAA
				for (int32_t j=0; j<(N>>1); j++)
					ARRAYS(dimension)[j]=CMath::logarithmic_sum(get_p(j<<1) + get_b(j<<1, p_observations->get_feature(dimension,0))+beta[j<<1],
							get_p((j<<1)+1) + get_b((j<<1)+1, p_observations->get_feature(dimension,0))+beta[(j<<1)+1]) ;
				if (N%2==1)
					BETA_CACHE(dimension).sum=CMath::logarithmic_sum(get_p(N-1) + get_b(N-1, p_observations->get_feature(dimension,0))+beta[N-1],
							CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1)) ;
				else
					BETA_CACHE(dimension).sum=CMath::logarithmic_sum_array(ARRAYS(dimension), N>>1) ;
#else //USE_LOGSUMARRAY
				float64_t sum=-CMath::INFTY;
				for (register int32_t j=0; j<N; j++)
					sum= CMath::logarithmic_sum(sum, get_p(j) + get_b(j, p_observations->get_feature(dimension,0))+beta[j]);
				BETA_CACHE(dimension).sum=sum;
#endif //USE_LOGSUMARRAY
				BETA_CACHE(dimension).dimension=dimension;
				BETA_CACHE(dimension).updated=true;

				if (wanted_time<p_observations->get_vector_length(dimension))
					return BETA_CACHE(dimension).table[wanted_time*N+state];
				else
					return BETA_CACHE(dimension).sum;
			}
			else
			{
				float64_t sum=-CMath::INFTY; // apply LOG_SUM_ARRAY -- no cache ... does not make very sense anyway...
				for (register int32_t j=0; j<N; j++)
					sum= CMath::logarithmic_sum(sum, get_p(j) + get_b(j, p_observations->get_feature(dimension,0))+beta[j]);
				return sum;
			}
		}
	}
}

//calculates probability  of best path through the model lambda AND path itself
//using viterbi algorithm
float64_t CHMM::best_path(int32_t dimension)
{
	if (!p_observations)
		return -1;

	if (dimension==-1)
	{
		if (!all_path_prob_updated)
		{
			SG_INFO("computing full viterbi likelihood\n")
			float64_t sum = 0 ;
			for (int32_t i=0; i<p_observations->get_num_vectors(); i++)
				sum+=best_path(i) ;
			sum /= p_observations->get_num_vectors() ;
			all_pat_prob=sum ;
			all_path_prob_updated=true ;
			return sum ;
		} else
			return all_pat_prob ;
	} ;

	if (!STATES_PER_OBSERVATION_PSI(dimension))
		return -1 ;

	if (dimension >= p_observations->get_num_vectors())
		return -1;

	if (PATH_PROB_UPDATED(dimension) && dimension==PATH_PROB_DIMENSION(dimension))
		return pat_prob;
	else
	{
		register float64_t* delta= ARRAYN2(dimension);
		register float64_t* delta_new= ARRAYN1(dimension);

		{ //initialization
			for (register int32_t i=0; i<N; i++)
			{
				delta[i]=get_p(i)+get_b(i, p_observations->get_feature(dimension,0));
				set_psi(0, i, 0, dimension);
			}
		}

#ifdef USE_PATHDEBUG
		float64_t worst=-CMath::INFTY/4 ;
#endif
		//recursion
		for (register int32_t t=1; t<p_observations->get_vector_length(dimension); t++)
		{
			register float64_t* dummy;
			register int32_t NN=N ;
			for (register int32_t j=0; j<NN; j++)
			{
				register float64_t * matrix_a=&transition_matrix_a[j*N] ; // sorry for that
				register float64_t maxj=delta[0] + matrix_a[0];
				register int32_t argmax=0;

				for (register int32_t i=1; i<NN; i++)
				{
					register float64_t temp = delta[i] + matrix_a[i];

					if (temp>maxj)
					{
						maxj=temp;
						argmax=i;
					}
				}
#ifdef FIX_POS
				if ((!model) || (model->get_fix_pos_state(t,j,NN)!=Model::FIX_DISALLOWED))
#endif
					delta_new[j]=maxj + get_b(j,p_observations->get_feature(dimension,t));
#ifdef FIX_POS
				else
					delta_new[j]=maxj + get_b(j,p_observations->get_feature(dimension,t)) + Model::DISALLOWED_PENALTY;
#endif
				set_psi(t, j, argmax, dimension);
			}

#ifdef USE_PATHDEBUG
			float64_t best=log(0) ;
			for (int32_t jj=0; jj<N; jj++)
				if (delta_new[jj]>best)
					best=delta_new[jj] ;

			if (best<-CMath::INFTY/2)
			{
				SG_DEBUG("worst case at %i: %e:%e\n", t, best, worst)
				worst=best ;
			} ;
#endif

			dummy=delta;
			delta=delta_new;
			delta_new=dummy;	//switch delta/delta_new
		}


		{ //termination
			register float64_t maxj=delta[0]+get_q(0);
			register int32_t argmax=0;

			for (register int32_t i=1; i<N; i++)
			{
				register float64_t temp=delta[i]+get_q(i);

				if (temp>maxj)
				{
					maxj=temp;
					argmax=i;
				}
			}
			pat_prob=maxj;
			PATH(dimension)[p_observations->get_vector_length(dimension)-1]=argmax;
		} ;


		{ //state sequence backtracking
			for (register int32_t t=p_observations->get_vector_length(dimension)-1; t>0; t--)
			{
				PATH(dimension)[t-1]=get_psi(t, PATH(dimension)[t], dimension);
			}
		}
		PATH_PROB_UPDATED(dimension)=true;
		PATH_PROB_DIMENSION(dimension)=dimension;
		return pat_prob ;
	}
}

#ifndef USE_HMMPARALLEL
float64_t CHMM::model_probability_comp()
{
	//for faster calculation cache model probability
	mod_prob=0 ;
	for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++) //sum in log space
		mod_prob+=forward(p_observations->get_vector_length(dim), 0, dim);

	mod_prob_updated=true;
	return mod_prob;
}

#else

float64_t CHMM::model_probability_comp()
{
	pthread_t *threads=SG_MALLOC(pthread_t, parallel->get_num_threads());
	S_BW_THREAD_PARAM *params=SG_MALLOC(S_BW_THREAD_PARAM, parallel->get_num_threads());

	SG_INFO("computing full model probablity\n")
	mod_prob=0;

	for (int32_t cpu=0; cpu<parallel->get_num_threads(); cpu++)
	{
		params[cpu].hmm=this ;
		params[cpu].dim_start= p_observations->get_num_vectors()*cpu/parallel->get_num_threads();
		params[cpu].dim_stop= p_observations->get_num_vectors()*(cpu+1)/parallel->get_num_threads();
		params[cpu].p_buf=SG_MALLOC(float64_t, N);
		params[cpu].q_buf=SG_MALLOC(float64_t, N);
		params[cpu].a_buf=SG_MALLOC(float64_t, N*N);
		params[cpu].b_buf=SG_MALLOC(float64_t, N*M);
		pthread_create(&threads[cpu], NULL, bw_dim_prefetch, (void*)&params[cpu]);
	}

	for (int32_t cpu=0; cpu<parallel->get_num_threads(); cpu++)
	{
		pthread_join(threads[cpu], NULL);
		mod_prob+=params[cpu].ret;
	}

	for (int32_t i=0; i<parallel->get_num_threads(); i++)
	{
		SG_FREE(params[i].p_buf);
		SG_FREE(params[i].q_buf);
		SG_FREE(params[i].a_buf);
		SG_FREE(params[i].b_buf);
	}

	SG_FREE(threads);
	SG_FREE(params);

	mod_prob_updated=true;
	return mod_prob;
}

void* CHMM::bw_dim_prefetch(void* params)
{
	CHMM* hmm=((S_BW_THREAD_PARAM*) params)->hmm;
	int32_t start=((S_BW_THREAD_PARAM*) params)->dim_start;
	int32_t stop=((S_BW_THREAD_PARAM*) params)->dim_stop;
	float64_t* p_buf=((S_BW_THREAD_PARAM*) params)->p_buf;
	float64_t* q_buf=((S_BW_THREAD_PARAM*) params)->q_buf;
	float64_t* a_buf=((S_BW_THREAD_PARAM*) params)->a_buf;
	float64_t* b_buf=((S_BW_THREAD_PARAM*) params)->b_buf;
	((S_BW_THREAD_PARAM*)params)->ret=0;

	for (int32_t dim=start; dim<stop; dim++)
	{
		hmm->forward_comp(hmm->p_observations->get_vector_length(dim), hmm->N-1, dim) ;
		hmm->backward_comp(hmm->p_observations->get_vector_length(dim), hmm->N-1, dim) ;
		float64_t modprob=hmm->model_probability(dim) ;
		hmm->ab_buf_comp(p_buf, q_buf, a_buf, b_buf, dim) ;
		((S_BW_THREAD_PARAM*)params)->ret+= modprob;
	}
	return NULL ;
}

void* CHMM::bw_single_dim_prefetch(void * params)
{
	CHMM* hmm=((S_BW_THREAD_PARAM*)params)->hmm ;
	int32_t dim=((S_DIM_THREAD_PARAM*)params)->dim ;
	((S_DIM_THREAD_PARAM*)params)->prob_sum = hmm->model_probability(dim);
	return NULL ;
}

void* CHMM::vit_dim_prefetch(void * params)
{
	CHMM* hmm=((S_DIM_THREAD_PARAM*)params)->hmm ;
	int32_t dim=((S_DIM_THREAD_PARAM*)params)->dim ;
	((S_DIM_THREAD_PARAM*)params)->prob_sum = hmm->best_path(dim);
	return NULL ;
}

#endif //USE_HMMPARALLEL


#ifdef USE_HMMPARALLEL

void CHMM::ab_buf_comp(
	float64_t* p_buf, float64_t* q_buf, float64_t *a_buf, float64_t* b_buf,
	int32_t dim)
{
	int32_t i,j,t ;
	float64_t a_sum;
	float64_t b_sum;

	float64_t dimmodprob=model_probability(dim);

	for (i=0; i<N; i++)
	{
		//estimate initial+end state distribution numerator
		p_buf[i]=get_p(i)+get_b(i,p_observations->get_feature(dim,0))+backward(0,i,dim) - dimmodprob;
		q_buf[i]=forward(p_observations->get_vector_length(dim)-1, i, dim)+get_q(i) - dimmodprob;

		//estimate a
		for (j=0; j<N; j++)
		{
			a_sum=-CMath::INFTY;

			for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
			{
				a_sum= CMath::logarithmic_sum(a_sum, forward(t,i,dim)+
						get_a(i,j)+get_b(j,p_observations->get_feature(dim,t+1))+backward(t+1,j,dim));
			}
			a_buf[N*i+j]=a_sum-dimmodprob ;
		}

		//estimate b
		for (j=0; j<M; j++)
		{
			b_sum=-CMath::INFTY;

			for (t=0; t<p_observations->get_vector_length(dim); t++)
			{
				if (p_observations->get_feature(dim,t)==j)
					b_sum=CMath::logarithmic_sum(b_sum, forward(t,i,dim)+backward(t, i, dim));
			}

			b_buf[M*i+j]=b_sum-dimmodprob ;
		}
	}
}

//estimates new model lambda out of lambda_train using baum welch algorithm
void CHMM::estimate_model_baum_welch(CHMM* hmm)
{
	int32_t i,j,cpu;
	float64_t fullmodprob=0;	//for all dims

	//clear actual model a,b,p,q are used as numerator
	for (i=0; i<N; i++)
	{
	  if (hmm->get_p(i)>CMath::ALMOST_NEG_INFTY)
	    set_p(i,log(PSEUDO));
	  else
	    set_p(i,hmm->get_p(i));
	  if (hmm->get_q(i)>CMath::ALMOST_NEG_INFTY)
	    set_q(i,log(PSEUDO));
	  else
	    set_q(i,hmm->get_q(i));

	  for (j=0; j<N; j++)
	    if (hmm->get_a(i,j)>CMath::ALMOST_NEG_INFTY)
	      set_a(i,j, log(PSEUDO));
	    else
	      set_a(i,j,hmm->get_a(i,j));
	  for (j=0; j<M; j++)
	    if (hmm->get_b(i,j)>CMath::ALMOST_NEG_INFTY)
	      set_b(i,j, log(PSEUDO));
	    else
	      set_b(i,j,hmm->get_b(i,j));
	}
	invalidate_model();

	int32_t num_threads = parallel->get_num_threads();

	pthread_t *threads=SG_MALLOC(pthread_t, num_threads);
	S_BW_THREAD_PARAM *params=SG_MALLOC(S_BW_THREAD_PARAM, num_threads);

	if (p_observations->get_num_vectors()<num_threads)
		num_threads=p_observations->get_num_vectors();

	for (cpu=0; cpu<num_threads; cpu++)
	{
		params[cpu].p_buf=SG_MALLOC(float64_t, N);
		params[cpu].q_buf=SG_MALLOC(float64_t, N);
		params[cpu].a_buf=SG_MALLOC(float64_t, N*N);
		params[cpu].b_buf=SG_MALLOC(float64_t, N*M);

		params[cpu].hmm=hmm;
		int32_t start = p_observations->get_num_vectors()*cpu / num_threads;
		int32_t stop=p_observations->get_num_vectors()*(cpu+1) / num_threads;

		if (cpu == parallel->get_num_threads()-1)
			stop=p_observations->get_num_vectors();

		ASSERT(start<stop)
		params[cpu].dim_start=start;
		params[cpu].dim_stop=stop;

		pthread_create(&threads[cpu], NULL, bw_dim_prefetch, &params[cpu]);
	}

	for (cpu=0; cpu<num_threads; cpu++)
	{
		pthread_join(threads[cpu], NULL);

		for (i=0; i<N; i++)
		{
			//estimate initial+end state distribution numerator
			set_p(i, CMath::logarithmic_sum(get_p(i), params[cpu].p_buf[i]));
			set_q(i, CMath::logarithmic_sum(get_q(i), params[cpu].q_buf[i]));

			//estimate numerator for a
			for (j=0; j<N; j++)
				set_a(i,j, CMath::logarithmic_sum(get_a(i,j), params[cpu].a_buf[N*i+j]));

			//estimate numerator for b
			for (j=0; j<M; j++)
				set_b(i,j, CMath::logarithmic_sum(get_b(i,j), params[cpu].b_buf[M*i+j]));
		}

		fullmodprob+=params[cpu].ret;

	}

	for (cpu=0; cpu<num_threads; cpu++)
	{
		SG_FREE(params[cpu].p_buf);
		SG_FREE(params[cpu].q_buf);
		SG_FREE(params[cpu].a_buf);
		SG_FREE(params[cpu].b_buf);
	}

	SG_FREE(threads);
	SG_FREE(params);

	//cache hmm model probability
	hmm->mod_prob=fullmodprob;
	hmm->mod_prob_updated=true ;

	//new model probability is unknown
	normalize();
	invalidate_model();
}

#else // USE_HMMPARALLEL

//estimates new model lambda out of lambda_estimate using baum welch algorithm
void CHMM::estimate_model_baum_welch(CHMM* estimate)
{
	int32_t i,j,t,dim;
	float64_t a_sum, b_sum;	//numerator
	float64_t dimmodprob=0;	//model probability for dim
	float64_t fullmodprob=0;	//for all dims

	//clear actual model a,b,p,q are used as numerator
	for (i=0; i<N; i++)
	{
		if (estimate->get_p(i)>CMath::ALMOST_NEG_INFTY)
			set_p(i,log(PSEUDO));
		else
			set_p(i,estimate->get_p(i));
		if (estimate->get_q(i)>CMath::ALMOST_NEG_INFTY)
			set_q(i,log(PSEUDO));
		else
			set_q(i,estimate->get_q(i));

		for (j=0; j<N; j++)
			if (estimate->get_a(i,j)>CMath::ALMOST_NEG_INFTY)
				set_a(i,j, log(PSEUDO));
			else
				set_a(i,j,estimate->get_a(i,j));
		for (j=0; j<M; j++)
			if (estimate->get_b(i,j)>CMath::ALMOST_NEG_INFTY)
				set_b(i,j, log(PSEUDO));
			else
				set_b(i,j,estimate->get_b(i,j));
	}
	invalidate_model();

	//change summation order to make use of alpha/beta caches
	for (dim=0; dim<p_observations->get_num_vectors(); dim++)
	{
		dimmodprob=estimate->model_probability(dim);
		fullmodprob+=dimmodprob ;

		for (i=0; i<N; i++)
		{
			//estimate initial+end state distribution numerator
			set_p(i, CMath::logarithmic_sum(get_p(i), estimate->get_p(i)+estimate->get_b(i,p_observations->get_feature(dim,0))+estimate->backward(0,i,dim) - dimmodprob));
			set_q(i, CMath::logarithmic_sum(get_q(i), estimate->forward(p_observations->get_vector_length(dim)-1, i, dim)+estimate->get_q(i) - dimmodprob ));

			int32_t num = trans_list_backward_cnt[i] ;

			//estimate a
			for (j=0; j<num; j++)
			{
				int32_t jj = trans_list_backward[i][j] ;
				a_sum=-CMath::INFTY;

				for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
				{
					a_sum= CMath::logarithmic_sum(a_sum, estimate->forward(t,i,dim)+
							estimate->get_a(i,jj)+estimate->get_b(jj,p_observations->get_feature(dim,t+1))+estimate->backward(t+1,jj,dim));
				}
				set_a(i,jj, CMath::logarithmic_sum(get_a(i,jj), a_sum-dimmodprob));
			}

			//estimate b
			for (j=0; j<M; j++)
			{
				b_sum=-CMath::INFTY;

				for (t=0; t<p_observations->get_vector_length(dim); t++)
				{
					if (p_observations->get_feature(dim,t)==j)
						b_sum=CMath::logarithmic_sum(b_sum, estimate->forward(t,i,dim)+estimate->backward(t, i, dim));
				}

				set_b(i,j, CMath::logarithmic_sum(get_b(i,j), b_sum-dimmodprob));
			}
		}
	}

	//cache estimate model probability
	estimate->mod_prob=fullmodprob;
	estimate->mod_prob_updated=true ;

	//new model probability is unknown
	normalize();
	invalidate_model();
}

//estimates new model lambda out of lambda_estimate using baum welch algorithm
void CHMM::estimate_model_baum_welch_old(CHMM* estimate)
{
	int32_t i,j,t,dim;
	float64_t a_sum, b_sum;	//numerator
	float64_t dimmodprob=0;	//model probability for dim
	float64_t fullmodprob=0;	//for all dims

	//clear actual model a,b,p,q are used as numerator
	for (i=0; i<N; i++)
	  {
	    if (estimate->get_p(i)>CMath::ALMOST_NEG_INFTY)
	      set_p(i,log(PSEUDO));
	    else
	      set_p(i,estimate->get_p(i));
	    if (estimate->get_q(i)>CMath::ALMOST_NEG_INFTY)
	      set_q(i,log(PSEUDO));
	    else
	      set_q(i,estimate->get_q(i));

	    for (j=0; j<N; j++)
	      if (estimate->get_a(i,j)>CMath::ALMOST_NEG_INFTY)
		set_a(i,j, log(PSEUDO));
	      else
		set_a(i,j,estimate->get_a(i,j));
	    for (j=0; j<M; j++)
	      if (estimate->get_b(i,j)>CMath::ALMOST_NEG_INFTY)
		set_b(i,j, log(PSEUDO));
	      else
		set_b(i,j,estimate->get_b(i,j));
	  }
	invalidate_model();

	//change summation order to make use of alpha/beta caches
	for (dim=0; dim<p_observations->get_num_vectors(); dim++)
	  {
	    dimmodprob=estimate->model_probability(dim);
	    fullmodprob+=dimmodprob ;

	    for (i=0; i<N; i++)
	      {
		//estimate initial+end state distribution numerator
		set_p(i, CMath::logarithmic_sum(get_p(i), estimate->get_p(i)+estimate->get_b(i,p_observations->get_feature(dim,0))+estimate->backward(0,i,dim) - dimmodprob));
		set_q(i, CMath::logarithmic_sum(get_q(i), estimate->forward(p_observations->get_vector_length(dim)-1, i, dim)+estimate->get_q(i) - dimmodprob ));

		//estimate a
		for (j=0; j<N; j++)
		  {
		    a_sum=-CMath::INFTY;

		    for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
		      {
			a_sum= CMath::logarithmic_sum(a_sum, estimate->forward(t,i,dim)+
						    estimate->get_a(i,j)+estimate->get_b(j,p_observations->get_feature(dim,t+1))+estimate->backward(t+1,j,dim));
		      }
		    set_a(i,j, CMath::logarithmic_sum(get_a(i,j), a_sum-dimmodprob));
		  }

		//estimate b
		for (j=0; j<M; j++)
		  {
		    b_sum=-CMath::INFTY;

		    for (t=0; t<p_observations->get_vector_length(dim); t++)
		      {
			if (p_observations->get_feature(dim,t)==j)
			  b_sum=CMath::logarithmic_sum(b_sum, estimate->forward(t,i,dim)+estimate->backward(t, i, dim));
		      }

		    set_b(i,j, CMath::logarithmic_sum(get_b(i,j), b_sum-dimmodprob));
		  }
	      }
	  }

	//cache estimate model probability
	estimate->mod_prob=fullmodprob;
	estimate->mod_prob_updated=true ;

	//new model probability is unknown
	normalize();
	invalidate_model();
}
#endif // USE_HMMPARALLEL

//estimates new model lambda out of lambda_estimate using baum welch algorithm
// optimize only p, q, a but not b
void CHMM::estimate_model_baum_welch_trans(CHMM* estimate)
{
	int32_t i,j,t,dim;
	float64_t a_sum;	//numerator
	float64_t dimmodprob=0;	//model probability for dim
	float64_t fullmodprob=0;	//for all dims

	//clear actual model a,b,p,q are used as numerator
	for (i=0; i<N; i++)
	  {
	    if (estimate->get_p(i)>CMath::ALMOST_NEG_INFTY)
	      set_p(i,log(PSEUDO));
	    else
	      set_p(i,estimate->get_p(i));
	    if (estimate->get_q(i)>CMath::ALMOST_NEG_INFTY)
	      set_q(i,log(PSEUDO));
	    else
	      set_q(i,estimate->get_q(i));

	    for (j=0; j<N; j++)
	      if (estimate->get_a(i,j)>CMath::ALMOST_NEG_INFTY)
		set_a(i,j, log(PSEUDO));
	      else
		set_a(i,j,estimate->get_a(i,j));
	    for (j=0; j<M; j++)
	      set_b(i,j,estimate->get_b(i,j));
	  }
	invalidate_model();

	//change summation order to make use of alpha/beta caches
	for (dim=0; dim<p_observations->get_num_vectors(); dim++)
	  {
	    dimmodprob=estimate->model_probability(dim);
	    fullmodprob+=dimmodprob ;

	    for (i=0; i<N; i++)
	      {
		//estimate initial+end state distribution numerator
		set_p(i, CMath::logarithmic_sum(get_p(i), estimate->get_p(i)+estimate->get_b(i,p_observations->get_feature(dim,0))+estimate->backward(0,i,dim) - dimmodprob));
		set_q(i, CMath::logarithmic_sum(get_q(i), estimate->forward(p_observations->get_vector_length(dim)-1, i, dim)+estimate->get_q(i) - dimmodprob ));

		int32_t num = trans_list_backward_cnt[i] ;
		//estimate a
		for (j=0; j<num; j++)
		  {
		    int32_t jj = trans_list_backward[i][j] ;
		    a_sum=-CMath::INFTY;

		    for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
		      {
			a_sum= CMath::logarithmic_sum(a_sum, estimate->forward(t,i,dim)+
						    estimate->get_a(i,jj)+estimate->get_b(jj,p_observations->get_feature(dim,t+1))+estimate->backward(t+1,jj,dim));
		      }
		    set_a(i,jj, CMath::logarithmic_sum(get_a(i,jj), a_sum-dimmodprob));
		  }
	      }
	  }

	//cache estimate model probability
	estimate->mod_prob=fullmodprob;
	estimate->mod_prob_updated=true ;

	//new model probability is unknown
	normalize();
	invalidate_model();
}



//estimates new model lambda out of lambda_estimate using baum welch algorithm
void CHMM::estimate_model_baum_welch_defined(CHMM* estimate)
{
	int32_t i,j,old_i,k,t,dim;
	float64_t a_sum_num, b_sum_num;		//numerator
	float64_t a_sum_denom, b_sum_denom;	//denominator
	float64_t dimmodprob=-CMath::INFTY;	//model probability for dim
	float64_t fullmodprob=0;			//for all dims
	float64_t* A=ARRAYN1(0);
	float64_t* B=ARRAYN2(0);

	//clear actual model a,b,p,q are used as numerator
	//A,B as denominator for a,b
	for (k=0; (i=model->get_learn_p(k))!=-1; k++)
		set_p(i,log(PSEUDO));

	for (k=0; (i=model->get_learn_q(k))!=-1; k++)
		set_q(i,log(PSEUDO));

	for (k=0; (i=model->get_learn_a(k,0))!=-1; k++)
	{
		j=model->get_learn_a(k,1);
		set_a(i,j, log(PSEUDO));
	}

	for (k=0; (i=model->get_learn_b(k,0))!=-1; k++)
	{
		j=model->get_learn_b(k,1);
		set_b(i,j, log(PSEUDO));
	}

	for (i=0; i<N; i++)
	{
		A[i]=log(PSEUDO);
		B[i]=log(PSEUDO);
	}

#ifdef USE_HMMPARALLEL
	int32_t num_threads = parallel->get_num_threads();
	pthread_t *threads=SG_MALLOC(pthread_t, num_threads);
	S_DIM_THREAD_PARAM *params=SG_MALLOC(S_DIM_THREAD_PARAM, num_threads);

	if (p_observations->get_num_vectors()<num_threads)
		num_threads=p_observations->get_num_vectors();
#endif

	//change summation order to make use of alpha/beta caches
	for (dim=0; dim<p_observations->get_num_vectors(); dim++)
	{
#ifdef USE_HMMPARALLEL
		if (dim%num_threads==0)
		{
			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
				{
					params[i].hmm=estimate ;
					params[i].dim=dim+i ;
					pthread_create(&threads[i], NULL, bw_single_dim_prefetch, (void*)&params[i]) ;
				}
			}
			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
				{
					pthread_join(threads[i], NULL);
					dimmodprob = params[i].prob_sum;
				}
			}
		}
#else
		dimmodprob=estimate->model_probability(dim);
#endif // USE_HMMPARALLEL

		//and denominator
		fullmodprob+= dimmodprob;

		//estimate initial+end state distribution numerator
		for (k=0; (i=model->get_learn_p(k))!=-1; k++)
			set_p(i, CMath::logarithmic_sum(get_p(i), estimate->forward(0,i,dim)+estimate->backward(0,i,dim) - dimmodprob ) );

		for (k=0; (i=model->get_learn_q(k))!=-1; k++)
			set_q(i, CMath::logarithmic_sum(get_q(i), estimate->forward(p_observations->get_vector_length(dim)-1, i, dim)+
						estimate->backward(p_observations->get_vector_length(dim)-1, i, dim)  - dimmodprob ) );

		//estimate a
		old_i=-1;
		for (k=0; (i=model->get_learn_a(k,0))!=-1; k++)
		{
			//denominator is constant for j
			//therefore calculate it first
			if (old_i!=i)
			{
				old_i=i;
				a_sum_denom=-CMath::INFTY;

				for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
					a_sum_denom= CMath::logarithmic_sum(a_sum_denom, estimate->forward(t,i,dim)+estimate->backward(t,i,dim));

				A[i]= CMath::logarithmic_sum(A[i], a_sum_denom-dimmodprob);
			}

			j=model->get_learn_a(k,1);
			a_sum_num=-CMath::INFTY;
			for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
			{
				a_sum_num= CMath::logarithmic_sum(a_sum_num, estimate->forward(t,i,dim)+
						estimate->get_a(i,j)+estimate->get_b(j,p_observations->get_feature(dim,t+1))+estimate->backward(t+1,j,dim));
			}

			set_a(i,j, CMath::logarithmic_sum(get_a(i,j), a_sum_num-dimmodprob));
		}

		//estimate  b
		old_i=-1;
		for (k=0; (i=model->get_learn_b(k,0))!=-1; k++)
		{

			//denominator is constant for j
			//therefore calculate it first
			if (old_i!=i)
			{
				old_i=i;
				b_sum_denom=-CMath::INFTY;

				for (t=0; t<p_observations->get_vector_length(dim); t++)
					b_sum_denom= CMath::logarithmic_sum(b_sum_denom, estimate->forward(t,i,dim)+estimate->backward(t,i,dim));

				B[i]= CMath::logarithmic_sum(B[i], b_sum_denom-dimmodprob);
			}

			j=model->get_learn_b(k,1);
			b_sum_num=-CMath::INFTY;
			for (t=0; t<p_observations->get_vector_length(dim); t++)
			{
				if (p_observations->get_feature(dim,t)==j)
					b_sum_num=CMath::logarithmic_sum(b_sum_num, estimate->forward(t,i,dim)+estimate->backward(t, i, dim));
			}

			set_b(i,j, CMath::logarithmic_sum(get_b(i,j), b_sum_num-dimmodprob));
		}
	}
#ifdef USE_HMMPARALLEL
	SG_FREE(threads);
	SG_FREE(params);
#endif


	//calculate estimates
	for (k=0; (i=model->get_learn_p(k))!=-1; k++)
		set_p(i, get_p(i)-log(p_observations->get_num_vectors()+N*PSEUDO) );

	for (k=0; (i=model->get_learn_q(k))!=-1; k++)
		set_q(i, get_q(i)-log(p_observations->get_num_vectors()+N*PSEUDO) );

	for (k=0; (i=model->get_learn_a(k,0))!=-1; k++)
	{
		j=model->get_learn_a(k,1);
		set_a(i,j, get_a(i,j) - A[i]);
	}

	for (k=0; (i=model->get_learn_b(k,0))!=-1; k++)
	{
		j=model->get_learn_b(k,1);
		set_b(i,j, get_b(i,j) - B[i]);
	}

	//cache estimate model probability
	estimate->mod_prob=fullmodprob;
	estimate->mod_prob_updated=true ;

	//new model probability is unknown
	normalize();
	invalidate_model();
}

//estimates new model lambda out of lambda_estimate using viterbi algorithm
void CHMM::estimate_model_viterbi(CHMM* estimate)
{
	int32_t i,j,t;
	float64_t sum;
	float64_t* P=ARRAYN1(0);
	float64_t* Q=ARRAYN2(0);

	path_deriv_updated=false ;

	//initialize with pseudocounts
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
			set_A(i,j, PSEUDO);

		for (j=0; j<M; j++)
			set_B(i,j, PSEUDO);

		P[i]=PSEUDO;
		Q[i]=PSEUDO;
	}

	float64_t allpatprob=0 ;

#ifdef USE_HMMPARALLEL
	int32_t num_threads = parallel->get_num_threads();
	pthread_t *threads=SG_MALLOC(pthread_t, num_threads);
	S_DIM_THREAD_PARAM *params=SG_MALLOC(S_DIM_THREAD_PARAM, num_threads);

	if (p_observations->get_num_vectors()<num_threads)
		num_threads=p_observations->get_num_vectors();
#endif

	for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
	{

#ifdef USE_HMMPARALLEL
		if (dim%num_threads==0)
		{
			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
				{
					params[i].hmm=estimate ;
					params[i].dim=dim+i ;
					pthread_create(&threads[i], NULL, vit_dim_prefetch, (void*)&params[i]) ;
				}
			}
			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
				{
					pthread_join(threads[i], NULL);
					allpatprob += params[i].prob_sum;
				}
			}
		}
#else
		//using viterbi to find best path
		allpatprob += estimate->best_path(dim);
#endif // USE_HMMPARALLEL

		//counting occurences for A and B
		for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
		{
			set_A(estimate->PATH(dim)[t], estimate->PATH(dim)[t+1], get_A(estimate->PATH(dim)[t], estimate->PATH(dim)[t+1])+1);
			set_B(estimate->PATH(dim)[t], p_observations->get_feature(dim,t),  get_B(estimate->PATH(dim)[t], p_observations->get_feature(dim,t))+1);
		}

		set_B(estimate->PATH(dim)[p_observations->get_vector_length(dim)-1], p_observations->get_feature(dim,p_observations->get_vector_length(dim)-1),  get_B(estimate->PATH(dim)[p_observations->get_vector_length(dim)-1], p_observations->get_feature(dim,p_observations->get_vector_length(dim)-1)) + 1 );

		P[estimate->PATH(dim)[0]]++;
		Q[estimate->PATH(dim)[p_observations->get_vector_length(dim)-1]]++;
	}

#ifdef USE_HMMPARALLEL
	SG_FREE(threads);
	SG_FREE(params);
#endif

	allpatprob/=p_observations->get_num_vectors() ;
	estimate->all_pat_prob=allpatprob ;
	estimate->all_path_prob_updated=true ;

	//converting A to probability measure a
	for (i=0; i<N; i++)
	{
		sum=0;
		for (j=0; j<N; j++)
			sum+=get_A(i,j);

		for (j=0; j<N; j++)
			set_a(i,j, log(get_A(i,j)/sum));
	}

	//converting B to probability measures b
	for (i=0; i<N; i++)
	{
		sum=0;
		for (j=0; j<M; j++)
			sum+=get_B(i,j);

		for (j=0; j<M; j++)
			set_b(i,j, log(get_B(i, j)/sum));
	}

	//converting P to probability measure p
	sum=0;
	for (i=0; i<N; i++)
		sum+=P[i];

	for (i=0; i<N; i++)
		set_p(i, log(P[i]/sum));

	//converting Q to probability measure q
	sum=0;
	for (i=0; i<N; i++)
		sum+=Q[i];

	for (i=0; i<N; i++)
		set_q(i, log(Q[i]/sum));

	//new model probability is unknown
	invalidate_model();
}

// estimate parameters listed in learn_x
void CHMM::estimate_model_viterbi_defined(CHMM* estimate)
{
	int32_t i,j,k,t;
	float64_t sum;
	float64_t* P=ARRAYN1(0);
	float64_t* Q=ARRAYN2(0);

	path_deriv_updated=false ;

	//initialize with pseudocounts
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
			set_A(i,j, PSEUDO);

		for (j=0; j<M; j++)
			set_B(i,j, PSEUDO);

		P[i]=PSEUDO;
		Q[i]=PSEUDO;
	}

#ifdef USE_HMMPARALLEL
	int32_t num_threads = parallel->get_num_threads();
	pthread_t *threads=SG_MALLOC(pthread_t, num_threads);
	S_DIM_THREAD_PARAM *params=SG_MALLOC(S_DIM_THREAD_PARAM, num_threads);
#endif

	float64_t allpatprob=0.0 ;
	for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
	{

#ifdef USE_HMMPARALLEL
		if (dim%num_threads==0)
		{
			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
				{
					params[i].hmm=estimate ;
					params[i].dim=dim+i ;
					pthread_create(&threads[i], NULL, vit_dim_prefetch, (void*)&params[i]) ;
				}
			}
			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
				{
					pthread_join(threads[i], NULL);
					allpatprob += params[i].prob_sum;
				}
			}
		}
#else // USE_HMMPARALLEL
		//using viterbi to find best path
		allpatprob += estimate->best_path(dim);
#endif // USE_HMMPARALLEL


		//counting occurences for A and B
		for (t=0; t<p_observations->get_vector_length(dim)-1; t++)
		{
			set_A(estimate->PATH(dim)[t], estimate->PATH(dim)[t+1], get_A(estimate->PATH(dim)[t], estimate->PATH(dim)[t+1])+1);
			set_B(estimate->PATH(dim)[t], p_observations->get_feature(dim,t),  get_B(estimate->PATH(dim)[t], p_observations->get_feature(dim,t))+1);
		}

		set_B(estimate->PATH(dim)[p_observations->get_vector_length(dim)-1], p_observations->get_feature(dim,p_observations->get_vector_length(dim)-1),  get_B(estimate->PATH(dim)[p_observations->get_vector_length(dim)-1], p_observations->get_feature(dim,p_observations->get_vector_length(dim)-1)) + 1 );

		P[estimate->PATH(dim)[0]]++;
		Q[estimate->PATH(dim)[p_observations->get_vector_length(dim)-1]]++;
	}

#ifdef USE_HMMPARALLEL
	SG_FREE(threads);
	SG_FREE(params);
#endif

	//estimate->invalidate_model() ;
	//float64_t q=estimate->best_path(-1) ;

	allpatprob/=p_observations->get_num_vectors() ;
	estimate->all_pat_prob=allpatprob ;
	estimate->all_path_prob_updated=true ;


	//copy old model
	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
			set_a(i,j, estimate->get_a(i,j));

		for (j=0; j<M; j++)
			set_b(i,j, estimate->get_b(i,j));
	}

	//converting A to probability measure a
	i=0;
	sum=0;
	j=model->get_learn_a(i,0);
	k=i;
	while (model->get_learn_a(i,0)!=-1 || k<i)
	{
		if (j==model->get_learn_a(i,0))
		{
			sum+=get_A(model->get_learn_a(i,0), model->get_learn_a(i,1));
			i++;
		}
		else
		{
			while (k<i)
			{
				set_a(model->get_learn_a(k,0), model->get_learn_a(k,1), log (get_A(model->get_learn_a(k,0), model->get_learn_a(k,1)) / sum));
				k++;
			}

			sum=0;
			j=model->get_learn_a(i,0);
			k=i;
		}
	}

	//converting B to probability measures b
	i=0;
	sum=0;
	j=model->get_learn_b(i,0);
	k=i;
	while (model->get_learn_b(i,0)!=-1 || k<i)
	{
		if (j==model->get_learn_b(i,0))
		{
			sum+=get_B(model->get_learn_b(i,0),model->get_learn_b(i,1));
			i++;
		}
		else
		{
			while (k<i)
			{
				set_b(model->get_learn_b(k,0),model->get_learn_b(k,1), log (get_B(model->get_learn_b(k,0), model->get_learn_b(k,1)) / sum));
				k++;
			}

			sum=0;
			j=model->get_learn_b(i,0);
			k=i;
		}
	}

	i=0;
	sum=0;
	while (model->get_learn_p(i)!=-1)
	{
		sum+=P[model->get_learn_p(i)] ;
		i++ ;
	} ;
	i=0 ;
	while (model->get_learn_p(i)!=-1)
	{
		set_p(model->get_learn_p(i), log(P[model->get_learn_p(i)]/sum));
		i++ ;
	} ;

	i=0;
	sum=0;
	while (model->get_learn_q(i)!=-1)
	{
		sum+=Q[model->get_learn_q(i)] ;
		i++ ;
	} ;
	i=0 ;
	while (model->get_learn_q(i)!=-1)
	{
		set_q(model->get_learn_q(i), log(Q[model->get_learn_q(i)]/sum));
		i++ ;
	} ;


	//new model probability is unknown
	invalidate_model();
}
//------------------------------------------------------------------------------------//

//to give an idea what the model looks like
void CHMM::output_model(bool verbose)
{
	int32_t i,j;
	float64_t checksum;

	//generic info
	SG_INFO("log(Pr[O|model])=%e, #states: %i, #observationssymbols: %i, #observations: %ix%i\n",
			(float64_t)((p_observations) ? model_probability() : -CMath::INFTY),
			N, M, ((p_observations) ? p_observations->get_max_vector_length() : 0), ((p_observations) ? p_observations->get_num_vectors() : 0));

	if (verbose)
	{
		// tranisition matrix a
		SG_INFO("\ntransition matrix\n")
		for (i=0; i<N; i++)
		{
			checksum= get_q(i);
			for (j=0; j<N; j++)
			{
				checksum= CMath::logarithmic_sum(checksum, get_a(i,j));

				SG_INFO("a(%02i,%02i)=%1.4f ",i,j, (float32_t) exp(get_a(i,j)))

				if (j % 4 == 3)
					SG_PRINT("\n")
			}
			if (fabs(checksum)>1e-5)
				SG_DEBUG(" checksum % E ******* \n",checksum)
			else
				SG_DEBUG(" checksum % E\n",checksum)
		}

		// distribution of start states p
		SG_INFO("\ndistribution of start states\n")
		checksum=-CMath::INFTY;
		for (i=0; i<N; i++)
		{
			checksum= CMath::logarithmic_sum(checksum, get_p(i));
			SG_INFO("p(%02i)=%1.4f ",i, (float32_t) exp(get_p(i)))
			if (i % 4 == 3)
				SG_PRINT("\n")
		}
		if (fabs(checksum)>1e-5)
			SG_DEBUG(" checksum % E ******* \n",checksum)
		else
			SG_DEBUG(" checksum=% E\n", checksum)

		// distribution of terminal states p
		SG_INFO("\ndistribution of terminal states\n")
		checksum=-CMath::INFTY;
		for (i=0; i<N; i++)
		{
			checksum= CMath::logarithmic_sum(checksum, get_q(i));
			SG_INFO("q(%02i)=%1.4f ",i, (float32_t) exp(get_q(i)))
			if (i % 4 == 3)
				SG_INFO("\n")
		}
		if (fabs(checksum)>1e-5)
			SG_DEBUG(" checksum % E ******* \n",checksum)
		else
			SG_DEBUG(" checksum=% E\n", checksum)

		// distribution of observations given the state b
		SG_INFO("\ndistribution of observations given the state\n")
		for (i=0; i<N; i++)
		{
			checksum=-CMath::INFTY;
			for (j=0; j<M; j++)
			{
				checksum=CMath::logarithmic_sum(checksum, get_b(i,j));
				SG_INFO("b(%02i,%02i)=%1.4f ",i,j, (float32_t) exp(get_b(i,j)))
				if (j % 4 == 3)
					SG_PRINT("\n")
			}
			if (fabs(checksum)>1e-5)
				SG_DEBUG(" checksum % E ******* \n",checksum)
			else
				SG_DEBUG(" checksum % E\n",checksum)
		}
	}
	SG_PRINT("\n")
}

//to give an idea what the model looks like
void CHMM::output_model_defined(bool verbose)
{
	int32_t i,j;
	if (!model)
		return ;

	//generic info
	SG_INFO("log(Pr[O|model])=%e, #states: %i, #observationssymbols: %i, #observations: %ix%i\n",
			(float64_t)((p_observations) ? model_probability() : -CMath::INFTY),
			N, M, ((p_observations) ? p_observations->get_max_vector_length() : 0), ((p_observations) ? p_observations->get_num_vectors() : 0));

	if (verbose)
	{
		// tranisition matrix a
		SG_INFO("\ntransition matrix\n")

		//initialize a values that have to be learned
		i=0;
		j=model->get_learn_a(i,0);
		while (model->get_learn_a(i,0)!=-1)
		{
			if (j!=model->get_learn_a(i,0))
			{
				j=model->get_learn_a(i,0);
				SG_PRINT("\n")
			}

			SG_INFO("a(%02i,%02i)=%1.4f ",model->get_learn_a(i,0), model->get_learn_a(i,1), (float32_t) exp(get_a(model->get_learn_a(i,0), model->get_learn_a(i,1))))
			i++;
		}

		// distribution of observations given the state b
		SG_INFO("\n\ndistribution of observations given the state\n")
		i=0;
		j=model->get_learn_b(i,0);
		while (model->get_learn_b(i,0)!=-1)
		{
			if (j!=model->get_learn_b(i,0))
			{
				j=model->get_learn_b(i,0);
				SG_PRINT("\n")
			}

			SG_INFO("b(%02i,%02i)=%1.4f ",model->get_learn_b(i,0),model->get_learn_b(i,1), (float32_t) exp(get_b(model->get_learn_b(i,0),model->get_learn_b(i,1))))
			i++;
		}

		SG_PRINT("\n")
	}
	SG_PRINT("\n")
}

//------------------------------------------------------------------------------------//

//convert model to log probabilities
void CHMM::convert_to_log()
{
	int32_t i,j;

	for (i=0; i<N; i++)
	{
		if (get_p(i)!=0)
			set_p(i, log(get_p(i)));
		else
			set_p(i, -CMath::INFTY);;
	}

	for (i=0; i<N; i++)
	{
		if (get_q(i)!=0)
			set_q(i, log(get_q(i)));
		else
			set_q(i, -CMath::INFTY);;
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<N; j++)
		{
			if (get_a(i,j)!=0)
				set_a(i,j, log(get_a(i,j)));
			else
				set_a(i,j, -CMath::INFTY);
		}
	}

	for (i=0; i<N; i++)
	{
		for (j=0; j<M; j++)
		{
			if (get_b(i,j)!=0)
				set_b(i,j, log(get_b(i,j)));
			else
				set_b(i,j, -CMath::INFTY);
		}
	}
	loglikelihood=true;

	invalidate_model();
}

//init model with random values
void CHMM::init_model_random()
{
	const float64_t MIN_RAND=23e-3;

	float64_t sum;
	int32_t i,j;

	//initialize a with random values
	for (i=0; i<N; i++)
	{
		sum=0;
		for (j=0; j<N; j++)
		{
			set_a(i,j, CMath::random(MIN_RAND, 1.0));

			sum+=get_a(i,j);
		}

		for (j=0; j<N; j++)
			set_a(i,j, get_a(i,j)/sum);
	}

	//initialize pi with random values
	sum=0;
	for (i=0; i<N; i++)
	{
		set_p(i, CMath::random(MIN_RAND, 1.0));

		sum+=get_p(i);
	}

	for (i=0; i<N; i++)
		set_p(i, get_p(i)/sum);

	//initialize q with random values
	sum=0;
	for (i=0; i<N; i++)
	{
		set_q(i, CMath::random(MIN_RAND, 1.0));

		sum+=get_q(i);
	}

	for (i=0; i<N; i++)
		set_q(i, get_q(i)/sum);

	//initialize b with random values
	for (i=0; i<N; i++)
	{
		sum=0;
		for (j=0; j<M; j++)
		{
			set_b(i,j, CMath::random(MIN_RAND, 1.0));

			sum+=get_b(i,j);
		}

		for (j=0; j<M; j++)
			set_b(i,j, get_b(i,j)/sum);
	}

	//initialize pat/mod_prob as not calculated
	invalidate_model();
}

//init model according to const_x
void CHMM::init_model_defined()
{
	int32_t i,j,k,r;
	float64_t sum;
	const float64_t MIN_RAND=23e-3;

	//initialize a with zeros
	for (i=0; i<N; i++)
		for (j=0; j<N; j++)
			set_a(i,j, 0);

	//initialize p with zeros
	for (i=0; i<N; i++)
		set_p(i, 0);

	//initialize q with zeros
	for (i=0; i<N; i++)
		set_q(i, 0);

	//initialize b with zeros
	for (i=0; i<N; i++)
		for (j=0; j<M; j++)
			set_b(i,j, 0);


	//initialize a values that have to be learned
	float64_t *R=SG_MALLOC(float64_t, N);
	for (r=0; r<N; r++) R[r]=CMath::random(MIN_RAND,1.0);
	i=0; sum=0; k=i;
	j=model->get_learn_a(i,0);
	while (model->get_learn_a(i,0)!=-1 || k<i)
	{
		if (j==model->get_learn_a(i,0))
		{
			sum+=R[model->get_learn_a(i,1)] ;
			i++;
		}
		else
		{
			while (k<i)
			{
				set_a(model->get_learn_a(k,0), model->get_learn_a(k,1),
						R[model->get_learn_a(k,1)]/sum);
				k++;
			}
			j=model->get_learn_a(i,0);
			k=i;
			sum=0;
			for (r=0; r<N; r++) R[r]=CMath::random(MIN_RAND,1.0);
		}
	}
	SG_FREE(R); R=NULL ;

	//initialize b values that have to be learned
	R=SG_MALLOC(float64_t, M);
	for (r=0; r<M; r++) R[r]=CMath::random(MIN_RAND,1.0);
	i=0; sum=0; k=0 ;
	j=model->get_learn_b(i,0);
	while (model->get_learn_b(i,0)!=-1 || k<i)
	{
		if (j==model->get_learn_b(i,0))
		{
			sum+=R[model->get_learn_b(i,1)] ;
			i++;
		}
		else
		{
			while (k<i)
			{
				set_b(model->get_learn_b(k,0),model->get_learn_b(k,1),
						R[model->get_learn_b(k,1)]/sum);
				k++;
			}

			j=model->get_learn_b(i,0);
			k=i;
			sum=0;
			for (r=0; r<M; r++) R[r]=CMath::random(MIN_RAND,1.0);
		}
	}
	SG_FREE(R); R=NULL ;

	//set consts into a
	i=0;
	while (model->get_const_a(i,0) != -1)
	{
		set_a(model->get_const_a(i,0), model->get_const_a(i,1), model->get_const_a_val(i));
		i++;
	}

	//set consts into b
	i=0;
	while (model->get_const_b(i,0) != -1)
	{
		set_b(model->get_const_b(i,0), model->get_const_b(i,1), model->get_const_b_val(i));
		i++;
	}

	//set consts into p
	i=0;
	while (model->get_const_p(i) != -1)
	{
		set_p(model->get_const_p(i), model->get_const_p_val(i));
		i++;
	}

	//initialize q with zeros
	for (i=0; i<N; i++)
		set_q(i, 0.0);

	//set consts into q
	i=0;
	while (model->get_const_q(i) != -1)
	{
		set_q(model->get_const_q(i), model->get_const_q_val(i));
		i++;
	}

	// init p
	i=0;
	sum=0;
	while (model->get_learn_p(i)!=-1)
	{
		set_p(model->get_learn_p(i),CMath::random(MIN_RAND,1.0)) ;
		sum+=get_p(model->get_learn_p(i)) ;
		i++ ;
	} ;
	i=0 ;
	while (model->get_learn_p(i)!=-1)
	{
		set_p(model->get_learn_p(i), get_p(model->get_learn_p(i))/sum);
		i++ ;
	} ;

	// initialize q
	i=0;
	sum=0;
	while (model->get_learn_q(i)!=-1)
	{
		set_q(model->get_learn_q(i),CMath::random(MIN_RAND,1.0)) ;
		sum+=get_q(model->get_learn_q(i)) ;
		i++ ;
	} ;
	i=0 ;
	while (model->get_learn_q(i)!=-1)
	{
		set_q(model->get_learn_q(i), get_q(model->get_learn_q(i))/sum);
		i++ ;
	} ;

	//initialize pat/mod_prob as not calculated
	invalidate_model();
}

void CHMM::clear_model()
{
	int32_t i,j;
	for (i=0; i<N; i++)
	{
		set_p(i, log(PSEUDO));
		set_q(i, log(PSEUDO));

		for (j=0; j<N; j++)
			set_a(i,j, log(PSEUDO));

		for (j=0; j<M; j++)
			set_b(i,j, log(PSEUDO));
	}
}

void CHMM::clear_model_defined()
{
	int32_t i,j,k;

	for (i=0; (j=model->get_learn_p(i))!=-1; i++)
		set_p(j, log(PSEUDO));

	for (i=0; (j=model->get_learn_q(i))!=-1; i++)
		set_q(j, log(PSEUDO));

	for (i=0; (j=model->get_learn_a(i,0))!=-1; i++)
	{
		k=model->get_learn_a(i,1); // catch (j,k) as indizes to be learned
		set_a(j,k, log(PSEUDO));
	}

	for (i=0; (j=model->get_learn_b(i,0))!=-1; i++)
	{
		k=model->get_learn_b(i,1); // catch (j,k) as indizes to be learned
		set_b(j,k, log(PSEUDO));
	}
}

void CHMM::copy_model(CHMM* l)
{
	int32_t i,j;
	for (i=0; i<N; i++)
	{
		set_p(i, l->get_p(i));
		set_q(i, l->get_q(i));

		for (j=0; j<N; j++)
			set_a(i,j, l->get_a(i,j));

		for (j=0; j<M; j++)
			set_b(i,j, l->get_b(i,j));
	}
}

void CHMM::invalidate_model()
{
	//initialize pat/mod_prob/alpha/beta cache as not calculated
	this->mod_prob=0.0;
	this->mod_prob_updated=false;

	if (mem_initialized)
	{
	  if (trans_list_forward_cnt)
	    SG_FREE(trans_list_forward_cnt);
	  trans_list_forward_cnt=NULL ;
	  if (trans_list_backward_cnt)
	    SG_FREE(trans_list_backward_cnt);
	  trans_list_backward_cnt=NULL ;
	  if (trans_list_forward)
	    {
	      for (int32_t i=0; i<trans_list_len; i++)
		if (trans_list_forward[i])
		  SG_FREE(trans_list_forward[i]);
	      SG_FREE(trans_list_forward);
	      trans_list_forward=NULL ;
	    }
	  if (trans_list_backward)
	    {
	      for (int32_t i=0; i<trans_list_len; i++)
		if (trans_list_backward[i])
		  SG_FREE(trans_list_backward[i]);
	      SG_FREE(trans_list_backward);
	      trans_list_backward = NULL ;
	    } ;

	  trans_list_len = N ;
	  trans_list_forward = SG_MALLOC(T_STATES*, N);
	  trans_list_forward_cnt = SG_MALLOC(T_STATES, N);

	  for (int32_t j=0; j<N; j++)
	    {
	      trans_list_forward_cnt[j]= 0 ;
	      trans_list_forward[j]= SG_MALLOC(T_STATES, N);
	      for (int32_t i=0; i<N; i++)
		if (get_a(i,j)>CMath::ALMOST_NEG_INFTY)
		  {
		    trans_list_forward[j][trans_list_forward_cnt[j]]=i ;
		    trans_list_forward_cnt[j]++ ;
		  }
	    } ;

	  trans_list_backward = SG_MALLOC(T_STATES*, N);
	  trans_list_backward_cnt = SG_MALLOC(T_STATES, N);

	  for (int32_t i=0; i<N; i++)
	    {
	      trans_list_backward_cnt[i]= 0 ;
	      trans_list_backward[i]= SG_MALLOC(T_STATES, N);
	      for (int32_t j=0; j<N; j++)
		if (get_a(i,j)>CMath::ALMOST_NEG_INFTY)
		  {
		    trans_list_backward[i][trans_list_backward_cnt[i]]=j ;
		    trans_list_backward_cnt[i]++ ;
		  }
	    } ;
	} ;
	this->all_pat_prob=0.0;
	this->pat_prob=0.0;
	this->path_deriv_updated=false ;
	this->path_deriv_dimension=-1 ;
	this->all_path_prob_updated=false;

#ifdef USE_HMMPARALLEL_STRUCTURES
	{
		for (int32_t i=0; i<parallel->get_num_threads(); i++)
		{
			this->alpha_cache[i].updated=false;
			this->beta_cache[i].updated=false;
			path_prob_updated[i]=false ;
			path_prob_dimension[i]=-1 ;
		} ;
	}
#else // USE_HMMPARALLEL_STRUCTURES
	this->alpha_cache.updated=false;
	this->beta_cache.updated=false;
	this->path_prob_dimension=-1;
	this->path_prob_updated=false;

#endif // USE_HMMPARALLEL_STRUCTURES
}

void CHMM::open_bracket(FILE* file)
{
	int32_t value;
	while (((value=fgetc(file)) != EOF) && (value!='['))	//skip possible spaces and end if '[' occurs
	{
		if (value=='\n')
			line++;
	}

	if (value==EOF)
		error(line, "expected \"[\" in input file");

	while (((value=fgetc(file)) != EOF) && (isspace(value)))	//skip possible spaces
	{
		if (value=='\n')
			line++;
	}

	ungetc(value, file);
}

void CHMM::close_bracket(FILE* file)
{
	int32_t value;
	while (((value=fgetc(file)) != EOF) && (value!=']'))	//skip possible spaces and end if ']' occurs
	{
		if (value=='\n')
			line++;
	}

	if (value==EOF)
		error(line, "expected \"]\" in input file");
}

bool CHMM::comma_or_space(FILE* file)
{
	int32_t value;
	while (((value=fgetc(file)) != EOF) && (value!=',') && (value!=';') && (value!=']'))	 //skip possible spaces and end if ',' or ';' occurs
	{
		if (value=='\n')
			line++;
	}
	if (value==']')
	{
		ungetc(value, file);
		SG_ERROR("found ']' instead of ';' or ','\n")
		return false ;
	} ;

	if (value==EOF)
		error(line, "expected \";\" or \",\" in input file");

	while (((value=fgetc(file)) != EOF) && (isspace(value)))	//skip possible spaces
	{
		if (value=='\n')
			line++;
	}
	ungetc(value, file);
	return true ;
}

bool CHMM::get_numbuffer(FILE* file, char* buffer, int32_t length)
{
	signed char value;

	while (((value=fgetc(file)) != EOF) &&
			!isdigit(value) && (value!='A')
			&& (value!='C') && (value!='G') && (value!='T')
			&& (value!='N') && (value!='n')
			&& (value!='.') && (value!='-') && (value!='e') && (value!=']')) //skip possible spaces+crap
	{
		if (value=='\n')
			line++;
	}
	if (value==']')
	{
		ungetc(value,file) ;
		return false ;
	} ;
	if (value!=EOF)
	{
		int32_t i=0;
		switch (value)
		{
			case 'A':
				value='0' +CAlphabet::B_A;
				break;
			case 'C':
				value='0' +CAlphabet::B_C;
				break;
			case 'G':
				value='0' +CAlphabet::B_G;
				break;
			case 'T':
				value='0' +CAlphabet::B_T;
				break;
		};

		buffer[i++]=value;

		while (((value=fgetc(file)) != EOF) &&
				(isdigit(value) || (value=='.') || (value=='-') || (value=='e')
				 || (value=='A') || (value=='C') || (value=='G')|| (value=='T')
				 || (value=='N') || (value=='n')) && (i<length))
		{
			switch (value)
			{
				case 'A':
					value='0' +CAlphabet::B_A;
					break;
				case 'C':
					value='0' +CAlphabet::B_C;
					break;
				case 'G':
					value='0' +CAlphabet::B_G;
					break;
				case 'T':
					value='0' +CAlphabet::B_T;
					break;
				case '1': case '2': case'3': case '4': case'5':
				case '6': case '7': case'8': case '9': case '0': break ;
				case '.': case 'e': case '-': break ;
				default:
											  SG_ERROR("found crap: %i %c (pos:%li)\n",i,value,ftell(file))
			};
			buffer[i++]=value;
		}
		ungetc(value, file);
		buffer[i]='\0';

		return (i<=length) && (i>0);
	}
	return false;
}

/*
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

   N=<int32_t>;
   M=<int32_t>;

   p=[<float64_t>,<float64_t>...<DOUBLE>];
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
   */

bool CHMM::load_model(FILE* file)
{
	int32_t received_params=0;	//a,b,p,N,M,O

	bool result=false;
	E_STATE state=INITIAL;
	char buffer[1024];

	line=1;
	int32_t i,j;

	if (file)
	{
		while (state!=END)
		{
			int32_t value=fgetc(file);

			if (value=='\n')
				line++;
			if (value==EOF)
				state=END;

			switch (state)
			{
				case INITIAL:	// in the initial state only N,M initialisations and comments are allowed
					if (value=='N')
					{
						if (received_params & GOTN)
							error(line, "in model file: \"p double defined\"");
						else
							state=GET_N;
					}
					else if (value=='M')
					{
						if (received_params & GOTM)
							error(line, "in model file: \"p double defined\"");
						else
							state=GET_M;
					}
					else if (value=='%')
					{
						state=COMMENT;
					}
					break;
				case ARRAYs:	// when n,m, order are known p,a,b arrays are allowed to be read
					if (value=='p')
					{
						if (received_params & GOTp)
							error(line, "in model file: \"p double defined\"");
						else
							state=GET_p;
					}
					if (value=='q')
					{
						if (received_params & GOTq)
							error(line, "in model file: \"q double defined\"");
						else
							state=GET_q;
					}
					else if (value=='a')
					{
						if (received_params & GOTa)
							error(line, "in model file: \"a double defined\"");
						else
							state=GET_a;
					}
					else if (value=='b')
					{
						if (received_params & GOTb)
							error(line, "in model file: \"b double defined\"");
						else
							state=GET_b;
					}
					else if (value=='%')
					{
						state=COMMENT;
					}
					break;
				case GET_N:
					if (value=='=')
					{
						if (get_numbuffer(file, buffer, 4))	//get num
						{
							this->N= atoi(buffer);
							received_params|=GOTN;
							state= (received_params == (GOTN | GOTM | GOTO)) ? ARRAYs : INITIAL;
						}
						else
							state=END;		//end if error
					}
					break;
				case GET_M:
					if (value=='=')
					{
						if (get_numbuffer(file, buffer, 4))	//get num
						{
							this->M= atoi(buffer);
							received_params|=GOTM;
							state= (received_params == (GOTN | GOTM | GOTO)) ? ARRAYs : INITIAL;
						}
						else
							state=END;		//end if error
					}
					break;
				case GET_a:
					if (value=='=')
					{
						float64_t f;

						transition_matrix_a=SG_MALLOC(float64_t, N*N);
						open_bracket(file);
						for (i=0; i<this->N; i++)
						{
							open_bracket(file);

							for (j=0; j<this->N ; j++)
							{

								if (fscanf( file, "%le", &f ) != 1)
									error(line, "float64_t expected");
								else
									set_a(i,j, f);

								if (j<this->N-1)
									comma_or_space(file);
								else
									close_bracket(file);
							}

							if (i<this->N-1)
								comma_or_space(file);
							else
								close_bracket(file);
						}
						received_params|=GOTa;
					}
					state= (received_params == (GOTa | GOTb | GOTp | GOTq)) ? END : ARRAYs;
					break;
				case GET_b:
					if (value=='=')
					{
						float64_t f;

						observation_matrix_b=SG_MALLOC(float64_t, N*M);
						open_bracket(file);
						for (i=0; i<this->N; i++)
						{
							open_bracket(file);

							for (j=0; j<this->M ; j++)
							{

								if (fscanf( file, "%le", &f ) != 1)
									error(line, "float64_t expected");
								else
									set_b(i,j, f);

								if (j<this->M-1)
									comma_or_space(file);
								else
									close_bracket(file);
							}

							if (i<this->N-1)
								comma_or_space(file);
							else
								close_bracket(file);
						}
						received_params|=GOTb;
					}
					state= ((received_params & (GOTa | GOTb | GOTp | GOTq)) == (GOTa | GOTb | GOTp | GOTq)) ? END : ARRAYs;
					break;
				case GET_p:
					if (value=='=')
					{
						float64_t f;

						initial_state_distribution_p=SG_MALLOC(float64_t, N);
						open_bracket(file);
						for (i=0; i<this->N ; i++)
						{
							if (fscanf( file, "%le", &f ) != 1)
								error(line, "float64_t expected");
							else
								set_p(i, f);

							if (i<this->N-1)
								comma_or_space(file);
							else
								close_bracket(file);
						}
						received_params|=GOTp;
					}
					state= (received_params == (GOTa | GOTb | GOTp | GOTq)) ? END : ARRAYs;
					break;
				case GET_q:
					if (value=='=')
					{
						float64_t f;

						end_state_distribution_q=SG_MALLOC(float64_t, N);
						open_bracket(file);
						for (i=0; i<this->N ; i++)
						{
							if (fscanf( file, "%le", &f ) != 1)
								error(line, "float64_t expected");
							else
								set_q(i, f);

							if (i<this->N-1)
								comma_or_space(file);
							else
								close_bracket(file);
						}
						received_params|=GOTq;
					}
					state= (received_params == (GOTa | GOTb | GOTp | GOTq)) ? END : ARRAYs;
					break;
				case COMMENT:
					if (value==EOF)
						state=END;
					else if (value=='\n')
					{
						line++;
						state=INITIAL;
					}
					break;

				default:
					break;
			}
		}
		result= (received_params== (GOTa | GOTb | GOTp | GOTq | GOTN | GOTM | GOTO));
	}

	SG_WARNING("not normalizing anymore, call normalize_hmm to make sure the hmm is valid!!\n")
	////////!!!!!!!!!!!!!!normalize();
	return result;
}

/*
	-format specs: train_file (train.trn)
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

	learn_a=[ [<int32_t>,<int32_t>];
	[<int32_t>,<int32_t>];
	[<int32_t>,<int32_t>];
	........
	[<int32_t>,<int32_t>];
	[-1,-1];
	];

	learn_b=[ [<int32_t>,<int32_t>];
	[<int32_t>,<int32_t>];
	[<int32_t>,<int32_t>];
	........
	[<int32_t>,<int32_t>];
	[-1,-1];
	];

	learn_p= [ <int32_t>, ... , <int32_t>, -1 ];
	learn_q= [ <int32_t>, ... , <int32_t>, -1 ];


	const_a=[ [<int32_t>,<int32_t>,<DOUBLE>];
	[<int32_t>,<int32_t>,<DOUBLE>];
	[<int32_t>,<int32_t>,<DOUBLE>];
	........
	[<int32_t>,<int32_t>,<DOUBLE>];
	[-1,-1,-1];
	];

	const_b=[ [<int32_t>,<int32_t>,<DOUBLE>];
	[<int32_t>,<int32_t>,<DOUBLE>];
	[<int32_t>,<int32_t>,<DOUBLE];
	........
	[<int32_t>,<int32_t>,<DOUBLE>];
	[-1,-1];
	];

	const_p[]=[ [<int32_t>, <DOUBLE>], ... , [<int32_t>,<DOUBLE>], [-1,-1] ];
	const_q[]=[ [<int32_t>, <DOUBLE>], ... , [<int32_t>,<DOUBLE>], [-1,-1] ];
	*/
bool CHMM::load_definitions(FILE* file, bool verbose, bool _initialize)
{
	if (model)
		delete model ;
	model=new Model();

	int32_t received_params=0x0000000;	//a,b,p,q,N,M,O
	char buffer[1024];

	bool result=false;
	E_STATE state=INITIAL;

	{ // do some useful initializations
		model->set_learn_a(0, -1);
		model->set_learn_a(1, -1);
		model->set_const_a(0, -1);
		model->set_const_a(1, -1);
		model->set_const_a_val(0, 1.0);
		model->set_learn_b(0, -1);
		model->set_const_b(0, -1);
		model->set_const_b_val(0, 1.0);
		model->set_learn_p(0, -1);
		model->set_learn_q(0, -1);
		model->set_const_p(0, -1);
		model->set_const_q(0, -1);
	} ;

	line=1;

	if (file)
	{
		while (state!=END)
		{
			int32_t value=fgetc(file);

			if (value=='\n')
				line++;

			if (value==EOF)
				state=END;

			switch (state)
			{
				case INITIAL:
					if (value=='l')
					{
						if (fgetc(file)=='e' && fgetc(file)=='a' && fgetc(file)=='r' && fgetc(file)=='n' && fgetc(file)=='_')
						{
							switch(fgetc(file))
							{
								case 'a':
									state=GET_learn_a;
									break;
								case 'b':
									state=GET_learn_b;
									break;
								case 'p':
									state=GET_learn_p;
									break;
								case 'q':
									state=GET_learn_q;
									break;
								default:
									error(line, "a,b,p or q expected in train definition file");
							};
						}
					}
					else if (value=='c')
					{
						if (fgetc(file)=='o' && fgetc(file)=='n' && fgetc(file)=='s'
								&& fgetc(file)=='t' && fgetc(file)=='_')
						{
							switch(fgetc(file))
							{
								case 'a':
									state=GET_const_a;
									break;
								case 'b':
									state=GET_const_b;
									break;
								case 'p':
									state=GET_const_p;
									break;
								case 'q':
									state=GET_const_q;
									break;
								default:
									error(line, "a,b,p or q expected in train definition file");
							};
						}
					}
					else if (value=='%')
					{
						state=COMMENT;
					}
					else if (value==EOF)
					{
						state=END;
					}
					break;
				case GET_learn_a:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						int32_t i=0;

						if (verbose)
							SG_DEBUG("\nlearn for transition matrix: ")
						while (!finished)
						{
							open_bracket(file);

							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);
								model->set_learn_a(i++, value);

								if (value<0)
								{
									finished=true;
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for learn_a(%i,0): %i\n",i/2,(int)value)
							}
							else
								break;

							comma_or_space(file);

							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);
								model->set_learn_a(i++, value);

								if (value<0)
								{
									finished=true;
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for learn_a(%i,1): %i\n",i/2-1,(int)value)

							}
							else
								break;
							close_bracket(file);
						}
						close_bracket(file);
						if (verbose)
							SG_DEBUG("%i Entries",(int)(i/2))

						if (finished)
						{
							received_params|=GOTlearn_a;

							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;
				case GET_learn_b:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						int32_t i=0;

						if (verbose)
							SG_DEBUG("\nlearn for emission matrix:   ")

						while (!finished)
						{
							open_bracket(file);

							int32_t combine=0;

							for (int32_t j=0; j<2; j++)
							{
								if (get_numbuffer(file, buffer, 4))   //get num
								{
									value=atoi(buffer);

									if (j==0)
									{
										model->set_learn_b(i++, value);

										if (value<0)
										{
											finished=true;
											break;
										}
										if (value>=N)
											SG_ERROR("invalid value for learn_b(%i,0): %i\n",i/2,(int)value)
									}
									else
										combine=value;
								}
								else
									break;

								if (j<1)
									comma_or_space(file);
								else
									close_bracket(file);
							}
							model->set_learn_b(i++, combine);
							if (combine>=M)

								SG_ERROR("invalid value for learn_b(%i,1): %i\n",i/2-1,(int)value)
						}
						close_bracket(file);
						if (verbose)
							SG_DEBUG("%i Entries",(int)(i/2-1))

						if (finished)
						{
							received_params|=GOTlearn_b;
							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;
				case GET_learn_p:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						int32_t i=0;

						if (verbose)
							SG_DEBUG("\nlearn start states: ")
						while (!finished)
						{
							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);

								model->set_learn_p(i++, value);

								if (value<0)
								{
									finished=true;
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for learn_p(%i): %i\n",i-1,(int)value)
							}
							else
								break;

							comma_or_space(file);
						}

						close_bracket(file);
						if (verbose)
							SG_DEBUG("%i Entries",i-1)

						if (finished)
						{
							received_params|=GOTlearn_p;
							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;
				case GET_learn_q:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						int32_t i=0;

						if (verbose)
							SG_DEBUG("\nlearn terminal states: ")
						while (!finished)
						{
							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);
								model->set_learn_q(i++, value);

								if (value<0)
								{
									finished=true;
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for learn_q(%i): %i\n",i-1,(int)value)
							}
							else
								break;

							comma_or_space(file);
						}

						close_bracket(file);
						if (verbose)
							SG_DEBUG("%i Entries",i-1)

						if (finished)
						{
							received_params|=GOTlearn_q;
							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;
				case GET_const_a:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						int32_t i=0;

						if (verbose)
#ifdef USE_HMMDEBUG
							SG_DEBUG("\nconst for transition matrix: \n")
#else
						SG_DEBUG("\nconst for transition matrix: ")
#endif
						while (!finished)
						{
							open_bracket(file);

							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);
								model->set_const_a(i++, value);

								if (value<0)
								{
									finished=true;
									model->set_const_a(i++, value);
									model->set_const_a_val((int32_t)i/2 - 1, value);
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for const_a(%i,0): %i\n",i/2,(int)value)
							}
							else
								break;

							comma_or_space(file);

							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);
								model->set_const_a(i++, value);

								if (value<0)
								{
									finished=true;
									model->set_const_a_val((int32_t)i/2 - 1, value);
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for const_a(%i,1): %i\n",i/2-1,(int)value)
							}
							else
								break;

							if (!comma_or_space(file))
								model->set_const_a_val((int32_t)i/2 - 1, 1.0);
							else
								if (get_numbuffer(file, buffer, 10))	//get num
								{
									float64_t dvalue=atof(buffer);
									model->set_const_a_val((int32_t)i/2 - 1, dvalue);
									if (dvalue<0)
									{
										finished=true;
										break;
									}
									if ((dvalue>1.0) || (dvalue<0.0))
										SG_ERROR("invalid value for const_a_val(%i): %e\n",(int)i/2-1,dvalue)
								}
								else
									model->set_const_a_val((int32_t)i/2 - 1, 1.0);

#ifdef USE_HMMDEBUG
							if (verbose)
								SG_ERROR("const_a(%i,%i)=%e\n", model->get_const_a((int32_t)i/2-1,0),model->get_const_a((int32_t)i/2-1,1),model->get_const_a_val((int32_t)i/2-1))
#endif
							close_bracket(file);
						}
						close_bracket(file);
						if (verbose)
							SG_DEBUG("%i Entries",(int)i/2-1)

						if (finished)
						{
							received_params|=GOTconst_a;
							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;

				case GET_const_b:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						int32_t i=0;

						if (verbose)
#ifdef USE_HMMDEBUG
							SG_DEBUG("\nconst for emission matrix:   \n")
#else
						SG_DEBUG("\nconst for emission matrix:   ")
#endif
						while (!finished)
						{
							open_bracket(file);
							int32_t combine=0;
							for (int32_t j=0; j<3; j++)
							{
								if (get_numbuffer(file, buffer, 10))	//get num
								{
									if (j==0)
									{
										value=atoi(buffer);

										model->set_const_b(i++, value);

										if (value<0)
										{
											finished=true;
											//model->set_const_b_val((int32_t)(i-1)/2, value);
											break;
										}
										if (value>=N)
											SG_ERROR("invalid value for const_b(%i,0): %i\n",i/2-1,(int)value)
									}
									else if (j==2)
									{
										float64_t dvalue=atof(buffer);
										model->set_const_b_val((int32_t)(i-1)/2, dvalue);
										if (dvalue<0)
										{
											finished=true;
											break;
										} ;
										if ((dvalue>1.0) || (dvalue<0.0))
											SG_ERROR("invalid value for const_b_val(%i,1): %e\n",i/2-1,dvalue)
									}
									else
									{
										value=atoi(buffer);
										combine= value;
									} ;
								}
								else
								{
									if (j==2)
										model->set_const_b_val((int32_t)(i-1)/2, 1.0);
									break;
								} ;
								if (j<2)
									if ((!comma_or_space(file)) && (j==1))
									{
										model->set_const_b_val((int32_t)(i-1)/2, 1.0) ;
										break ;
									} ;
							}
							close_bracket(file);
							model->set_const_b(i++, combine);
							if (combine>=M)
								SG_ERROR("invalid value for const_b(%i,1): %i\n",i/2-1, combine)
#ifdef USE_HMMDEBUG
							if (verbose && !finished)
								SG_ERROR("const_b(%i,%i)=%e\n", model->get_const_b((int32_t)i/2-1,0),model->get_const_b((int32_t)i/2-1,1),model->get_const_b_val((int32_t)i/2-1))
#endif
						}
						close_bracket(file);
						if (verbose)
							SG_ERROR("%i Entries",(int)i/2-1)

						if (finished)
						{
							received_params|=GOTconst_b;
							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;
				case GET_const_p:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						int32_t i=0;

						if (verbose)
#ifdef USE_HMMDEBUG
							SG_DEBUG("\nconst for start states:     \n")
#else
						SG_DEBUG("\nconst for start states:     ")
#endif
						while (!finished)
						{
							open_bracket(file);

							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);
								model->set_const_p(i, value);

								if (value<0)
								{
									finished=true;
									model->set_const_p_val(i++, value);
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for const_p(%i): %i\n",i,(int)value)

							}
							else
								break;

							if (!comma_or_space(file))
								model->set_const_p_val(i++, 1.0);
							else
								if (get_numbuffer(file, buffer, 10))	//get num
								{
									float64_t dvalue=atof(buffer);
									model->set_const_p_val(i++, dvalue);
									if (dvalue<0)
									{
										finished=true;
										break;
									}
									if ((dvalue>1) || (dvalue<0))
										SG_ERROR("invalid value for const_p_val(%i): %e\n",i,dvalue)
								}
								else
									model->set_const_p_val(i++, 1.0);

							close_bracket(file);

#ifdef USE_HMMDEBUG
							if (verbose)
								SG_DEBUG("const_p(%i)=%e\n", model->get_const_p(i-1),model->get_const_p_val(i-1))
#endif
						}
						if (verbose)
							SG_DEBUG("%i Entries",i-1)

						close_bracket(file);

						if (finished)
						{
							received_params|=GOTconst_p;
							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;
				case GET_const_q:
					if (value=='=')
					{
						open_bracket(file);
						bool finished=false;
						if (verbose)
#ifdef USE_HMMDEBUG
							SG_DEBUG("\nconst for terminal states: \n")
#else
						SG_DEBUG("\nconst for terminal states: ")
#endif
						int32_t i=0;

						while (!finished)
						{
							open_bracket(file) ;
							if (get_numbuffer(file, buffer, 4))	//get num
							{
								value=atoi(buffer);
								model->set_const_q(i, value);
								if (value<0)
								{
									finished=true;
									model->set_const_q_val(i++, value);
									break;
								}
								if (value>=N)
									SG_ERROR("invalid value for const_q(%i): %i\n",i,(int)value)
							}
							else
								break;

							if (!comma_or_space(file))
								model->set_const_q_val(i++, 1.0);
							else
								if (get_numbuffer(file, buffer, 10))	//get num
								{
									float64_t dvalue=atof(buffer);
									model->set_const_q_val(i++, dvalue);
									if (dvalue<0)
									{
										finished=true;
										break;
									}
									if ((dvalue>1) || (dvalue<0))
										SG_ERROR("invalid value for const_q_val(%i): %e\n",i,(double) dvalue)
								}
								else
									model->set_const_q_val(i++, 1.0);

							close_bracket(file);
#ifdef USE_HMMDEBUG
							if (verbose)
								SG_DEBUG("const_q(%i)=%e\n", model->get_const_q(i-1),model->get_const_q_val(i-1))
#endif
						}
						if (verbose)
							SG_DEBUG("%i Entries",i-1)

						close_bracket(file);

						if (finished)
						{
							received_params|=GOTconst_q;
							state= (received_params == (GOTlearn_a | GOTlearn_b | GOTlearn_p | GOTlearn_q |GOTconst_a | GOTconst_b | GOTconst_p | GOTconst_q)) ? END : INITIAL;
						}
						else
							state=END;
					}
					break;
				case COMMENT:
					if (value==EOF)
						state=END;
					else if (value=='\n')
						state=INITIAL;
					break;

				default:
					break;
			}
		}
	}

	/*result=((received_params&(GOTlearn_a | GOTconst_a))!=0) ;
	  result=result && ((received_params&(GOTlearn_b | GOTconst_b))!=0) ;
	  result=result && ((received_params&(GOTlearn_p | GOTconst_p))!=0) ;
	  result=result && ((received_params&(GOTlearn_q | GOTconst_q))!=0) ; */
	result=1 ;
	if (result)
	{
		model->sort_learn_a() ;
		model->sort_learn_b() ;
		if (_initialize)
		{
			init_model_defined(); ;
			convert_to_log();
		} ;
	}
	if (verbose)
		SG_DEBUG("\n")
	return result;
}

/*
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

   N=<int32_t>;
   M=<int32_t>;

   p=[<DOUBLE>,<DOUBLE>...<DOUBLE>];

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
   */

bool CHMM::save_model(FILE* file)
{
	bool result=false;
	int32_t i,j;
	const float32_t NAN_REPLACEMENT = (float32_t) CMath::ALMOST_NEG_INFTY ;

	if (file)
	{
		fprintf(file,"%s","% HMM - specification\n% N  - number of states\n% M  - number of observation_tokens\n% a is state_transition_matrix\n% size(a)= [N,N]\n%\n% b is observation_per_state_matrix\n% size(b)= [N,M]\n%\n% p is initial distribution\n% size(p)= [1, N]\n\n% q is distribution of end states\n% size(q)= [1, N]\n");
		fprintf(file,"N=%d;\n",N);
		fprintf(file,"M=%d;\n",M);

		fprintf(file,"p=[");
		for (i=0; i<N; i++)
		{
			if (i<N-1) {
				if (CMath::is_finite(get_p(i)))
					fprintf(file, "%e,", (double)get_p(i));
				else
					fprintf(file, "%f,", NAN_REPLACEMENT);
			}
			else {
				if (CMath::is_finite(get_p(i)))
					fprintf(file, "%e", (double)get_p(i));
				else
					fprintf(file, "%f", NAN_REPLACEMENT);
			}
		}

		fprintf(file,"];\n\nq=[");
		for (i=0; i<N; i++)
		{
			if (i<N-1) {
				if (CMath::is_finite(get_q(i)))
					fprintf(file, "%e,", (double)get_q(i));
				else
					fprintf(file, "%f,", NAN_REPLACEMENT);
			}
			else {
				if (CMath::is_finite(get_q(i)))
					fprintf(file, "%e", (double)get_q(i));
				else
					fprintf(file, "%f", NAN_REPLACEMENT);
			}
		}
		fprintf(file,"];\n\na=[");

		for (i=0; i<N; i++)
		{
			fprintf(file, "\t[");

			for (j=0; j<N; j++)
			{
				if (j<N-1) {
					if (CMath::is_finite(get_a(i,j)))
						fprintf(file, "%e,", (double)get_a(i,j));
					else
						fprintf(file, "%f,", NAN_REPLACEMENT);
				}
				else {
					if (CMath::is_finite(get_a(i,j)))
						fprintf(file, "%e];\n", (double)get_a(i,j));
					else
						fprintf(file, "%f];\n", NAN_REPLACEMENT);
				}
			}
		}

		fprintf(file,"  ];\n\nb=[");

		for (i=0; i<N; i++)
		{
			fprintf(file, "\t[");

			for (j=0; j<M; j++)
			{
				if (j<M-1) {
					if (CMath::is_finite(get_b(i,j)))
						fprintf(file, "%e,",  (double)get_b(i,j));
					else
						fprintf(file, "%f,", NAN_REPLACEMENT);
				}
				else {
					if (CMath::is_finite(get_b(i,j)))
						fprintf(file, "%e];\n", (double)get_b(i,j));
					else
						fprintf(file, "%f];\n", NAN_REPLACEMENT);
				}
			}
		}
		result= (fprintf(file,"  ];\n") == 5);
	}

	return result;
}

T_STATES* CHMM::get_path(int32_t dim, float64_t& prob)
{
	T_STATES* result = NULL;

	prob = best_path(dim);
	result = SG_MALLOC(T_STATES, p_observations->get_vector_length(dim));

	for (int32_t i=0; i<p_observations->get_vector_length(dim); i++)
		result[i]=PATH(dim)[i];

	return result;
}

bool CHMM::save_path(FILE* file)
{
	bool result=false;

	if (file)
	{
	  for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
	    {
	      if (dim%100==0)
		SG_PRINT("%i..", dim)
	      float64_t prob = best_path(dim);
	      fprintf(file,"%i. path probability:%e\nstate sequence:\n", dim, prob);
	      for (int32_t i=0; i<p_observations->get_vector_length(dim)-1; i++)
		fprintf(file,"%d ", PATH(dim)[i]);
	      fprintf(file,"%d", PATH(dim)[p_observations->get_vector_length(dim)-1]);
	      fprintf(file,"\n\n") ;
	    }
	  SG_DONE()
	  result=true;
	}

	return result;
}

bool CHMM::save_likelihood_bin(FILE* file)
{
	bool result=false;

	if (file)
	{
		for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
		{
			float32_t prob= (float32_t) model_probability(dim);
			fwrite(&prob, sizeof(float32_t), 1, file);
		}
		result=true;
	}

	return result;
}

bool CHMM::save_likelihood(FILE* file)
{
	bool result=false;

	if (file)
	{
		fprintf(file, "%% likelihood of model per observation\n%% P[O|model]=[ P[O|model]_1 P[O|model]_2 ... P[O|model]_dim ]\n%%\n");

		fprintf(file, "P=[");
		for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
			fprintf(file, "%e ", (double) model_probability(dim));

		fprintf(file,"];");
		result=true;
	}

	return result;
}

#define FLOATWRITE(file, value) { float32_t rrr=float32_t(value); fwrite(&rrr, sizeof(float32_t), 1, file); num_floats++;}

bool CHMM::save_model_bin(FILE* file)
{
	int32_t i,j,q, num_floats=0 ;
	if (!model)
	{
		if (file)
		{
			// write id
			FLOATWRITE(file, (float32_t)CMath::INFTY);
			FLOATWRITE(file, (float32_t) 1);

			//derivates log(dp),log(dq)
			for (i=0; i<N; i++)
				FLOATWRITE(file, get_p(i));
			SG_INFO("wrote %i parameters for p\n",N)

			for (i=0; i<N; i++)
				FLOATWRITE(file, get_q(i)) ;
			SG_INFO("wrote %i parameters for q\n",N)

			//derivates log(da),log(db)
			for (i=0; i<N; i++)
				for (j=0; j<N; j++)
					FLOATWRITE(file, get_a(i,j));
			SG_INFO("wrote %i parameters for a\n",N*N)

			for (i=0; i<N; i++)
				for (j=0; j<M; j++)
					FLOATWRITE(file, get_b(i,j));
			SG_INFO("wrote %i parameters for b\n",N*M)

			// write id
			FLOATWRITE(file, (float32_t)CMath::INFTY);
			FLOATWRITE(file, (float32_t) 3);

			// write number of parameters
			FLOATWRITE(file, (float32_t) N);
			FLOATWRITE(file, (float32_t) N);
			FLOATWRITE(file, (float32_t) N*N);
			FLOATWRITE(file, (float32_t) N*M);
			FLOATWRITE(file, (float32_t) N);
			FLOATWRITE(file, (float32_t) M);
		} ;
	}
	else
	{
		if (file)
		{
			int32_t num_p, num_q, num_a, num_b ;
			// write id
			FLOATWRITE(file, (float32_t)CMath::INFTY);
			FLOATWRITE(file, (float32_t) 2);

			for (i=0; model->get_learn_p(i)>=0; i++)
				FLOATWRITE(file, get_p(model->get_learn_p(i)));
			num_p=i ;
			SG_INFO("wrote %i parameters for p\n",num_p)

			for (i=0; model->get_learn_q(i)>=0; i++)
				FLOATWRITE(file, get_q(model->get_learn_q(i)));
			num_q=i ;
			SG_INFO("wrote %i parameters for q\n",num_q)

			//derivates log(da),log(db)
			for (q=0; model->get_learn_a(q,1)>=0; q++)
			{
				i=model->get_learn_a(q,0) ;
				j=model->get_learn_a(q,1) ;
				FLOATWRITE(file, (float32_t)i);
				FLOATWRITE(file, (float32_t)j);
				FLOATWRITE(file, get_a(i,j));
			} ;
			num_a=q ;
			SG_INFO("wrote %i parameters for a\n",num_a)

			for (q=0; model->get_learn_b(q,0)>=0; q++)
			{
				i=model->get_learn_b(q,0) ;
				j=model->get_learn_b(q,1) ;
				FLOATWRITE(file, (float32_t)i);
				FLOATWRITE(file, (float32_t)j);
				FLOATWRITE(file, get_b(i,j));
			} ;
			num_b=q ;
			SG_INFO("wrote %i parameters for b\n",num_b)

			// write id
			FLOATWRITE(file, (float32_t)CMath::INFTY);
			FLOATWRITE(file, (float32_t) 3);

			// write number of parameters
			FLOATWRITE(file, (float32_t) num_p);
			FLOATWRITE(file, (float32_t) num_q);
			FLOATWRITE(file, (float32_t) num_a);
			FLOATWRITE(file, (float32_t) num_b);
			FLOATWRITE(file, (float32_t) N);
			FLOATWRITE(file, (float32_t) M);
		} ;
	} ;
	return true ;
}

bool CHMM::save_path_derivatives(FILE* logfile)
{
	int32_t dim,i,j;

	if (logfile)
	{
		fprintf(logfile,"%% lambda denotes the model\n%% O denotes the observation sequence\n%% Q denotes the path\n%% \n%% calculating derivatives of P[O,Q|lambda]=p_{Q1}b_{Q1}(O_1}*a_{Q1}{Q2}b_{Q2}(O2)*...*q_{T-1}{T}b_{QT}(O_T}q_{Q_T} against p,q,a,b\n%%\n");
		fprintf(logfile,"%% dPr[...]=[ [dp_1,...,dp_N,dq_1,...,dq_N, da_11,da_12,..,da_1N,..,da_NN, db_11,.., db_NN]\n");
		fprintf(logfile,"%%            [dp_1,...,dp_N,dq_1,...,dq_N, da_11,da_12,..,da_1N,..,da_NN, db_11,.., db_NN]\n");
		fprintf(logfile,"%%                            .............................                                \n");
		fprintf(logfile,"%%            [dp_1,...,dp_N,dq_1,...,dq_N, da_11,da_12,..,da_1N,..,da_NN, db_11,.., db_MM]\n");
		fprintf(logfile,"%%          ];\n%%\n\ndPr(log()) = [\n");
	}
	else
		return false ;

	for (dim=0; dim<p_observations->get_num_vectors(); dim++)
	{
		best_path(dim);

		fprintf(logfile, "[ ");

		//derivates dlogp,dlogq
		for (i=0; i<N; i++)
			fprintf(logfile,"%e, ", (double) path_derivative_p(i,dim) );

		for (i=0; i<N; i++)
			fprintf(logfile,"%e, ", (double) path_derivative_q(i,dim) );

		//derivates dloga,dlogb
		for (i=0; i<N; i++)
			for (j=0; j<N; j++)
				fprintf(logfile, "%e,", (double) path_derivative_a(i,j,dim) );

		for (i=0; i<N; i++)
			for (j=0; j<M; j++)
				fprintf(logfile, "%e,", (double) path_derivative_b(i,j,dim) );

		fseek(logfile,ftell(logfile)-1,SEEK_SET);
		fprintf(logfile, " ];\n");
	}

	fprintf(logfile, "];");

	return true ;
}

bool CHMM::save_path_derivatives_bin(FILE* logfile)
{
	bool result=false;
	int32_t dim,i,j,q;
	float64_t prob=0 ;
	int32_t num_floats=0 ;

	float64_t sum_prob=0.0 ;
	if (!model)
		SG_WARNING("No definitions loaded -- writing derivatives of all weights\n")
	else
		SG_INFO("writing derivatives of changed weights only\n")

	for (dim=0; dim<p_observations->get_num_vectors(); dim++)
	{
		if (dim%100==0)
		{
			SG_PRINT(".")

		} ;

		prob=best_path(dim);
		sum_prob+=prob ;

		if (!model)
		{
			if (logfile)
			{
				// write prob
				FLOATWRITE(logfile, prob);

				for (i=0; i<N; i++)
					FLOATWRITE(logfile, path_derivative_p(i,dim));

				for (i=0; i<N; i++)
					FLOATWRITE(logfile, path_derivative_q(i,dim));

				for (i=0; i<N; i++)
					for (j=0; j<N; j++)
						FLOATWRITE(logfile, path_derivative_a(i,j,dim));

				for (i=0; i<N; i++)
					for (j=0; j<M; j++)
						FLOATWRITE(logfile, path_derivative_b(i,j,dim));

			}
		}
		else
		{
			if (logfile)
			{
				// write prob
				FLOATWRITE(logfile, prob);

				for (i=0; model->get_learn_p(i)>=0; i++)
					FLOATWRITE(logfile, path_derivative_p(model->get_learn_p(i),dim));

				for (i=0; model->get_learn_q(i)>=0; i++)
					FLOATWRITE(logfile, path_derivative_q(model->get_learn_q(i),dim));

				for (q=0; model->get_learn_a(q,0)>=0; q++)
				{
					i=model->get_learn_a(q,0) ;
					j=model->get_learn_a(q,1) ;
					FLOATWRITE(logfile, path_derivative_a(i,j,dim));
				} ;

				for (q=0; model->get_learn_b(q,0)>=0; q++)
				{
					i=model->get_learn_b(q,0) ;
					j=model->get_learn_b(q,1) ;
					FLOATWRITE(logfile, path_derivative_b(i,j,dim));
				} ;
			}
		} ;
	}
	save_model_bin(logfile) ;

	result=true;
	SG_PRINT("\n")
	return result;
}

bool CHMM::save_model_derivatives_bin(FILE* file)
{
	bool result=false;
	int32_t dim,i,j,q ;
	int32_t num_floats=0 ;

	if (!model)
		SG_WARNING("No definitions loaded -- writing derivatives of all weights\n")
	else
		SG_INFO("writing derivatives of changed weights only\n")

#ifdef USE_HMMPARALLEL
	int32_t num_threads = parallel->get_num_threads();
	pthread_t *threads=SG_MALLOC(pthread_t, num_threads);
	S_DIM_THREAD_PARAM *params=SG_MALLOC(S_DIM_THREAD_PARAM, num_threads);

	if (p_observations->get_num_vectors()<num_threads)
		num_threads=p_observations->get_num_vectors();
#endif

	for (dim=0; dim<p_observations->get_num_vectors(); dim++)
	{
		if (dim%20==0)
		{
			SG_PRINT(".")

		} ;

#ifdef USE_HMMPARALLEL
		if (dim%num_threads==0)
		{
			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
				{
					params[i].hmm=this ;
					params[i].dim=dim+i ;
					pthread_create(&threads[i], NULL, bw_dim_prefetch, (void*)&params[i]) ;
				}
			}

			for (i=0; i<num_threads; i++)
			{
				if (dim+i<p_observations->get_num_vectors())
					pthread_join(threads[i], NULL);
			}
		}
#endif

		float64_t prob=model_probability(dim) ;
		if (!model)
		{
			if (file)
			{
				// write prob
				FLOATWRITE(file, prob);

				//derivates log(dp),log(dq)
				for (i=0; i<N; i++)
					FLOATWRITE(file, model_derivative_p(i,dim));

				for (i=0; i<N; i++)
					FLOATWRITE(file, model_derivative_q(i,dim));

				//derivates log(da),log(db)
				for (i=0; i<N; i++)
					for (j=0; j<N; j++)
						FLOATWRITE(file, model_derivative_a(i,j,dim));

				for (i=0; i<N; i++)
					for (j=0; j<M; j++)
						FLOATWRITE(file, model_derivative_b(i,j,dim));

				if (dim==0)
					SG_INFO("Number of parameters (including posterior prob.): %i\n", num_floats)
			} ;
		}
		else
		{
			if (file)
			{
				// write prob
				FLOATWRITE(file, prob);

				for (i=0; model->get_learn_p(i)>=0; i++)
					FLOATWRITE(file, model_derivative_p(model->get_learn_p(i),dim));

				for (i=0; model->get_learn_q(i)>=0; i++)
					FLOATWRITE(file, model_derivative_q(model->get_learn_q(i),dim));

				//derivates log(da),log(db)
				for (q=0; model->get_learn_a(q,1)>=0; q++)
				{
					i=model->get_learn_a(q,0) ;
					j=model->get_learn_a(q,1) ;
					FLOATWRITE(file, model_derivative_a(i,j,dim));
				} ;

				for (q=0; model->get_learn_b(q,0)>=0; q++)
				{
					i=model->get_learn_b(q,0) ;
					j=model->get_learn_b(q,1) ;
					FLOATWRITE(file, model_derivative_b(i,j,dim));
				} ;
				if (dim==0)
					SG_INFO("Number of parameters (including posterior prob.): %i\n", num_floats)
			} ;
		} ;
	}
	save_model_bin(file) ;

#ifdef USE_HMMPARALLEL
	SG_FREE(threads);
	SG_FREE(params);
#endif

	result=true;
	SG_PRINT("\n")
	return result;
}

bool CHMM::save_model_derivatives(FILE* file)
{
	bool result=false;
	int32_t dim,i,j;

	if (file)
	{

		fprintf(file,"%% lambda denotes the model\n%% O denotes the observation sequence\n%% Q denotes the path\n%%\n%% calculating derivatives of P[O|lambda]=sum_{all Q}p_{Q1}b_{Q1}(O_1}*a_{Q1}{Q2}b_{Q2}(O2)*...*q_{T-1}{T}b_{QT}(O_T}q_{Q_T} against p,q,a,b\n%%\n");
		fprintf(file,"%% dPr[...]=[ [dp_1,...,dp_N,dq_1,...,dq_N, da_11,da_12,..,da_1N,..,da_NN, db_11,.., db_NN]\n");
		fprintf(file,"%%            [dp_1,...,dp_N,dq_1,...,dq_N, da_11,da_12,..,da_1N,..,da_NN, db_11,.., db_NN]\n");
		fprintf(file,"%%                            .............................                                \n");
		fprintf(file,"%%            [dp_1,...,dp_N,dq_1,...,dq_N, da_11,da_12,..,da_1N,..,da_NN, db_11,.., db_MM]\n");
		fprintf(file,"%%          ];\n%%\n\nlog(dPr) = [\n");


		for (dim=0; dim<p_observations->get_num_vectors(); dim++)
		{
			fprintf(file, "[ ");

			//derivates log(dp),log(dq)
			for (i=0; i<N; i++)
				fprintf(file,"%e, ", (double) model_derivative_p(i, dim) );		//log (dp)

			for (i=0; i<N; i++)
				fprintf(file,"%e, ", (double) model_derivative_q(i, dim) );	//log (dq)

			//derivates log(da),log(db)
			for (i=0; i<N; i++)
				for (j=0; j<N; j++)
					fprintf(file, "%e,", (double) model_derivative_a(i,j,dim) );

			for (i=0; i<N; i++)
				for (j=0; j<M; j++)
					fprintf(file, "%e,", (double) model_derivative_b(i,j,dim) );

			fseek(file,ftell(file)-1,SEEK_SET);
			fprintf(file, " ];\n");
		}


		fprintf(file, "];");

		result=true;
	}
	return result;
}

bool CHMM::check_model_derivatives_combined()
{
	//	bool result=false;
	const float64_t delta=5e-4 ;

	int32_t i ;
	//derivates log(da)
	/*	for (i=0; i<N; i++)
		{
		for (int32_t j=0; j<N; j++)
		{
		float64_t old_a=get_a(i,j) ;

		set_a(i,j, log(exp(old_a)-delta)) ;
		invalidate_model() ;
		float64_t prob_old=exp(model_probability(-1)*p_observations->get_num_vectors()) ;

		set_a(i,j, log(exp(old_a)+delta)) ;
		invalidate_model() ;
		float64_t prob_new=exp(model_probability(-1)*p_observations->get_num_vectors());

		float64_t deriv = (prob_new-prob_old)/(2*delta) ;

		set_a(i,j, old_a) ;
		invalidate_model() ;

		float64_t prod_prob=model_probability(-1)*p_observations->get_num_vectors() ;

		float64_t deriv_calc=0 ;
		for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
		deriv_calc+=exp(model_derivative_a(i, j, dim)+
		prod_prob-model_probability(dim)) ;

		SG_DEBUG("da(%i,%i) = %e:%e\t (%1.5f%%)\n", i,j, deriv_calc,  deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
		} ;
		} ;*/
	//derivates log(db)
	i=0;//for (i=0; i<N; i++)
	{
		for (int32_t j=0; j<M; j++)
		{
			float64_t old_b=get_b(i,j) ;

			set_b(i,j, log(exp(old_b)-delta)) ;
			invalidate_model() ;
			float64_t prob_old=(model_probability(-1)*p_observations->get_num_vectors()) ;

			set_b(i,j, log(exp(old_b)+delta)) ;
			invalidate_model() ;
			float64_t prob_new=(model_probability(-1)*p_observations->get_num_vectors());

			float64_t deriv = (prob_new-prob_old)/(2*delta) ;

			set_b(i,j, old_b) ;
			invalidate_model() ;

			float64_t deriv_calc=0 ;
			for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
			{
				deriv_calc+=exp(model_derivative_b(i, j, dim)-model_probability(dim)) ;
				if (j==1)
					SG_INFO("deriv_calc[%i]=%e\n",dim,deriv_calc)
			} ;

			SG_ERROR("b(%i,%i)=%e  db(%i,%i) = %e:%e\t (%1.5f%%)\n", i,j,exp(old_b),i,j, deriv_calc,  deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
		} ;
	} ;
	return true ;
}

bool CHMM::check_model_derivatives()
{
	bool result=false;
	const float64_t delta=3e-4 ;

	for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
	{
		int32_t i ;
		//derivates log(dp),log(dq)
		for (i=0; i<N; i++)
		{
			for (int32_t j=0; j<N; j++)
			{
				float64_t old_a=get_a(i,j) ;

				set_a(i,j, log(exp(old_a)-delta)) ;
				invalidate_model() ;
				float64_t prob_old=exp(model_probability(dim)) ;

				set_a(i,j, log(exp(old_a)+delta)) ;
				invalidate_model() ;
				float64_t prob_new=exp(model_probability(dim));

				float64_t deriv = (prob_new-prob_old)/(2*delta) ;

				set_a(i,j, old_a) ;
				invalidate_model() ;
				float64_t deriv_calc=exp(model_derivative_a(i, j, dim)) ;

				SG_DEBUG("da(%i,%i) = %e:%e\t (%1.5f%%)\n", i,j, deriv_calc,  deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
				invalidate_model() ;
			} ;
		} ;
		for (i=0; i<N; i++)
		{
			for (int32_t j=0; j<M; j++)
			{
				float64_t old_b=get_b(i,j) ;

				set_b(i,j, log(exp(old_b)-delta)) ;
				invalidate_model() ;
				float64_t prob_old=exp(model_probability(dim)) ;

				set_b(i,j, log(exp(old_b)+delta)) ;
				invalidate_model() ;
				float64_t prob_new=exp(model_probability(dim));

				float64_t deriv = (prob_new-prob_old)/(2*delta) ;

				set_b(i,j, old_b) ;
				invalidate_model() ;
				float64_t deriv_calc=exp(model_derivative_b(i, j, dim));

				SG_DEBUG("db(%i,%i) = %e:%e\t (%1.5f%%)\n", i,j, deriv_calc, deriv, 100.0*(deriv-deriv_calc)/(deriv_calc))
			} ;
		} ;

#ifdef TEST
		for (i=0; i<N; i++)
		{
			float64_t old_p=get_p(i) ;

			set_p(i, log(exp(old_p)-delta)) ;
			invalidate_model() ;
			float64_t prob_old=exp(model_probability(dim)) ;

			set_p(i, log(exp(old_p)+delta)) ;
			invalidate_model() ;
			float64_t prob_new=exp(model_probability(dim));
			float64_t deriv = (prob_new-prob_old)/(2*delta) ;

			set_p(i, old_p) ;
			invalidate_model() ;
			float64_t deriv_calc=exp(model_derivative_p(i, dim));

			//if (fabs(deriv_calc_old-deriv)>1e-4)
			SG_DEBUG("dp(%i) = %e:%e\t (%1.5f%%)\n", i, deriv_calc, deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
		} ;
		for (i=0; i<N; i++)
		{
			float64_t old_q=get_q(i) ;

			set_q(i, log(exp(old_q)-delta)) ;
			invalidate_model() ;
			float64_t prob_old=exp(model_probability(dim)) ;

			set_q(i, log(exp(old_q)+delta)) ;
			invalidate_model() ;
			float64_t prob_new=exp(model_probability(dim));

			float64_t deriv = (prob_new-prob_old)/(2*delta) ;

			set_q(i, old_q) ;
			invalidate_model() ;
			float64_t deriv_calc=exp(model_derivative_q(i, dim));

			//if (fabs(deriv_calc_old-deriv)>1e-4)
			SG_DEBUG("dq(%i) = %e:%e\t (%1.5f%%)\n", i, deriv_calc, deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
		} ;
#endif
	}
	return result;
}

#ifdef USE_HMMDEBUG
bool CHMM::check_path_derivatives()
{
	bool result=false;
	const float64_t delta=1e-4 ;

	for (int32_t dim=0; dim<p_observations->get_num_vectors(); dim++)
	{
		int32_t i ;
		//derivates log(dp),log(dq)
		for (i=0; i<N; i++)
		{
			for (int32_t j=0; j<N; j++)
			{
				float64_t old_a=get_a(i,j) ;

				set_a(i,j, log(exp(old_a)-delta)) ;
				invalidate_model() ;
				float64_t prob_old=best_path(dim) ;

				set_a(i,j, log(exp(old_a)+delta)) ;
				invalidate_model() ;
				float64_t prob_new=best_path(dim);

				float64_t deriv = (prob_new-prob_old)/(2*delta) ;

				set_a(i,j, old_a) ;
				invalidate_model() ;
				float64_t deriv_calc=path_derivative_a(i, j, dim) ;

				SG_DEBUG("da(%i,%i) = %e:%e\t (%1.5f%%)\n", i,j, deriv_calc,  deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
			} ;
		} ;
		for (i=0; i<N; i++)
		{
			for (int32_t j=0; j<M; j++)
			{
				float64_t old_b=get_b(i,j) ;

				set_b(i,j, log(exp(old_b)-delta)) ;
				invalidate_model() ;
				float64_t prob_old=best_path(dim) ;

				set_b(i,j, log(exp(old_b)+delta)) ;
				invalidate_model() ;
				float64_t prob_new=best_path(dim);

				float64_t deriv = (prob_new-prob_old)/(2*delta) ;

				set_b(i,j, old_b) ;
				invalidate_model() ;
				float64_t deriv_calc=path_derivative_b(i, j, dim);

				SG_DEBUG("db(%i,%i) = %e:%e\t (%1.5f%%)\n", i,j, deriv_calc, deriv, 100.0*(deriv-deriv_calc)/(deriv_calc))
			} ;
		} ;

		for (i=0; i<N; i++)
		{
			float64_t old_p=get_p(i) ;

			set_p(i, log(exp(old_p)-delta)) ;
			invalidate_model() ;
			float64_t prob_old=best_path(dim) ;

			set_p(i, log(exp(old_p)+delta)) ;
			invalidate_model() ;
			float64_t prob_new=best_path(dim);
			float64_t deriv = (prob_new-prob_old)/(2*delta) ;

			set_p(i, old_p) ;
			invalidate_model() ;
			float64_t deriv_calc=path_derivative_p(i, dim);

			//if (fabs(deriv_calc_old-deriv)>1e-4)
			SG_DEBUG("dp(%i) = %e:%e\t (%1.5f%%)\n", i, deriv_calc, deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
		} ;
		for (i=0; i<N; i++)
		{
			float64_t old_q=get_q(i) ;

			set_q(i, log(exp(old_q)-delta)) ;
			invalidate_model() ;
			float64_t prob_old=best_path(dim) ;

			set_q(i, log(exp(old_q)+delta)) ;
			invalidate_model() ;
			float64_t prob_new=best_path(dim);

			float64_t deriv = (prob_new-prob_old)/(2*delta) ;

			set_q(i, old_q) ;
			invalidate_model() ;
			float64_t deriv_calc=path_derivative_q(i, dim);

			//if (fabs(deriv_calc_old-deriv)>1e-4)
			SG_DEBUG("dq(%i) = %e:%e\t (%1.5f%%)\n", i, deriv_calc, deriv, 100.0*(deriv-deriv_calc)/deriv_calc)
		} ;
	}
	return result;
}
#endif // USE_HMMDEBUG

//normalize model (sum to one constraint)
void CHMM::normalize(bool keep_dead_states)
{
	int32_t i,j;
	const float64_t INF=-1e10;
	float64_t sum_p =INF;

	for (i=0; i<N; i++)
	{
		sum_p=CMath::logarithmic_sum(sum_p, get_p(i));

		float64_t sum_b =INF;
		float64_t sum_a =get_q(i);

		for (j=0; j<N; j++)
			sum_a=CMath::logarithmic_sum(sum_a, get_a(i,j));

		if (sum_a>CMath::ALMOST_NEG_INFTY/N || (!keep_dead_states) )
		{
			for (j=0; j<N; j++)
				set_a(i,j, get_a(i,j)-sum_a);
			set_q(i, get_q(i)-sum_a);
		}

		for (j=0; j<M; j++)
			sum_b=CMath::logarithmic_sum(sum_b, get_b(i,j));
		for (j=0; j<M; j++)
			set_b(i,j, get_b(i,j)-sum_b);
	}

	for (i=0; i<N; i++)
		set_p(i, get_p(i)-sum_p);

	invalidate_model();
}

bool CHMM::append_model(CHMM* app_model)
{
	bool result=false;
	const int32_t num_states=app_model->get_N();
	int32_t i,j;

	SG_DEBUG("cur N:%d M:%d\n", N, M)
	SG_DEBUG("old N:%d M:%d\n", app_model->get_N(), app_model->get_M())
	if (app_model->get_M() == get_M())
	{
		float64_t* n_p=SG_MALLOC(float64_t, N+num_states);
		float64_t* n_q=SG_MALLOC(float64_t, N+num_states);
		float64_t* n_a=SG_MALLOC(float64_t, (N+num_states)*(N+num_states));
		//SG_PRINT("size n_b: %d\n", (N+num_states)*M)
		float64_t* n_b=SG_MALLOC(float64_t, (N+num_states)*M);

		//clear n_x
		for (i=0; i<N+num_states; i++)
		{
			n_p[i]=-CMath::INFTY;
			n_q[i]=-CMath::INFTY;

			for (j=0; j<N+num_states; j++)
				n_a[(N+num_states)*i+j]=-CMath::INFTY;

			for (j=0; j<M; j++)
				n_b[M*i+j]=-CMath::INFTY;
		}

		//copy models first
		// warning pay attention to the ordering of
		// transition_matrix_a, observation_matrix_b !!!

		// cur_model
		for (i=0; i<N; i++)
		{
			n_p[i]=get_p(i);

			for (j=0; j<N; j++)
				n_a[(N+num_states)*j+i]=get_a(i,j);

			for (j=0; j<M; j++)
			{
				n_b[M*i+j]=get_b(i,j);
			}
		}

		// append_model
		for (i=0; i<app_model->get_N(); i++)
		{
			n_q[i+N]=app_model->get_q(i);

			for (j=0; j<app_model->get_N(); j++)
				n_a[(N+num_states)*(j+N)+(i+N)]=app_model->get_a(i,j);
			for (j=0; j<app_model->get_M(); j++)
				n_b[M*(i+N)+j]=app_model->get_b(i,j);
		}


		// transition to the two and back
		for (i=0; i<N; i++)
		{
			for (j=N; j<N+num_states; j++)
				n_a[(N+num_states)*j + i]=CMath::logarithmic_sum(get_q(i)+app_model->get_p(j-N), n_a[(N+num_states)*j + i]);
		}

		free_state_dependend_arrays();
		N+=num_states;

		alloc_state_dependend_arrays();

		//delete + adjust pointers
		SG_FREE(initial_state_distribution_p);
		SG_FREE(end_state_distribution_q);
		SG_FREE(transition_matrix_a);
		SG_FREE(observation_matrix_b);

		transition_matrix_a=n_a;
		observation_matrix_b=n_b;
		initial_state_distribution_p=n_p;
		end_state_distribution_q=n_q;

		SG_WARNING("not normalizing anymore, call normalize_hmm to make sure the hmm is valid!!\n")
		////////!!!!!!!!!!!!!!normalize();
		invalidate_model();
	}
	else
		SG_ERROR("number of observations is different for append model, doing nothing!\n")

	return result;
}

bool CHMM::append_model(CHMM* app_model, float64_t* cur_out, float64_t* app_out)
{
	bool result=false;
	const int32_t num_states=app_model->get_N()+2;
	int32_t i,j;

	if (app_model->get_M() == get_M())
	{
		float64_t* n_p=SG_MALLOC(float64_t, N+num_states);
		float64_t* n_q=SG_MALLOC(float64_t, N+num_states);
		float64_t* n_a=SG_MALLOC(float64_t, (N+num_states)*(N+num_states));
		//SG_PRINT("size n_b: %d\n", (N+num_states)*M)
		float64_t* n_b=SG_MALLOC(float64_t, (N+num_states)*M);

		//clear n_x
		for (i=0; i<N+num_states; i++)
		{
			n_p[i]=-CMath::INFTY;
			n_q[i]=-CMath::INFTY;

			for (j=0; j<N+num_states; j++)
				n_a[(N+num_states)*j+i]=-CMath::INFTY;

			for (j=0; j<M; j++)
				n_b[M*i+j]=-CMath::INFTY;
		}

		//copy models first
		// warning pay attention to the ordering of
		// transition_matrix_a, observation_matrix_b !!!

		// cur_model
		for (i=0; i<N; i++)
		{
			n_p[i]=get_p(i);

			for (j=0; j<N; j++)
				n_a[(N+num_states)*j+i]=get_a(i,j);

			for (j=0; j<M; j++)
			{
				n_b[M*i+j]=get_b(i,j);
			}
		}

		// append_model
		for (i=0; i<app_model->get_N(); i++)
		{
			n_q[i+N+2]=app_model->get_q(i);

			for (j=0; j<app_model->get_N(); j++)
				n_a[(N+num_states)*(j+N+2)+(i+N+2)]=app_model->get_a(i,j);
			for (j=0; j<app_model->get_M(); j++)
				n_b[M*(i+N+2)+j]=app_model->get_b(i,j);
		}

		//initialize the two special states

		// output
		for (i=0; i<M; i++)
		{
			n_b[M*N+i]=cur_out[i];
			n_b[M*(N+1)+i]=app_out[i];
		}

		// transition to the two and back
		for (i=0; i<N+num_states; i++)
		{
			// the first state is only connected to the second
			if (i==N+1)
				n_a[(N+num_states)*i + N]=0;

			// only states of the cur_model can reach the
			// first state
			if ((uint32_t)i<(uint32_t)N)
				n_a[(N+num_states)*N+i]=get_q(i);

			// the second state is only connected to states of
			// the append_model (with probab app->p(i))
			if (i>=N+2)
				n_a[(N+num_states)*i+(N+1)]=app_model->get_p(i-(N+2));
		}

		free_state_dependend_arrays();
		N+=num_states;

		alloc_state_dependend_arrays();

		//delete + adjust pointers
		SG_FREE(initial_state_distribution_p);
		SG_FREE(end_state_distribution_q);
		SG_FREE(transition_matrix_a);
		SG_FREE(observation_matrix_b);

		transition_matrix_a=n_a;
		observation_matrix_b=n_b;
		initial_state_distribution_p=n_p;
		end_state_distribution_q=n_q;

		SG_WARNING("not normalizing anymore, call normalize_hmm to make sure the hmm is valid!!\n")
		////////!!!!!!!!!!!!!!normalize();
		invalidate_model();
	}

	return result;
}


void CHMM::add_states(int32_t num_states, float64_t default_value)
{
	int32_t i,j;
	const float64_t MIN_RAND=1e-2; //this is the range of the random values for the new variables
	const float64_t MAX_RAND=2e-1;

	float64_t* n_p=SG_MALLOC(float64_t, N+num_states);
	float64_t* n_q=SG_MALLOC(float64_t, N+num_states);
	float64_t* n_a=SG_MALLOC(float64_t, (N+num_states)*(N+num_states));
	//SG_PRINT("size n_b: %d\n", (N+num_states)*M)
	float64_t* n_b=SG_MALLOC(float64_t, (N+num_states)*M);

	// warning pay attention to the ordering of
	// transition_matrix_a, observation_matrix_b !!!
	for (i=0; i<N; i++)
	{
		n_p[i]=get_p(i);
		n_q[i]=get_q(i);

		for (j=0; j<N; j++)
			n_a[(N+num_states)*j+i]=get_a(i,j);

		for (j=0; j<M; j++)
			n_b[M*i+j]=get_b(i,j);
	}

	for (i=N; i<N+num_states; i++)
	{
		n_p[i]=VAL_MACRO;
		n_q[i]=VAL_MACRO;

		for (j=0; j<N; j++)
			n_a[(N+num_states)*i+j]=VAL_MACRO;

		for (j=0; j<N+num_states; j++)
			n_a[(N+num_states)*j+i]=VAL_MACRO;

		for (j=0; j<M; j++)
			n_b[M*i+j]=VAL_MACRO;
	}
	free_state_dependend_arrays();
	N+=num_states;

	alloc_state_dependend_arrays();

	//delete + adjust pointers
	SG_FREE(initial_state_distribution_p);
	SG_FREE(end_state_distribution_q);
	SG_FREE(transition_matrix_a);
	SG_FREE(observation_matrix_b);

	transition_matrix_a=n_a;
	observation_matrix_b=n_b;
	initial_state_distribution_p=n_p;
	end_state_distribution_q=n_q;

	invalidate_model();
	normalize();
}

void CHMM::chop(float64_t value)
{
	for (int32_t i=0; i<N; i++)
	{
		int32_t j;

		if (exp(get_p(i)) < value)
			set_p(i, CMath::ALMOST_NEG_INFTY);

		if (exp(get_q(i)) < value)
			set_q(i, CMath::ALMOST_NEG_INFTY);

		for (j=0; j<N; j++)
		{
			if (exp(get_a(i,j)) < value)
				set_a(i,j, CMath::ALMOST_NEG_INFTY);
		}

		for (j=0; j<M; j++)
		{
			if (exp(get_b(i,j)) < value)
				set_b(i,j, CMath::ALMOST_NEG_INFTY);
		}
	}
	normalize();
	invalidate_model();
}

bool CHMM::linear_train(bool right_align)
{
	if (p_observations)
	{
		int32_t histsize=(get_M()*get_N());
		int32_t* hist=SG_MALLOC(int32_t, histsize);
		int32_t* startendhist=SG_MALLOC(int32_t, get_N());
		int32_t i,dim;

		ASSERT(p_observations->get_max_vector_length()<=get_N())

		for (i=0; i<histsize; i++)
			hist[i]=0;

		for (i=0; i<get_N(); i++)
			startendhist[i]=0;

		if (right_align)
		{
			for (dim=0; dim<p_observations->get_num_vectors(); dim++)
			{
				int32_t len=0;
				bool free_vec;
				uint16_t* obs=p_observations->get_feature_vector(dim, len, free_vec);

				ASSERT(len<=get_N())
				startendhist[(get_N()-len)]++;

				for (i=0;i<len;i++)
					hist[(get_N()-len+i)*get_M() + *obs++]++;

				p_observations->free_feature_vector(obs, dim, free_vec);
			}

			set_q(get_N()-1, 1);
			for (i=0; i<get_N()-1; i++)
				set_q(i, 0);

			for (i=0; i<get_N(); i++)
				set_p(i, startendhist[i]+PSEUDO);

			for (i=0;i<get_N();i++)
			{
				for (int32_t j=0; j<get_N(); j++)
				{
					if (i==j-1)
						set_a(i,j, 1);
					else
						set_a(i,j, 0);
				}
			}
		}
		else
		{
			for (dim=0; dim<p_observations->get_num_vectors(); dim++)
			{
				int32_t len=0;
				bool free_vec;
				uint16_t* obs=p_observations->get_feature_vector(dim, len, free_vec);

				ASSERT(len<=get_N())
				for (i=0;i<len;i++)
					hist[i*get_M() + *obs++]++;

				startendhist[len-1]++;

				p_observations->free_feature_vector(obs, dim, free_vec);
			}

			set_p(0, 1);
			for (i=1; i<get_N(); i++)
				set_p(i, 0);

			for (i=0; i<get_N(); i++)
				set_q(i, startendhist[i]+PSEUDO);

			int32_t total=p_observations->get_num_vectors();

			for (i=0;i<get_N();i++)
			{
				total-= startendhist[i] ;

				for (int32_t j=0; j<get_N(); j++)
				{
					if (i==j-1)
						set_a(i,j, total+PSEUDO);
					else
						set_a(i,j, 0);
				}
			}
			ASSERT(total==0)
		}

		for (i=0;i<get_N();i++)
		{
			for (int32_t j=0; j<get_M(); j++)
			{
				float64_t sum=0;
				int32_t offs=i*get_M()+ p_observations->get_masked_symbols((uint16_t) j, (uint8_t) 254);

				for (int32_t k=0; k<p_observations->get_original_num_symbols(); k++)
					sum+=hist[offs+k];

				set_b(i,j, (PSEUDO+hist[i*get_M()+j])/(sum+PSEUDO*p_observations->get_original_num_symbols()));
			}
		}

		SG_FREE(hist);
		SG_FREE(startendhist);
		convert_to_log();
		invalidate_model();
		return true;
	}
	else
		return false;
}

void CHMM::set_observation_nocache(CStringFeatures<uint16_t>* obs)
{
	ASSERT(obs)
	p_observations=obs;
	SG_REF(obs);

	if (obs)
		if (obs->get_num_symbols() > M)
			SG_ERROR("number of symbols in observation (%ld) larger than M (%d)\n", (long) obs->get_num_symbols(), M)

	if (!reused_caches)
	{
#ifdef USE_HMMPARALLEL_STRUCTURES
		for (int32_t i=0; i<parallel->get_num_threads(); i++)
		{
			SG_FREE(alpha_cache[i].table);
			SG_FREE(beta_cache[i].table);
			SG_FREE(states_per_observation_psi[i]);
			SG_FREE(path[i]);

			alpha_cache[i].table=NULL;
			beta_cache[i].table=NULL;
			states_per_observation_psi[i]=NULL;
			path[i]=NULL;
		} ;
#else
		SG_FREE(alpha_cache.table);
		SG_FREE(beta_cache.table);
		SG_FREE(states_per_observation_psi);
		SG_FREE(path);

		alpha_cache.table=NULL;
		beta_cache.table=NULL;
		states_per_observation_psi=NULL;
		path=NULL;

#endif //USE_HMMPARALLEL_STRUCTURES
	}

	invalidate_model();
}

void CHMM::set_observations(CStringFeatures<uint16_t>* obs, CHMM* lambda)
{
	ASSERT(obs)
	SG_REF(obs);
	p_observations=obs;

	/* from Distribution, necessary for calls to base class methods, like
	 * get_log_likelihood_sample():
	 */
	SG_REF(obs);
	features=obs;

	SG_DEBUG("num symbols alphabet: %ld\n", obs->get_alphabet()->get_num_symbols())
	SG_DEBUG("num symbols: %ld\n", obs->get_num_symbols())
	SG_DEBUG("M: %d\n", M)

	if (obs)
	{
		if (obs->get_num_symbols() > M)
		{
			SG_ERROR("number of symbols in observation (%ld) larger than M (%d)\n", (long) obs->get_num_symbols(), M)
		}
	}

	if (!reused_caches)
	{
#ifdef USE_HMMPARALLEL_STRUCTURES
		for (int32_t i=0; i<parallel->get_num_threads(); i++)
		{
			SG_FREE(alpha_cache[i].table);
			SG_FREE(beta_cache[i].table);
			SG_FREE(states_per_observation_psi[i]);
			SG_FREE(path[i]);

			alpha_cache[i].table=NULL;
			beta_cache[i].table=NULL;
			states_per_observation_psi[i]=NULL;
			path[i]=NULL;
		} ;
#else
		SG_FREE(alpha_cache.table);
		SG_FREE(beta_cache.table);
		SG_FREE(states_per_observation_psi);
		SG_FREE(path);

		alpha_cache.table=NULL;
		beta_cache.table=NULL;
		states_per_observation_psi=NULL;
		path=NULL;

#endif //USE_HMMPARALLEL_STRUCTURES
	}

	if (obs!=NULL)
	{
		int32_t max_T=obs->get_max_vector_length();

		if (lambda)
		{
#ifdef USE_HMMPARALLEL_STRUCTURES
			for (int32_t i=0; i<parallel->get_num_threads(); i++)
			{
				this->alpha_cache[i].table= lambda->alpha_cache[i].table;
				this->beta_cache[i].table=	lambda->beta_cache[i].table;
				this->states_per_observation_psi[i]=lambda->states_per_observation_psi[i] ;
				this->path[i]=lambda->path[i];
			} ;
#else
			this->alpha_cache.table= lambda->alpha_cache.table;
			this->beta_cache.table= lambda->beta_cache.table;
			this->states_per_observation_psi= lambda->states_per_observation_psi;
			this->path=lambda->path;
#endif //USE_HMMPARALLEL_STRUCTURES

			this->reused_caches=true;
		}
		else
		{
			this->reused_caches=false;
#ifdef USE_HMMPARALLEL_STRUCTURES
			SG_INFO("allocating mem for path-table of size %.2f Megabytes (%d*%d) each:\n", ((float32_t)max_T)*N*sizeof(T_STATES)/(1024*1024), max_T, N)
			for (int32_t i=0; i<parallel->get_num_threads(); i++)
			{
				if ((states_per_observation_psi[i]=SG_MALLOC(T_STATES,max_T*N))!=NULL)
					SG_DEBUG("path_table[%i] successfully allocated\n",i)
				else
					SG_ERROR("failed allocating memory for path_table[%i].\n",i)
				path[i]=SG_MALLOC(T_STATES, max_T);
			}
#else // no USE_HMMPARALLEL_STRUCTURES
			SG_INFO("allocating mem of size %.2f Megabytes (%d*%d) for path-table ....", ((float32_t)max_T)*N*sizeof(T_STATES)/(1024*1024), max_T, N)
			if ((states_per_observation_psi=SG_MALLOC(T_STATES,max_T*N)) != NULL)
				SG_DONE()
			else
				SG_ERROR("failed.\n")

			path=SG_MALLOC(T_STATES, max_T);
#endif // USE_HMMPARALLEL_STRUCTURES
#ifdef USE_HMMCACHE
			SG_INFO("allocating mem for caches each of size %.2f Megabytes (%d*%d) ....\n", ((float32_t)max_T)*N*sizeof(T_ALPHA_BETA_TABLE)/(1024*1024), max_T, N)

#ifdef USE_HMMPARALLEL_STRUCTURES
			for (int32_t i=0; i<parallel->get_num_threads(); i++)
			{
				if ((alpha_cache[i].table=SG_MALLOC(T_ALPHA_BETA_TABLE, max_T*N))!=NULL)
					SG_DEBUG("alpha_cache[%i].table successfully allocated\n",i)
				else
					SG_ERROR("allocation of alpha_cache[%i].table failed\n",i)

				if ((beta_cache[i].table=SG_MALLOC(T_ALPHA_BETA_TABLE, max_T*N)) != NULL)
					SG_DEBUG("beta_cache[%i].table successfully allocated\n",i)
				else
					SG_ERROR("allocation of beta_cache[%i].table failed\n",i)
			} ;
#else // USE_HMMPARALLEL_STRUCTURES
			if ((alpha_cache.table=SG_MALLOC(T_ALPHA_BETA_TABLE, max_T*N)) != NULL)
				SG_DEBUG("alpha_cache.table successfully allocated\n")
			else
				SG_ERROR("allocation of alpha_cache.table failed\n")

			if ((beta_cache.table=SG_MALLOC(T_ALPHA_BETA_TABLE, max_T*N)) != NULL)
				SG_DEBUG("beta_cache.table successfully allocated\n")
			else
				SG_ERROR("allocation of beta_cache.table failed\n")

#endif // USE_HMMPARALLEL_STRUCTURES
#else // USE_HMMCACHE
#ifdef USE_HMMPARALLEL_STRUCTURES
			for (int32_t i=0; i<parallel->get_num_threads(); i++)
			{
				alpha_cache[i].table=NULL ;
				beta_cache[i].table=NULL ;
			} ;
#else //USE_HMMPARALLEL_STRUCTURES
			alpha_cache.table=NULL ;
			beta_cache.table=NULL ;
#endif //USE_HMMPARALLEL_STRUCTURES
#endif //USE_HMMCACHE
		}
	}

	//initialize pat/mod_prob as not calculated
	invalidate_model();
}

bool CHMM::permutation_entropy(int32_t window_width, int32_t sequence_number)
{
	if (p_observations && window_width>0 &&
			( sequence_number<0 || sequence_number < p_observations->get_num_vectors()))
	{
		int32_t min_sequence=sequence_number;
		int32_t max_sequence=sequence_number;

		if (sequence_number<0)
		{
			min_sequence=0;
			max_sequence=p_observations->get_num_vectors();
			SG_INFO("numseq: %d\n", max_sequence)
		}

		SG_INFO("min_sequence: %d max_sequence: %d\n", min_sequence, max_sequence)
		for (sequence_number=min_sequence; sequence_number<max_sequence; sequence_number++)
		{
			int32_t sequence_length=0;
			bool free_vec;
			uint16_t* obs=p_observations->get_feature_vector(sequence_number, sequence_length, free_vec);

			int32_t histsize=get_M();
			int64_t* hist=SG_MALLOC(int64_t, histsize);
			int32_t i,j;

			for (i=0; i<sequence_length-window_width; i++)
			{
				for (j=0; j<histsize; j++)
					hist[j]=0;

				uint16_t* ptr=&obs[i];
				for (j=0; j<window_width; j++)
				{
					hist[*ptr++]++;
				}

				float64_t perm_entropy=0;
				for (j=0; j<get_M(); j++)
				{
					float64_t p=
						(((float64_t) hist[j])+PSEUDO)/
						(window_width+get_M()*PSEUDO);
					perm_entropy+=p*log(p);
				}

				SG_PRINT("%f\n", perm_entropy)
			}
			p_observations->free_feature_vector(obs, sequence_number, free_vec);

			SG_FREE(hist);
		}
		return true;
	}
	else
		return false;
}

float64_t CHMM::get_log_derivative(int32_t num_param, int32_t num_example)
{
	if (num_param<N)
		return model_derivative_p(num_param, num_example);
	else if (num_param<2*N)
		return model_derivative_q(num_param-N, num_example);
	else if (num_param<N*(N+2))
	{
		int32_t k=num_param-2*N;
		int32_t i=k/N;
		int32_t j=k%N;
		return model_derivative_a(i,j, num_example);
	}
	else if (num_param<N*(N+2+M))
	{
		int32_t k=num_param-N*(N+2);
		int32_t i=k/M;
		int32_t j=k%M;
		return model_derivative_b(i,j, num_example);
	}

	ASSERT(false)
	return -1;
}

float64_t CHMM::get_log_model_parameter(int32_t num_param)
{
	if (num_param<N)
		return get_p(num_param);
	else if (num_param<2*N)
		return get_q(num_param-N);
	else if (num_param<N*(N+2))
		return transition_matrix_a[num_param-2*N];
	else if (num_param<N*(N+2+M))
		return observation_matrix_b[num_param-N*(N+2)];

	ASSERT(false)
	return -1;
}


//convergence criteria  -tobeadjusted-
bool CHMM::converged(float64_t x, float64_t y)
{
	float64_t diff=y-x;
	float64_t absdiff=fabs(diff);

	SG_INFO("\n #%03d\tbest result so far: %G (eps: %f)", iteration_count, y, diff)

	if (iteration_count--==0 || (absdiff<epsilon && conv_it<=0))
	{
		iteration_count=iterations;
		SG_INFO("...finished\n")
		conv_it=5;
		return true;
	}
	else
	{
		if (absdiff<epsilon)
			conv_it--;
		else
			conv_it=5;

		return false;
	}
}

bool CHMM::baum_welch_viterbi_train(BaumWelchViterbiType type)
{
	CHMM* estimate=new CHMM(this);
	CHMM* working=this;
	float64_t prob_max=-CMath::INFTY;
	float64_t prob=-CMath::INFTY;
	float64_t prob_train=CMath::ALMOST_NEG_INFTY;
	iteration_count=iterations;

	while (!converged(prob, prob_train) && (!CSignal::cancel_computations()))
	{
		CMath::swap(working, estimate);
		prob=prob_train;

		switch (type) {
			case BW_NORMAL:
				working->estimate_model_baum_welch(estimate); break;
			case BW_TRANS:
				working->estimate_model_baum_welch_trans(estimate); break;
			case BW_DEFINED:
				working->estimate_model_baum_welch_defined(estimate); break;
			case VIT_NORMAL:
				working->estimate_model_viterbi(estimate); break;
			case VIT_DEFINED:
				working->estimate_model_viterbi_defined(estimate); break;
		}
		prob_train=estimate->model_probability();

		if (prob_max<prob_train)
			prob_max=prob_train;
	}

	if (estimate == this)
	{
		estimate->copy_model(working);
		delete working;
	}
	else
		delete estimate;

	return true;
}
