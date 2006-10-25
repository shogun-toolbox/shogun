/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Gunnar Raetsch
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <math.h>
#include "distributions/hmm/HMM.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "lib/Plif.h"
#include "features/StringFeatures.h"
#include "features/CharFeatures.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>

#ifdef SUNOS
extern "C" int	finite(double);
#endif

static const INT num_words = 4 ;
static const INT word_degree = 1 ;
static const INT num_svms = 1 ;
static bool word_used[num_words] ;
static DREAL svm_value_unnormalized[num_svms] ;
static DREAL *dict_weights ;
static INT num_unique_words = 0 ;


static void translate_from_single_order(WORD* obs, INT sequence_length, 
										INT start, INT order, 
										INT max_val=2/*DNA->2bits*/)
{
	INT i,j;
	WORD value=0;
	
	for (i=sequence_length-1; i>= ((int) order)-1; i--)	//convert interval of size T
	{
		value=0;
		for (j=i; j>=i-((int) order)+1; j--)
			value= (value >> max_val) | (obs[j] << (max_val * (order-1)));
		
		obs[i]= (WORD) value;
	}
	
	for (i=order-2;i>=0;i--)
	{
		value=0;
		for (j=i; j>=i-order+1; j--)
		{
			value= (value >> max_val);
			if (j>=0)
				value|=obs[j] << (max_val * (order-1));
		}
		obs[i]=value;
		ASSERT(value<num_words) ;
	}
	if (start>0)
		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
}

static void reset_svm_value(INT pos, INT & last_svm_pos, DREAL * svm_value) 
{
	for (int i=0; i<num_words; i++)
		word_used[i]=false ;
	for (INT s=0; s<num_svms; s++)
		svm_value_unnormalized[s] = 0 ;
	for (INT s=0; s<num_svms; s++)
		svm_value[s] = 0 ;
	last_svm_pos = pos - 6+1 ;
	num_unique_words=0 ;
}

static void extend_svm_value(WORD* wordstr, INT pos, INT &last_svm_pos, DREAL* svm_value) 
{
	bool did_something = false ;
	for (int i=last_svm_pos-1; (i>=pos) && (i>=0); i--)
	{
		if (wordstr[i]>=num_words)
			CIO::message(M_DEBUG, "wordstr[%i]=%i\n", i, wordstr[i]) ;
		
		if (!word_used[wordstr[i]])
		{
			for (INT s=0; s<num_svms; s++)
				svm_value_unnormalized[s]+=dict_weights[wordstr[i]+s*num_words] ;

			word_used[wordstr[i]]=true ;
			num_unique_words++ ;
			did_something=true ;
		}
	} ;
	if (num_unique_words>0)
	{
		last_svm_pos=pos ;
		if (did_something)
			for (INT s=0; s<num_svms; s++)
				svm_value[s]= svm_value_unnormalized[s]/sqrt((double)num_unique_words) ;  // full normalization
	}
	else
	{
		// what should I do?
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
}


static void reset_segment_sum_value(INT num_states, INT pos, INT & last_segment_sum_pos, DREAL * segment_sum_value) 
{
	for (INT s=0; s<num_states; s++)
		segment_sum_value[s] = 0 ;
	last_segment_sum_pos = pos ;
	//fprintf(stderr, "start: %i\n", pos) ;
}

static void extend_segment_sum_value(DREAL *segment_sum_weights, INT seqlen, INT num_states,
							  INT pos, INT &last_segment_sum_pos, DREAL* segment_sum_value) 
{
	for (int i=last_segment_sum_pos-1; (i>=pos) && (i>=0); i--)
	{
		for (INT s=0; s<num_states; s++)
			segment_sum_value[s] += segment_sum_weights[i*num_states+s] ;
	} ;
	//fprintf(stderr, "extend %i: %f\n", pos, segment_sum_value[0]) ;
	last_segment_sum_pos = pos ;
}


#define PSI(t,j,k) psi[nbest*((t)*N+(j))+(k)]	
#define DELTA(t,j,k) delta[(j)*nbest*max_look_back+((t)%max_look_back)*nbest+k]
#define KTAB(t,j,k) ktable[nbest*((t)*N+j)+k]
#define PTAB(t,j,k) ptable[nbest*((t)*N+j)+k]
#define DELTA_END(k) delta_end[k]
#define KTAB_END(k) ktable_end[k]
#define PATH_END(k) path_end[k]
#define SEQ(j,t) seq[j+(t)*N]
#define PEN(i,j) PEN_matrix[(j)*N+i]

void CHMM::best_path_2struct(const DREAL *seq, INT seq_len, const INT *pos,
							 CPlif **PEN_matrix, 
							 const char *genestr, INT genestr_len,
							 short int nbest, 
							 DREAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
							 DREAL *dictionary_weights, INT dict_len, DREAL *segment_sum_weights, 
							 DREAL *&PEN_values, DREAL *&PEN_input_values, INT &num_PEN_id)
{
	const INT default_look_back = 100 ;
	INT max_look_back = default_look_back ;
	bool use_svm = false ;
	ASSERT(dict_len==num_svms*num_words) ;
	dict_weights=dictionary_weights ;

	DREAL svm_value[num_svms] ;
	DREAL segment_sum_value[N] ;
	
	{ // initialize svm_svalue
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
	{ // determine maximal length of look-back
		for (INT i=0; i<N; i++)
			for (INT j=0; j<N; j++)
			{
				CPlif *penij=PEN(i,j) ;
				while (penij!=NULL)
				{
					if (penij->get_max_len()>max_look_back)
						max_look_back=penij->get_max_len() ;
					if (penij->get_use_svm())
						use_svm=true ;
					if (penij->get_id()+1>num_PEN_id)
						num_PEN_id=penij->get_id()+1 ;
					penij=penij->get_next_pen() ;
				} 
			}
	}
	max_look_back = CMath::min(genestr_len, max_look_back) ;
	//fprintf(stderr,"use_svm=%i\n", use_svm) ;
	fprintf(stderr,"max_look_back=%i\n", max_look_back) ;
	
	const INT look_back_buflen = (max_look_back+1)*nbest*N ;
	//fprintf(stderr,"look_back_buflen=%i\n", look_back_buflen) ;
	const DREAL mem_use = (DREAL)(seq_len*N*nbest*(sizeof(T_STATES)+sizeof(short int)+sizeof(INT))+
								look_back_buflen*(2*sizeof(DREAL)+sizeof(INT))+
								seq_len*(sizeof(T_STATES)+sizeof(INT))+
								genestr_len*sizeof(bool))/(1024*1024)
		 ;
    bool is_big = (mem_use>200) || (seq_len>5000) ;

	if (is_big)
	{
		CIO::message(M_DEBUG,"calling best_path_trans: seq_len=%i, N=%i, lookback=%i nbest=%i\n", 
					 seq_len, N, max_look_back, nbest) ;
		CIO::message(M_DEBUG,"allocating %1.2fMB of memory\n", 
					 mem_use) ;
	}
	ASSERT(nbest<32000) ;
		
	DREAL* delta= new DREAL[look_back_buflen] ;
	ASSERT(delta!=NULL) ;
	T_STATES *psi=new T_STATES[seq_len*N*nbest] ;
	ASSERT(psi!=NULL) ;
	short int *ktable=new short int[seq_len*N*nbest] ;
	ASSERT(ktable!=NULL) ;
	INT *ptable=new INT[seq_len*N*nbest] ;
	ASSERT(ptable!=NULL) ;

	DREAL* delta_end= new DREAL[nbest] ;
	ASSERT(delta_end!=NULL) ;
	T_STATES* path_end = new T_STATES[nbest] ;
	ASSERT(path_end!=NULL) ;
	short int *ktable_end=new short int[nbest] ;
	ASSERT(ktable_end!=NULL) ;

	DREAL* tempvv=new DREAL[look_back_buflen] ;
	ASSERT(tempvv!=NULL) ;
	INT* tempii=new INT[look_back_buflen] ;
	ASSERT(tempii!=NULL) ;

	T_STATES* state_seq = new T_STATES[seq_len] ;
	ASSERT(state_seq!=NULL) ;
	INT * pos_seq   = new INT[seq_len] ;
	ASSERT(pos_seq!=NULL) ;

	// translate to words, if svm is used
	WORD* wordstr=NULL ;
	if (use_svm)
	{
		ASSERT(dict_weights!=NULL) ;
		wordstr=new WORD[genestr_len] ;
		for (INT i=0; i<genestr_len; i++)
			switch (genestr[i])
			{
			case 'a': wordstr[i]=0 ; break ;
			case 'c': wordstr[i]=1 ; break ;
			case 'g': wordstr[i]=2 ; break ;
			case 't': wordstr[i]=3 ; break ;
			default: ASSERT(0) ;
			}
		translate_from_single_order(wordstr, genestr_len, word_degree-1, word_degree) ;
	}
	
	
	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			DELTA(0,i,0) = get_p(i) + SEQ(i,0) ;
			PSI(0,i,0)   = 0 ;
			KTAB(0,i,0)  = 0 ;
			PTAB(0,i,0)  = 0 ;
			for (short int k=1; k<nbest; k++)
			{
				DELTA(0,i,k)    = -CMath::INFTY ;
				PSI(0,i,0)      = 0 ;
				KTAB(0,i,k)     = 0 ;
				PTAB(0,i,k)     = 0 ;
			}
		}
	}

	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		if (is_big && t%(seq_len/1000)==1)
			CIO::progress(t, 0, seq_len);
		
		for (T_STATES j=0; j<N; j++)
		{
			if (SEQ(j,t)<-1e20)
			{ // if we cannot observe the symbol here, then we can omit the rest
				for (short int k=0; k<nbest; k++)
				{
					DELTA(t,j,k)    = SEQ(j,t) ;
					PSI(t,j,k)      = 0 ;
					KTAB(t,j,k)     = 0 ;
					PTAB(t,j,k)     = 0 ;
				}
			}
			else
			{
				const T_STATES num_elem   = trans_list_forward_cnt[j] ;
				const T_STATES *elem_list = trans_list_forward[j] ;
				const DREAL *elem_val      = trans_list_forward_val[j] ;
				
				INT list_len=0 ;
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					//fprintf(stderr, "i=%i  ii=%i  num_elem=%i  PEN=%ld\n", i, ii, num_elem, PEN(j,ii)) ;
					
					const CPlif * penalty = PEN(j,ii) ;
					INT look_back = default_look_back ;
					if (penalty!=NULL)
						look_back=penalty->get_max_len() ;
					
					INT last_svm_pos ;
					if (use_svm)
						reset_svm_value(pos[t], last_svm_pos, svm_value) ;

					INT last_segment_sum_pos ;
					reset_segment_sum_value(N, pos[t], last_segment_sum_pos, segment_sum_value) ;

					for (INT ts=t-1; ts>=0 && pos[t]-pos[ts]<=look_back; ts--)
					{
						if (use_svm)
							extend_svm_value(wordstr, pos[ts], last_svm_pos, svm_value) ;

						extend_segment_sum_value(segment_sum_weights, seq_len, N, pos[ts], last_segment_sum_pos, segment_sum_value) ;
						
						DREAL input_value ;
						DREAL pen_val = penalty->lookup_penalty(pos[t]-pos[ts], svm_value, true, input_value) + segment_sum_value[j] ;
						for (short int diff=0; diff<nbest; diff++)
						{
							DREAL  val        = DELTA(ts,ii,diff) + elem_val[i] ;
							val             += pen_val ;
							
							tempvv[list_len] = -val ;
							tempii[list_len] =  ii + diff*N + ts*N*nbest;
							//fprintf(stderr, "%i (%i,%i,%i, %i, %i) ", list_len, diff, ts, i, pos[t]-pos[ts], look_back) ;
							list_len++ ;
						}
					}
				}
				CMath::nmin<INT>(tempvv, tempii, list_len, nbest) ;
				
				for (short int k=0; k<nbest; k++)
				{
					if (k<list_len)
					{
						DELTA(t,j,k)    = -tempvv[k] + SEQ(j,t);
						PSI(t,j,k)      = (tempii[k]%N) ;
						KTAB(t,j,k)     = (tempii[k]%(N*nbest)-PSI(t,j,k))/N ;
						PTAB(t,j,k)     = (tempii[k]-(tempii[k]%(N*nbest)))/(N*nbest) ;
					}
					else
					{
						DELTA(t,j,k)    = -CMath::INFTY ;
						PSI(t,j,k)      = 0 ;
						KTAB(t,j,k)     = 0 ;
						PTAB(t,j,k)     = 0 ;
					}
				}
			}
		}
	}
	
	{ //termination
		INT list_len = 0 ;
		for (short int diff=0; diff<nbest; diff++)
		{
			for (T_STATES i=0; i<N; i++)
			{
				tempvv[list_len] = -(DELTA(seq_len-1,i,diff)+get_q(i)) ;
				tempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(tempvv, tempii, list_len, nbest) ;
		
		for (short int k=0; k<nbest; k++)
		{
			DELTA_END(k) = -tempvv[k] ;
			PATH_END(k) = (tempii[k]%N) ;
			KTAB_END(k) = (tempii[k]-PATH_END(k))/N ;
		}
	}
	
	{ //state sequence backtracking		
		for (short int k=0; k<nbest; k++)
		{
			prob_nbest[k]= DELTA_END(k) ;
			
			INT i         = 0 ;
			state_seq[i]  = PATH_END(k) ;
			short int q   = KTAB_END(k) ;
			pos_seq[i]    = seq_len-1 ;

			while (pos_seq[i]>0)
			{
				//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
				state_seq[i+1] = PSI(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = PTAB(pos_seq[i], state_seq[i], q) ;
				q              = KTAB(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			//fprintf(stderr,"s=%i p=%i q=%i\n", state_seq[i], pos_seq[i], q) ;
			INT num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i+k*(seq_len+1)] = state_seq[num_states-i-1] ;
				my_pos_seq[i+k*(seq_len+1)]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states+k*(seq_len+1)]=-1 ;
			my_pos_seq[num_states+k*(seq_len+1)]=-1 ;
		}
		DREAL svm_value[num_svms] ;
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
		// one more for the emissions: the first
		num_PEN_id++ ;
		PEN_values = new DREAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_values[s]=0 ;
		PEN_input_values = new DREAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_input_values[s]=0 ;
		char * PEN_names[num_PEN_id] ;
		for (INT s=0; s<num_PEN_id; s++)
			PEN_names[s]=NULL ;

		for (short int k=0; k<nbest; k++)
		{
			for (INT i=0; i<seq_len-1; i++)
			{
				if (my_state_seq[i+1+k*(seq_len+1)]==-1)
					break ;
				INT from_state = my_state_seq[i+k*(seq_len+1)] ;
				INT to_state   = my_state_seq[i+1+k*(seq_len+1)] ;
				INT from_pos   = my_pos_seq[i+k*(seq_len+1)] ;
				INT to_pos     = my_pos_seq[i+1+k*(seq_len+1)] ;
				
				//CIO::message(M_DEBUG, "%i. from state %i pos %i[%i]  to  state %i pos %i[%i]  penalties:", k, from_state, pos[from_pos], from_pos, to_state, pos[to_pos], to_pos) ;
				INT last_svm_pos = -1 ;
				INT last_segment_sum_pos=-1 ;
								
				if (use_svm)
				{
					reset_svm_value(pos[to_pos], last_svm_pos, svm_value) ;
					extend_svm_value(wordstr, pos[from_pos], last_svm_pos, svm_value) ;
				}
				reset_segment_sum_value(N, pos[to_pos], last_segment_sum_pos, segment_sum_value) ;
				extend_segment_sum_value(segment_sum_weights, seq_len, N, pos[from_pos], last_segment_sum_pos, segment_sum_value) ;
				//fprintf(stderr, "%i -> %i  %f %f\n", pos[from_pos], pos[to_pos], segment_sum_value[0], segment_sum_value[1]) ;
				
				PEN_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] = SEQ(to_state, to_pos) + segment_sum_value[to_state] ;
				//PEN_input_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] = segment_sum_value[to_state] ;

				CPlif *penalty = PEN(to_state, from_state) ;
				while (penalty)
				{
					DREAL input_value=0 ;
					DREAL pen_val = penalty->lookup_penalty(pos[to_pos]-pos[from_pos], svm_value, false, input_value) ;
					PEN_values[penalty->get_id() + i*num_PEN_id + seq_len*num_PEN_id*k] += pen_val ;
					PEN_input_values[penalty->get_id() + i*num_PEN_id + seq_len*num_PEN_id*k] += input_value ;
					PEN_names[penalty->get_id()] = penalty->get_name() ;
					//CIO::message(M_DEBUG, "%s(%i;%1.2f), ", penalty->name, penalty->id, pen_val) ;
					penalty = penalty->get_next_pen() ;
				}
				//CIO::message(M_DEBUG, "\n") ;
			}
			/*for (INT s=0; s<num_PEN_id; s++)
			{
				if (PEN_names[s])
					CIO::message(M_DEBUG, "%s:\t%1.2f\n", PEN_names[s], PEN_values[s+num_PEN_id*k]) ;
				else
					ASSERT(PEN_values[s]==0.0) ;
					}*/
		}
	}
	if (is_big)
		CIO::message(M_MESSAGEONLY, "DONE.     \n") ;

	delete[] delta ;
	delete[] psi ;
	delete[] ktable;
	delete[] ptable;

	delete[] ktable_end;
	delete[] path_end ;
	delete[] delta_end ;

	delete[] tempvv ;
	delete[] tempii ;

	delete[] state_seq ;
	delete[] pos_seq ;
}
