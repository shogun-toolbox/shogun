#include <math.h>
#include "distributions/hmm/HMM.h"
#include "lib/Mathmatics.h"
#include "lib/io.h"
#include "features/StringFeatures.h"
#include "features/CharFeatures.h"

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <ctype.h>

#ifdef SUNOS
extern "C" int	finite(double);
#endif

static const INT num_words = 4096 ;
static const INT word_degree = 6 ;
static const INT num_svms = 4 ;
static bool word_used[num_words] ;
static REAL svm_value_unnormalized[num_svms] ;
static REAL *dict_weights ;
static INT svm_pos_start ;
static INT num_unique_words = 0 ;

inline void translate_from_single_order(WORD* obs, INT sequence_length, 
										INT start=5, INT order=word_degree, 
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
		assert(value<num_words) ;
	}
	if (start>0)
		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
}

inline void reset_svm_value(INT pos, INT & last_svm_pos, REAL * svm_value) 
{
	for (int i=0; i<num_words; i++)
		word_used[i]=false ;
	for (INT s=0; s<num_svms; s++)
		svm_value_unnormalized[s] = 0 ;
	for (INT s=0; s<num_svms; s++)
		svm_value[s] = 0 ;
	last_svm_pos = pos - 6+1 ;
	svm_pos_start = pos - 6 ;
	num_unique_words=0 ;
}

void extend_svm_value(WORD* wordstr, INT pos, INT &last_svm_pos, REAL* svm_value) 
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

inline bool extend_orf(const bool* genestr_stop, INT orf_from, INT orf_to, INT start, INT &last_pos, INT to)
{
	if (start<0) 
		start=0 ;
	if (to<0)
		to=0 ;
	
	INT orf_target = orf_to-orf_from ;
	if (orf_target<0) orf_target+=3 ;
	
	INT pos ;
	if (last_pos==to)
		pos = to-orf_to-3 ;
	else
		pos=last_pos ;

	if (pos<0)
		return true ;
	
	for (; pos>=start; pos-=3)
		if (genestr_stop[pos])
			return false ;
	
	last_pos = CMath::min(pos+3,to-orf_to-3) ;

	return true ;
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
#define ORF_FROM(i) orf_info[i] 
#define ORF_TO(i) orf_info[N+i] 

void CHMM::best_path_trans(const REAL *seq, INT seq_len, const INT *pos, const INT *orf_info,
						   struct penalty_struct **PEN_matrix, 
						   const char *genestr, INT genestr_len,
						   short int nbest, 
						   REAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq,
						   REAL *dictionary_weights, INT dict_len, 
						   REAL *&PEN_values, INT &num_PEN_id, bool use_orf)
{
	const INT default_look_back = 30000 ;
	INT max_look_back = 0 ;
	bool use_svm = false ;
	assert(dict_len==num_svms*num_words) ;
	dict_weights=dictionary_weights ;

	REAL svm_value[num_svms] ;
	
	{ // initialize svm_svalue
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
	{ // determine maximal length of look-back
		for (INT i=0; i<N; i++)
			for (INT j=0; j<N; j++)
				if (PEN(i,j)!=NULL)
				{
					if (PEN(i,j)->max_len>max_look_back)
						max_look_back=PEN(i,j)->max_len ;
					if (PEN(i,j)->use_svm)
						use_svm=true ;
					if (PEN(i,j)->next_pen)
						if (PEN(i,j)->next_pen->use_svm)
						use_svm=true ;
					if (PEN(i,j)->id+1>num_PEN_id)
						num_PEN_id=PEN(i,j)->id+1 ;
				} else
					if (max_look_back<default_look_back)
						max_look_back=default_look_back ;
	}
	max_look_back = CMath::min(genestr_len, max_look_back) ;
	//fprintf(stderr,"use_svm=%i\n", use_svm) ;
	
	const INT look_back_buflen = max_look_back*nbest*N ;
	const REAL mem_use = (REAL)(seq_len*N*nbest*(sizeof(T_STATES)+sizeof(short int)+sizeof(INT))+
								look_back_buflen*(2*sizeof(REAL)+sizeof(INT))+
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
	assert(nbest<32000) ;
		
	bool * genestr_stop = new bool[genestr_len] ;

	REAL* delta= new REAL[look_back_buflen] ;
	assert(delta!=NULL) ;
	T_STATES *psi=new T_STATES[seq_len*N*nbest] ;
	assert(psi!=NULL) ;
	short int *ktable=new short int[seq_len*N*nbest] ;
	assert(ktable!=NULL) ;
	INT *ptable=new INT[seq_len*N*nbest] ;
	assert(ptable!=NULL) ;

	REAL* delta_end= new REAL[nbest] ;
	assert(delta_end!=NULL) ;
	T_STATES* path_end = new T_STATES[nbest] ;
	assert(path_end!=NULL) ;
	short int *ktable_end=new short int[nbest] ;
	assert(ktable_end!=NULL) ;

	REAL* tempvv=new REAL[look_back_buflen] ;
	assert(tempvv!=NULL) ;
	INT* tempii=new INT[look_back_buflen] ;
	assert(tempii!=NULL) ;

	T_STATES* state_seq = new T_STATES[seq_len] ;
	assert(state_seq!=NULL) ;
	INT * pos_seq   = new INT[seq_len] ;
	assert(pos_seq!=NULL) ;

	{ // precompute stop codons
		for (INT i=0; i<genestr_len-2; i++)
			if (genestr[i]=='t' && 
				((genestr[i+1]=='a' && 
				  (genestr[i+2]=='a' || genestr[i+2]=='g')) ||
				 (genestr[i+1]=='g' && genestr[i+2]=='a')))
				genestr_stop[i]=true ;
			else
				genestr_stop[i]=false ;
		genestr_stop[genestr_len-1]=false ;
		genestr_stop[genestr_len-1]=false ;
	}

	// translate to words, if svm is used
	WORD* wordstr=NULL ;
	if (use_svm)
	{
		assert(dict_weights!=NULL) ;
		wordstr=new WORD[genestr_len] ;
		for (INT i=0; i<genestr_len; i++)
			switch (genestr[i])
			{
			case 'a': wordstr[i]=0 ; break ;
			case 'c': wordstr[i]=1 ; break ;
			case 'g': wordstr[i]=2 ; break ;
			case 't': wordstr[i]=3 ; break ;
			default: assert(0) ;
			}
		translate_from_single_order(wordstr, genestr_len) ;
		//fprintf(stderr, "genestr_len=%i\n", genestr_len) ;
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
				const REAL *elem_val      = trans_list_forward_val[j] ;
				
				INT list_len=0 ;
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;

					const struct penalty_struct * penalty = PEN(j,ii) ;
					INT look_back = default_look_back ;
					if (penalty!=NULL)
						look_back=penalty->max_len ;
					INT orf_from = ORF_FROM(ii) ;
					INT orf_to   = ORF_TO(j) ;
					if((orf_from!=-1)!=(orf_to!=-1))
						fprintf(stderr,"j=%i  ii=%i  orf_from=%i orf_to=%i p=%1.2f\n", j, ii, orf_from, orf_to, elem_val[i]) ;
					assert((orf_from!=-1)==(orf_to!=-1)) ;
					
					INT orf_target = -1 ;
					if (orf_from!=-1)
					{
						orf_target=orf_to-orf_from ;
						if (orf_target<0) orf_target+=3 ;
						assert(orf_target>=0 && orf_target<3) ;
					}

					INT last_pos = pos[t] ;
					INT last_svm_pos ;
					if (use_svm)
						reset_svm_value(pos[t], last_svm_pos, svm_value) ;

					for (INT ts=t-1; ts>=0 && pos[t]-pos[ts]<=look_back; ts--)
					{
						bool ok ;
						if (orf_target==-1)
							ok=true ;
						else if (pos[ts]!=-1 && (pos[t]-pos[ts])%3==orf_target)
						{
								
							ok=(!use_orf) || extend_orf(genestr_stop, orf_from, orf_to, pos[ts], last_pos, pos[t]) ;
							if (!ok) 
							{
								//CIO::message(M_DEBUG, "no orf from %i[%i] to %i[%i]\n", pos[ts], orf_from, pos[t], orf_to) ;
								break ;
							}
						} else
							ok=false ;
						
						if (ok)
						{
							if (use_svm)
								extend_svm_value(wordstr, pos[ts], last_svm_pos, svm_value) ;
							
							REAL pen_val = lookup_penalty(penalty, pos[t]-pos[ts], svm_value, true) ;
							for (short int diff=0; diff<nbest; diff++)
							{
								REAL  val        = DELTA(ts,ii,diff) + elem_val[i] ;
								val             += pen_val ;
								
								tempvv[list_len] = -val ;
								tempii[list_len] =  ii + diff*N + ts*N*nbest;
								list_len++ ;
							} ;
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
				my_state_seq[i+k*seq_len] = state_seq[num_states-i-1] ;
				my_pos_seq[i+k*seq_len]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states+k*seq_len]=-1 ;
			my_pos_seq[num_states+k*seq_len]=-1 ;
		}
		REAL svm_value[num_svms] ;
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
		PEN_values = new REAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_values[s]=0 ;
		char * PEN_names[num_PEN_id] ;
		for (INT s=0; s<num_PEN_id; s++)
			PEN_names[s]=NULL ;

		for (short int k=0; k<nbest; k++)
		{
			for (INT i=0; i<seq_len-1; i++)
			{
				if (my_state_seq[i+1+k*seq_len]==-1)
					break ;
				INT from_state = my_state_seq[i+k*seq_len] ;
				INT to_state   = my_state_seq[i+1+k*seq_len] ;
				INT from_pos   = my_pos_seq[i+k*seq_len] ;
				INT to_pos     = my_pos_seq[i+1+k*seq_len] ;
				
				//CIO::message(M_DEBUG, "%i. from state %i pos %i[%i]  to  state %i pos %i[%i]  penalties:", k, from_state, pos[from_pos], from_pos, to_state, pos[to_pos], to_pos) ;
				INT last_svm_pos = -1 ;
				
				reset_svm_value(pos[to_pos], last_svm_pos, svm_value) ;
				extend_svm_value(wordstr, pos[from_pos], last_svm_pos, svm_value) ;
				struct penalty_struct *penalty = PEN(to_state, from_state) ;
				while (penalty)
				{
					REAL pen_val = lookup_penalty(penalty, pos[to_pos]-pos[from_pos], svm_value, false) ;
					PEN_values[penalty->id + i*num_PEN_id + seq_len*num_PEN_id*k] += pen_val ;
					PEN_names[penalty->id] = penalty->name ;
					//CIO::message(M_DEBUG, "%s(%i;%1.2f), ", penalty->name, penalty->id, pen_val) ;
					penalty = penalty->next_pen ;
				}
				//CIO::message(M_DEBUG, "\n") ;
			}
			/*for (INT s=0; s<num_PEN_id; s++)
			{
				if (PEN_names[s])
					CIO::message(M_DEBUG, "%s:\t%1.2f\n", PEN_names[s], PEN_values[s+num_PEN_id*k]) ;
				else
					assert(PEN_values[s]==0.0) ;
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
	delete[] genestr_stop ;
}
