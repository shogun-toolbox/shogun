
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
	
	last_pos = math.min(pos+3,to-orf_to-3) ;

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
#define STEP_PEN(i,j) step_penalties[(j)*N+i]
#define PEN(i,j) PEN_matrix[(j)*N+i]
#define ORF_FROM(i) orf_info[i] 
#define ORF_TO(i) orf_info[N+i] 

void CHMM::best_path_trans(const REAL *seq, INT seq_len, const INT *pos, const INT *orf_info,
						   struct penalty_struct **PEN_matrix, 
						   const char *genestr, INT genestr_len,
						   short int nbest, 
						   REAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq)
{
	const INT default_look_back = 10000 ;
	INT max_look_back = 0 ;
	{ // determine maximal length of look-back
		for (INT i=0; i<N; i++)
			for (INT j=0; j<N; j++)
				if (PEN(i,j)!=NULL)
				{
					if (PEN(i,j)->max_len>max_look_back)
						max_look_back=PEN(i,j)->max_len ;
				} else
					if (max_look_back<default_look_back)
						max_look_back=default_look_back ;
	}
	max_look_back = math.min(seq_len, max_look_back) ;

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
	
	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			DELTA(0,i,0) = get_p(i) + SEQ(i,0) ;
			PSI(0,i,0)   = 0 ;
			KTAB(0,i,0)  = 0 ;
			PTAB(0,i,0)  = 0 ;
			for (short int k=1; k<nbest; k++)
			{
				DELTA(0,i,k)    = -math.INFTY ;
				PSI(0,i,0)      = 0 ;
				KTAB(0,i,k)     = 0 ;
				PTAB(0,i,k)     = 0 ;
			}
		}
	}
	
	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		//fprintf(stderr, "t=%i  ", t) ;
		//for (INT i=0; i<N; i++)
		//	fprintf(stderr,"%i: %1.2f  ", i, DELTA(t-1,i,0)) ;
		//fprintf(stderr, "\n") ;
		if (is_big && t%(seq_len/1000)==1)
			CIO::message(M_PROGRESS, "%2.1f%%   \r", 100.0*t/seq_len) ;
		
		for (T_STATES j=0; j<N; j++)
		{
			if (finite(SEQ(j,t))==-1)
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
					for (INT ts=t-1; ts>=0 && pos[t]-pos[ts]<=look_back; ts--)
					{
						bool ok ;
						if (orf_target==-1)
							ok=true ;
						else if (pos[ts]!=-1 && (pos[t]-pos[ts])%3==orf_target)
						{
							ok=extend_orf(genestr_stop, orf_from, orf_to, pos[ts], last_pos, pos[t]) ;
							if (!ok) 
								break ;
						} else
							ok=false ;
						
						if (ok)
						{
							for (short int diff=0; diff<nbest; diff++)
							{
								REAL  val        = DELTA(ts,ii,diff) + elem_val[i] ;
								if (finite(val)>=0)
									val          += lookup_penalty(penalty, pos[t]-pos[ts]) ;
								tempvv[list_len] = -val ;
								tempii[list_len] =  ii + diff*N + ts*N*nbest;
								list_len++ ;
							} ;
						}
					}
				}
				math.nmin(tempvv, tempii, list_len, nbest) ;
				
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
						DELTA(t,j,k)    = -math.INFTY ;
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
		
		math.nmin(tempvv, tempii, list_len, nbest) ;
		
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
	}
	if (is_big)
		CIO::message(M_PROGRESS, "DONE.     \n") ;

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
