
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

#define ARRAY_SIZE 65336

#ifdef SUNOS
extern "C" int	finite(double);
#endif


void CHMM::best_path_trans(REAL *seq, INT seq_len, INT *pos,
						   struct penalty_struct **PEN_matrix,
						   short int nbest, 
						   REAL *prob_nbest, INT *my_state_seq, INT *my_pos_seq)
{
#define PSI(t,j,k) psi[nbest*((t)*N+(j))+(k)]	
#define DELTA(t,j,k) delta[(t)*nbest*N+(j)*nbest+k]
#define DELTA_END(k) delta_end[k]
#define KTAB(t,j,k) ktable[nbest*((t)*N+j)+k]
#define PTAB(t,j,k) ptable[nbest*((t)*N+j)+k]
#define KTAB_ENDS(k) ktable_ends[k]
#define PATH_ENDS(k) path_ends[k]
#define SEQ(j,t) seq[j+(t)*N]
#define STEP_PEN(i,j) step_penalties[(j)*N+i]
#define PEN(i,j) PEN_matrix[(j)*N+i]

	const INT max_look_back = 1000 ;
	
	//fprintf(stderr,"seq_len=%i N=%i, nbest=%i\n", seq_len, N,nbest) ;
	
	T_STATES *psi=new T_STATES[seq_len*N*nbest] ;
	assert(psi!=NULL) ;
	short int *ktable=new short int[seq_len*N*nbest] ;
	assert(ktable!=NULL) ;
	INT *ptable=new INT[seq_len*N*nbest] ;
	assert(ptable!=NULL) ;
	short int *ktable_ends=new short int[nbest] ;
	assert(ktable_ends!=NULL) ;

	const INT look_back_buflen = math.min(seq_len, seq_len)*nbest*N ;
	REAL* tempvv=new REAL[look_back_buflen] ;
	assert(tempvv!=NULL) ;
	INT* tempii=new INT[look_back_buflen] ;
	assert(tempii!=NULL) ;

	T_STATES* path_ends = new T_STATES[nbest] ;
	assert(path_ends!=NULL) ;
	REAL* delta= new REAL[N*nbest] ;
	assert(delta!=NULL) ;
	REAL* delta_end= new REAL[nbest] ;
	assert(delta_end!=NULL) ;

	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			DELTA(0,i,0) = get_p(i) + SEQ(i,0) ;
			for (short int k=1; k<nbest; k++)
			{
				DELTA(0,i,k)=-math.INFTY ;
				KTAB(0,i,k)=0 ;
			}
		}
	}
	
	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		fprintf(stderr, "t=%i  ", t) ;
		for (INT i=0; i<N; i++)
			fprintf(stderr,"%i: %1.2f  ", i, DELTA(t-1,i,0)) ;
		fprintf(stderr, "\n") ;
		
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
					for (INT ts=t-1; ts>=0 && pos[t]-pos[ts]<=max_look_back; ts--)
					{
						for (short int diff=0; diff<nbest; diff++)
						{
							REAL  val        = DELTA(ts,ii,diff) + elem_val[i] ;
							if (finite(val)>=0)
								val          += lookup_penalty(PEN(j,ii), pos[t]-pos[ts]) ;
							tempvv[list_len] = -val ;
							tempii[list_len] =  ii + diff*N + ts*N*nbest;
							list_len++ ;
							assert(list_len<look_back_buflen) ;
						} ;
					}
				}
				math.qsort(tempvv, tempii, list_len) ;
				
				for (short int k=0; k<nbest; k++)
				{
					if (k<list_len)
					{
						DELTA(t,j,k)    = -tempvv[k] + SEQ(j,t);
						PSI(t,j,k)      = (tempii[k]%N) ;
						KTAB(t,j,k)     = (tempii[k]%(N*nbest)-PSI(t,j,k))/N ;
						PTAB(t,j,k)     = (tempii[k]-(tempii[k]%(N*nbest)))/(N*nbest) ;
						
						assert(KTAB(t,j,k)<nbest) ;
						assert(PSI(t,j,k)<N) ;
						assert(PTAB(t,j,k)<seq_len) ;
						assert(finite(DELTA(t,j,k))>=0) ;
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
		math.qsort(tempvv, tempii, list_len) ;
		
		for (short int k=0; k<nbest; k++)
		{
			DELTA_END(k) = -tempvv[k] ;
			PATH_ENDS(k) = (tempii[k]%N) ;
			KTAB_ENDS(k) = (tempii[k]-PATH_ENDS(k))/N ;
		}
	}
	
	{ //state sequence backtracking
		INT * state_seq = new INT[seq_len] ;
		INT * pos_seq   = new INT[seq_len] ;
		
		for (short int k=0; k<nbest; k++)
		{
			prob_nbest[k]=-DELTA_END(k) ;
			
			INT i         = 0 ;
			state_seq[i]  = PATH_ENDS(k) ;
			short int q   = KTAB_ENDS(k) ;
			pos_seq[i]    = seq_len-1 ;
			while (state_seq[i]>0)
			{
				state_seq[i+1] = PSI(pos_seq[i], state_seq[i], q);
				pos_seq[i+1]   = PTAB(pos_seq[i], state_seq[i], q) ;
				q              = KTAB(pos_seq[i], state_seq[i], q) ;
				i++ ;
			}
			INT num_states = i+1 ;
			for (i=0; i<num_states;i++)
			{
				my_state_seq[i] = state_seq[num_states-i-1] ;
				my_pos_seq[i]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states]=-1 ;
			my_pos_seq[num_states]=-1 ;
		}
		delete[] state_seq ;
		delete[] pos_seq ;
	}

	delete[] psi ;
	delete[] ktable;
	delete[] ktable_ends;
	delete[] ptable;

	delete[] tempvv ;
	delete[] tempii ;

	delete[] path_ends ;
	delete[] delta ;
	delete[] delta_end ;
}
