
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
						   INT *step_penalties,
						   struct penalty_struct **PEN_matrix,
						   short int nbest, 
						   REAL *prob_nbest, INT *my_paths)
{
#define PSI(t,j,k) psi[nbest*((t)*N+(j))+(k)]	
#define DELTA(j,k) delta[(j)*nbest+k]
#define DELTA_NEW(j,k) delta_new[(j)*nbest+k]
#define DELTA_END(t,k) delta_end[(t)*nbest+k]
#define KTAB(t,j,k) ktable[nbest*((t)*N+j)+k]
#define KTAB_ENDS(t,k) ktable_ends[nbest*(t)+k]
#define PATHS(t,k) my_paths[(k)*(seq_len+1)+t] 
#define PATH_ENDS(t,k) path_ends[(t)*nbest+k]
#define SEQ(j,t) seq[j+(t)*N]
#define STEP_PEN(i,j) step_penalties[(j)*N+i]
#define PEN(i,j) PEN_matrix[(j)*N+i]
#define LEN(i,j) length[(j)*N+(i)]
#define LEN_NEW(i,j) length_new[(j)*N+(i)]

	//fprintf(stderr,"seq_len=%i N=%i, nbest=%i\n", seq_len, N,nbest) ;
	
	T_STATES *psi=new T_STATES[seq_len*N*nbest] ;
	assert(psi!=NULL) ;
	short int *ktable=new short int[seq_len*N*nbest] ;
	assert(ktable!=NULL) ;
	short int *ktable_ends=new short int[seq_len*nbest] ;
	assert(ktable_ends!=NULL) ;

	REAL* tempvv=new REAL[nbest*N] ;
	assert(tempvv!=NULL) ;
	INT* tempii=new INT[nbest*N] ;
	assert(tempii!=NULL) ;

	T_STATES* path_ends = new T_STATES[seq_len*nbest] ;
	assert(path_ends!=NULL) ;
	REAL* delta= new REAL[N*nbest] ;
	assert(delta!=NULL) ;
	REAL* delta_new= new REAL[N*nbest] ;
	assert(delta_new!=NULL) ;
	REAL* delta_end= new REAL[seq_len*nbest] ;
	assert(delta_end!=NULL) ;

	INT * length = new INT[N*N] ;
	INT * length_new = new INT[N*N] ;
	const INT LARGE = 1024*1024*1024 ;

	{ // initialization
		for (T_STATES i=0; i<N; i++)
		{
			DELTA(i,0) = get_p(i) + SEQ(i,0) ;
			for (short int k=1; k<nbest; k++)
			{
				DELTA(i,k)=-math.INFTY ;
				KTAB(0,i,k)=0 ;
			}
			for (INT j=0; j<N; j++)
				LEN(i,j) = LARGE ;
		}
	}
	
	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		fprintf(stderr, "t=%i  ", t) ;
		for (INT i=0; i<N; i++)
			fprintf(stderr,"%i: %1.2f  ", i, DELTA(i,0)) ;
		fprintf(stderr, "\n") ;
		fprintf(stderr, "LEN(23,0)=%i\n",LEN(23,0)) ;
		fprintf(stderr, "LEN(23,2)=%i\n",LEN(23,2)) ;
		
		for (T_STATES j=0; j<N; j++)
		{
			const T_STATES num_elem   = trans_list_forward_cnt[j] ;
			const T_STATES *elem_list = trans_list_forward[j] ;
			const REAL *elem_val      = trans_list_forward_val[j] ;

			INT list_len=0 ;
			for (short int diff=0; diff<nbest; diff++)
			{
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;
					REAL  val   = DELTA(ii,diff) + elem_val[i] ;
					if (t==4 && j==5) fprintf(stderr, "val1=%1.2f ii=%i\n", val,ii) ;
					
					val        += lookup_step_penalty(STEP_PEN(j,ii), pos[t]-pos[t-1]) ;
					if (t==4 && j==5) fprintf(stderr, "val2=%1.2f ii=%i\n", val,ii) ;
					
					if (PEN(j, ii)!=NULL)
					{
						//fprintf(stderr, "- PEN(%i,%i) len=%i\n", j,ii,pos[t]-pos[t-1]) ;
						val += lookup_penalty(PEN(j,ii), pos[t]-pos[t-1]) ;
						if (t==4 && j==5) fprintf(stderr, "val3=%1.2f ii=%i\n", val,ii) ;
					}
					else
					{
						INT min_len = LARGE ;
						INT min_s   = -1 ;
						INT found   = 0 ;
						
						for (INT s=0; s<N; s++)
							if (PEN(j,s)!=NULL)
							{
								found = 1 ;
								if (t==4 && j==5 && ii==2)
									fprintf(stderr, "len(%i,%i)=%i\n",s,ii,LEN(s,ii)) ;
								if (LEN(s,ii)<min_len)
								{
									min_len = LEN(s,ii) ;
									min_s   = s ;
								}
							} ;

						if (found)
						{
							if (min_len>=LARGE)
							{
								assert(min_s == -1) ;
								val += -math.INFTY ;
							} else
							{
								if (t==4 && j==5) fprintf(stderr, "+ PEN(%i,%i) min_len=%i len=%i\n", j,min_s,min_len,min_len+pos[t]-pos[t-1]) ;
								val += lookup_penalty(PEN(j,min_s), min_len + 
													  pos[t]-pos[t-1]) ;
								if (t==4 && j==5) fprintf(stderr, "val4=%1.2f ii=%i min_s=%i\n", val,ii,min_s) ;
							}
						}
					} ;

					tempvv[list_len] = -val ;
					tempii[list_len] = diff*N + ii ;
					list_len++ ;
				}
			}
			math.qsort(tempvv, tempii, list_len) ;
			
			for (short int k=0; k<nbest; k++)
			{
				if (k<list_len)
				{
					DELTA_NEW(j,k)  = -tempvv[k] + SEQ(j,t);
					PSI(t,j,k)      = (tempii[k]%N) ;
					KTAB(t,j,k)     = (tempii[k]-(tempii[k]%N))/N ;
				}
				else
				{
					DELTA_NEW(j,k)  = -math.INFTY ;
					PSI(t,j,k)      = 0 ;
					KTAB(t,j,k)     = 0 ;
				}
			}
		}
		for (T_STATES j=0; j<N; j++)
		{
			if (DELTA_NEW(j,0)>math.ALMOST_NEG_INFTY)
			{
				for (T_STATES i=0; i<N; i++)
				{
//  for j=1:size(A,2),
//	if ~isinf(delta2(j)),
//	  LEN(:,j,t+1) = LEN(:,P(t,j),t)+pos(t+1)-pos(t) ;
//	  LEN(P(t,j),j,t+1) = pos(t+1)-pos(t) ;
//	end ;
//  end ;
					INT tmp1 = PSI(t,j,0) ;
					LEN_NEW(i,j) = LEN(i,tmp1)+pos[t]-pos[t-1] ;
				}
				INT tmp2 = PSI(t,j,0) ;
				LEN_NEW(tmp2,j) = pos[t]-pos[t-1] ;
			} else
				for (T_STATES i=0; i<N; i++)
					LEN_NEW(i,j) = LARGE  ;
		}
		
		math.swap(delta,delta_new) ;
		math.swap(length,length_new) ;
		
		if (t==seq_len-1)
		{ //termination
			INT list_len = 0 ;
			for (short int diff=0; diff<nbest; diff++)
			{
				for (T_STATES i=0; i<N; i++)
				{
					tempvv[list_len] = -(DELTA(i,diff)+get_q(i));
					tempii[list_len] = diff*N + i ;
					list_len++ ;
				}
			}
			math.qsort(tempvv, tempii, list_len) ;
			
			for (short int k=0; k<nbest; k++)
			{
				DELTA_END(t-1,k) = -tempvv[k] ;
				PATH_ENDS(t-1,k) = (tempii[k]%N) ;
				KTAB_ENDS(t-1,k) = (tempii[k]-(tempii[k]%N))/N ;
			}
		}
	}
	
	{ //state sequence backtracking
		REAL* sort_delta_end=new REAL[nbest] ;
		assert(sort_delta_end!=NULL) ;
		INT* sort_idx=new INT[nbest] ;
		assert(sort_idx!=NULL) ;
		
		INT take_iter=seq_len-1 ;
		for (short int k=0; k<nbest; k++)
		{
			sort_delta_end[k]=-DELTA_END(take_iter-1,k) ;
			sort_idx[k]=k ;
		}
		
		math.qsort(sort_delta_end, sort_idx, nbest) ;

		for (short int n=0; n<nbest; n++)
		{
			short int k=sort_idx[n] ;
			prob_nbest[n]=-sort_delta_end[n] ;

			assert(k<nbest && k>=0) ;
			assert(take_iter<seq_len && take_iter>=0) ;
			
			PATHS(take_iter,n) = PATH_ENDS(take_iter-1, k) ;
			short int q   = KTAB_ENDS(take_iter-1, k) ;
			
			for (INT t = take_iter; t>0; t--)
			{
				PATHS(t-1,n)=PSI(t, PATHS(t,n), q);
				q = KTAB(t, PATHS(t,n), q) ;
			}
		}
		delete[] sort_delta_end ;
		delete[] sort_idx ;
	}

	delete[] psi ;
	delete[] ktable;
	delete[] ktable_ends;

	delete[] tempvv ;
	delete[] tempii ;

	delete[] path_ends ;
	delete[] delta ;
	delete[] delta_new ;
	delete[] delta_end ;
}
