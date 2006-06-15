/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

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

#define PSI(t,j,k) psi[nbest*((t)*N+(j))+(k)]	
#define DELTA(t,j,k) delta[(j)*nbest*max_look_back+((t)%max_look_back)*nbest+k]
#define KTAB(t,j,k) ktable[nbest*((t)*N+j)+k]
#define PTAB(t,j,k) ptable[nbest*((t)*N+j)+k]
#define DELTA_END(k) delta_end[k]
#define KTAB_END(k) ktable_end[k]
#define PATH_END(k) path_end[k]
#define SEQ(j,t) seq[j+(t)*N]

void CHMM::best_path_trans_simple(const DREAL *seq, INT seq_len, short int nbest, 
								  DREAL *prob_nbest, INT *my_state_seq)
{
	INT max_look_back = 2 ;
	const INT look_back_buflen = max_look_back*nbest*N ;
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

	DREAL* oldtempvv=new DREAL[look_back_buflen] ;
	ASSERT(oldtempvv!=NULL) ;
	INT* oldtempii=new INT[look_back_buflen] ;
	ASSERT(oldtempii!=NULL) ;


	T_STATES* state_seq = new T_STATES[seq_len] ;
	ASSERT(state_seq!=NULL) ;
	INT * pos_seq   = new INT[seq_len] ;
	ASSERT(pos_seq!=NULL) ;

	{ // initialization

		for (T_STATES i=0; i<N; i++)
		{
		  DELTA(0,i,0) = get_p(i) + SEQ(i,0) ;        // get_p defined in HMM.h to be equiv to initial_state_distribution
			PSI(0,i,0)   = 0 ;
			KTAB(0,i,0)  = 0 ;
			PTAB(0,i,0)  = 0 ;
			for (short int k=1; k<nbest; k++)
			{
				DELTA(0,i,k)    = -CMath::INFTY ;
				PSI(0,i,0)      = 0 ;                  // <--- what's this for?
				KTAB(0,i,k)     = 0 ;
				PTAB(0,i,k)     = 0 ;
			}
		}
	}

	// recursion
	for (INT t=1; t<seq_len; t++)
	{
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
				
				INT old_list_len = 0 ;
				
				for (INT i=0; i<num_elem; i++)
				{
					T_STATES ii = elem_list[i] ;

					INT ts=t-1; 
					if (ts>=0)
					{
						bool ok=true ;
						
						if (ok)
						{

						  
						  for (short int diff=0; diff<nbest; diff++)
						    {
						      DREAL  val        = DELTA(ts,ii,diff) + elem_val[i] ;
						      DREAL mval = -val;

						      oldtempvv[old_list_len] = mval ;
						      oldtempii[old_list_len] = ii + diff*N + ts*N*nbest;
						      old_list_len++ ;
						    }
						}
					}
				}
				
				CMath::nmin<INT>(oldtempvv, oldtempii, old_list_len, nbest) ;

				int numEnt = 0;
				numEnt = old_list_len;

				double minusscore;
				long int fromtjk;

				for (short int k=0; k<nbest; k++)
				{
					if (k<numEnt)
					{
					    minusscore = oldtempvv[k];
					    fromtjk = oldtempii[k];
					    
					    DELTA(t,j,k)    = -minusscore + SEQ(j,t);
					    PSI(t,j,k)      = (fromtjk%N) ;
					    KTAB(t,j,k)     = (fromtjk%(N*nbest)-PSI(t,j,k))/N ;
					    PTAB(t,j,k)     = (fromtjk-(fromtjk%(N*nbest)))/(N*nbest) ;
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
				oldtempvv[list_len] = -(DELTA(seq_len-1,i,diff)+get_q(i)) ;
				oldtempii[list_len] = i + diff*N ;
				list_len++ ;
			}
		}
		
		CMath::nmin(oldtempvv, oldtempii, list_len, nbest) ;
		
		for (short int k=0; k<nbest; k++)
		{
			DELTA_END(k) = -oldtempvv[k] ;
			PATH_END(k) = (oldtempii[k]%N) ;
			KTAB_END(k) = (oldtempii[k]-PATH_END(k))/N ;
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
			}
			//my_state_seq[num_states+k*seq_len]=-1 ;
		}

	}
	delete[] delta ;
	delete[] psi ;
	delete[] ktable;
	delete[] ptable;

	delete[] ktable_end;
	delete[] path_end ;
	delete[] delta_end ;

	delete[] oldtempvv ;
	delete[] oldtempii ;

	delete[] state_seq ;
	delete[] pos_seq ;

}
