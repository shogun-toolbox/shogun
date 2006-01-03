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

#include "fibheap.h"

#ifdef SUNOS
extern "C" int	finite(double);
#endif

// of the first three preprocessor directives below
//    0 means dont use it ever, 
//    1 means use it just for printing if DOPRINT is 1, and 
//    2 means use it for final result 
//  only one of the three possibilities below can be 2

#define USEHEAP 0
#define USEORIGINALLIST 0
#define USEFIXEDLENLIST 2
#define DOPRINT 0

static const INT num_degrees = 4;
static const INT num_svms = 8 ;

//static const INT word_degree[num_degrees] = {1,6} ;
//static const INT cum_num_words[num_degrees+1] = {0,4,4100} ;
//static const INT num_words[num_degrees] = {4,4096} ;
//static const INT word_degree[num_degrees] = {1,2,3,4,5,6} ;
//static const INT cum_num_words[num_degrees+1] = {0,4,20,84,340,1364,5460} ;
//static const INT num_words[num_degrees] = {4,20,64,256,1024,4096} ;
static const INT word_degree[num_degrees] = {3,4,5,6} ;
static const INT cum_num_words[num_degrees+1] = {0,64,320,1344,5440} ;
static const INT num_words[num_degrees] = {64,256,1024,4096} ;

static bool word_used[num_degrees][4096] ;
static REAL svm_values_unnormalized[num_degrees][num_svms] ;
static REAL *dict_weights ;
static INT svm_pos_start[num_degrees] ;
static INT num_unique_words[num_degrees] ;

inline void translate_from_single_order(WORD* obs, INT sequence_length, 
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
		//if (value>=num_words[order-1])
		//	fprintf(stderr,"%i %i\n", value, num_words[order-1]) ;
		//assert(value<num_words[order]) ;
	}
	if (start>0)
		for (i=start; i<sequence_length; i++)	
			obs[i-start]=obs[i];
}


struct svm_values_struct
{
  INT maxlookback ;
  INT seqlen;
  INT *num_unique_words ;

  REAL ** svm_values_unnormalized ;
  REAL * svm_values ;
  bool ** word_used ;
} ;

inline void init_svm_value(struct svm_values_struct & svs, INT start_pos, INT seqlen, INT howmuchlookback)
{
	/*
	  See find_svm_values_till_pos for comments
	  
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  where t_end is the end of all segments we are currently looking at
	*/
	
	if (!svs.svm_values)
	{
		svs.num_unique_words        = new INT[num_degrees] ;
		svs.svm_values              = new REAL[seqlen*num_svms] ;
		svs.svm_values_unnormalized = new (REAL*)[num_degrees] ;
		svs.word_used               = new (bool*)[num_degrees] ;
		for (INT j=0; j<num_degrees; j++)
		{
			//svs.svm_values[j]              = new REAL[seqlen*num_svms] ;
			svs.svm_values_unnormalized[j] = new REAL[num_svms] ;
			svs.word_used[j]               = new bool[num_words[j]] ;
		}
	}
	
	for (INT i=0; i<seqlen*num_svms; i++)       // initializing this for safety, though we should be able to live without it
		svs.svm_values[i] = 0;

	for (INT j=0; j<num_degrees; j++)
	{		
		for (INT s=0; s<num_svms; s++)
			svs.svm_values_unnormalized[j][s] = 0 ;
		
		for (INT i=0; i<num_words[j]; i++)
			svs.word_used[j][i] = false ;

		svs.num_unique_words[j] = 0 ;
	}
	
	svs.maxlookback = howmuchlookback ;
	svs.seqlen = seqlen;
}

inline void clear_svm_value(struct svm_values_struct & svs) 
{
	if (NULL != svs.svm_values)
	{
		for (INT j=0; j<num_degrees; j++)
			delete[] svs.word_used[j] ;
		for (INT j=0; j<num_degrees; j++)
			delete[] svs.svm_values_unnormalized[j] ;
		
		delete[] svs.svm_values_unnormalized;
		delete[] svs.svm_values;
		delete[] svs.word_used;
		svs.word_used=NULL ;
		svs.svm_values=NULL ;
		svs.svm_values_unnormalized=NULL ;
	}
}


inline void find_svm_values_till_pos(WORD** wordstr,  const INT *pos,  INT t_end, struct svm_values_struct &svs)
{
	/*
	  wordstr is a vector of L n-gram indices, with wordstr(i) representing a number betweeen 0 and 4095 corresponding to the 6-mer in genestr(i-5:i) 
	  pos is a vector of candidate transition positions (it is input to best_path_trans)
	  t_end is some index in pos
	  
	  svs has been initialized by init_svm_values
	  
	  At the end of this procedure, 
	  svs.svm_values[i+s*svs.seqlen] has the value of the s-th SVM on genestr(pos(t_end-i):pos(t_end)) 
	  for every i satisfying pos(t_end)-pos(t_end-i) <= svs.maxlookback
	  
	  The SVM weights are precomputed in dict_weights
	*/
	for (INT j=0; j<num_degrees; j++)
	{
		INT plen = 1;
		INT ts = t_end-1;        // index in pos; pos(ts) and pos(t) are indices of wordstr
		INT offset;
		
		/*
		  for (INT s=0; s<num_svms; s++)
		  {
		  offset = s*svs.seqlen;
		  for (INT i=0;i<word_degree; i++)
		  svs.svm_values[i+offset] = 0;
		  }
		*/
		
		INT posprev = pos[t_end]-word_degree[j]+1;
		INT poscurrent = pos[ts];
		
		if (poscurrent<0)
			poscurrent = 0;
		
		INT len = pos[t_end] - poscurrent;
		
		while ((ts>=0) && (len <= svs.maxlookback))
		{
			for (int i=posprev-1 ; (i>=poscurrent) && (i>=0) ; i--)
			{
				// 	  if (word_degree > (pos[t_end]-pos[ts]))
				//	    fprintf(fid, " *******  i=%d , wordstr[i]=%d   dict_weights[1,wordstr[i]]=%f  t_end=%d, ts=%d  pos[t_end]=%d  pos[ts]=%d   posprev=%d\n", i, wordstr[i], dict_weights[wordstr[i]], t_end,ts,pos[t_end],pos[ts],posprev);
				
				if (wordstr[j][i]>=num_words[j])
					fprintf(stderr, "wordstr[%i][%i]=%i\n", j, i, wordstr[j][i]) ;
				assert(wordstr[j][i]<num_words[j]) ;
				if (!svs.word_used[j][wordstr[j][i]])
				{
					for (INT s=0; s<num_svms; s++)
						svs.svm_values_unnormalized[j][s]+=dict_weights[wordstr[j][i]+s*cum_num_words[num_degrees]+cum_num_words[j]] ;
					
					svs.word_used[j][wordstr[j][i]]=true ;
					svs.num_unique_words[j]++ ;
				}
			}
			double normalization_factor = 1.0;
			if (svs.num_unique_words[j] > 0)
				normalization_factor = sqrt((double)svs.num_unique_words[j]);
			for (INT s=0; s<num_svms; s++)
			{
				offset = s*svs.seqlen;
				if (j==0)
					svs.svm_values[offset+plen]=0 ;
				svs.svm_values[offset+plen] += svs.svm_values_unnormalized[j][s] / normalization_factor;
			}
			
			if (posprev > poscurrent)         // remember posprev initially set to pos[t_end]-word_degree+1... pos[ts] could be e.g. pos[t_end]-2
				posprev = poscurrent;           
			
			ts--;
			plen++;
			
			if (ts>=0)
			{
				poscurrent=pos[ts];
				if (poscurrent<0)
					poscurrent = 0;
				len = pos[t_end] - poscurrent;
			}
		}
	}
}



inline void reset_svm_value(INT pos, INT * last_svm_pos, REAL * svm_value) 
{
	for (INT j=0; j<num_degrees; j++)
	{
		for (INT i=0; i<num_words[j]; i++)
			word_used[j][i]=false ;
		for (INT s=0; s<num_svms; s++)
			svm_values_unnormalized[j][s] = 0 ;
		num_unique_words[j]=0 ;
		last_svm_pos[j] = pos - word_degree[j]+1 ;
		svm_pos_start[j] = pos - word_degree[j] ;
	}
	for (INT s=0; s<num_svms; s++)
		svm_value[s] = 0 ;
}

void extend_svm_value(WORD** wordstr, INT pos, INT *last_svm_pos, REAL* svm_value) 
{
	bool did_something = false ;
	for (INT j=0; j<num_degrees; j++)
	{
		for (int i=last_svm_pos[j]-1; (i>=pos) && (i>=0); i--)
		{
			if (wordstr[j][i]>=num_words[j])
				CIO::message(M_DEBUG, "wordstr[%i]=%i\n", i, wordstr[j][i]) ;

			assert(wordstr[j][i]<num_words[j]) ;
			if (!word_used[j][wordstr[j][i]])
			{
				for (INT s=0; s<num_svms; s++)
					svm_values_unnormalized[j][s]+=dict_weights[wordstr[j][i]+s*cum_num_words[num_degrees]+cum_num_words[j]] ;
				
				word_used[j][wordstr[j][i]]=true ;
				num_unique_words[j]++ ;
				did_something=true ;
			} ;
		} ;
		if (num_unique_words[j]>0)
			last_svm_pos[j]=pos ;
	} ;
	
	if (did_something)
		for (INT s=0; s<num_svms; s++)
		{
			svm_value[s]=0.0 ;
			for (INT j=0; j<num_degrees; j++)
				if (num_unique_words[j]>0)
					svm_value[s]+= svm_values_unnormalized[j][s]/sqrt((double)num_unique_words[j]) ;  // full normalization
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
						   REAL *&PEN_values, REAL *&PEN_input_values, 
						   INT &num_PEN_id, bool use_orf)
{
	const INT default_look_back = 30000 ;
	INT max_look_back = default_look_back ;
	bool use_svm = false ;
	assert(dict_len==num_svms*cum_num_words[num_degrees]) ;
	dict_weights=dictionary_weights ;
	int offset=0;

	REAL svm_value[num_svms] ;
	
	{ // initialize svm_svalue
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;
	}
	
	{ // determine maximal length of look-back
		for (INT i=0; i<N; i++)
			for (INT j=0; j<N; j++)
			{
				struct penalty_struct *penij=PEN(i,j) ;
				while (penij!=NULL)
				{
					if (penij->max_len>max_look_back)
						max_look_back=penij->max_len ;
					if (penij->use_svm)
						use_svm=true ;
					if (penij->id+1>num_PEN_id)
						num_PEN_id=penij->id+1 ;
					penij=penij->next_pen ;
				} 
			}
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

	#if USEFIXEDLENLIST > 0
	REAL* fixedtempvv=new REAL[look_back_buflen] ;
	assert(fixedtempvv!=NULL) ;
	INT* fixedtempii=new INT[look_back_buflen] ;
	assert(fixedtempii!=NULL) ;
	#endif


	// we always use oldtempvv and oldtempii, even if USEORIGINALLIST is 0
	// as i didnt change the backtracking stuff

	REAL* oldtempvv=new REAL[look_back_buflen] ;
	assert(oldtempvv!=NULL) ;
	INT* oldtempii=new INT[look_back_buflen] ;
	assert(oldtempii!=NULL) ;


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
	WORD* wordstr[num_degrees] ;
	{
		for (INT j=0; j<num_degrees; j++)
		{
			wordstr[j]=NULL ;
			if (use_svm)
			{
				assert(dict_weights!=NULL) ;
				wordstr[j]=new WORD[genestr_len] ;
				for (INT i=0; i<genestr_len; i++)
					switch (genestr[i])
					{
					case 'a': wordstr[j][i]=0 ; break ;
					case 'c': wordstr[j][i]=1 ; break ;
					case 'g': wordstr[j][i]=2 ; break ;
					case 't': wordstr[j][i]=3 ; break ;
					default: assert(0) ;
					}
				translate_from_single_order(wordstr[j], genestr_len,
											word_degree[j]-1, word_degree[j]) ;
			}
		}
	}
	
  	
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

	struct svm_values_struct svs;
	svs.svm_values = NULL;
	svs.svm_values_unnormalized = NULL;
	svs.word_used = NULL;

	// recursion
	for (INT t=1; t<seq_len; t++)
	{
		if (is_big && t%(seq_len/1000)==1)
			CIO::progress(t, 0, seq_len);
		
		init_svm_value(svs, t, seq_len, max_look_back);
		find_svm_values_till_pos(wordstr, pos, t, svs);  
	
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
				
				#if USEFIXEDLENLIST > 0
				INT fixed_list_len = 0 ;
				#endif

				#if USEORIGINALLIST > 0
				INT old_list_len = 0 ;
				#endif
				
				#if USEHEAP > 0
				Heap* tempheap = new Heap;
				#endif

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
						int plen=t-ts;

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

						  for (INT ss=0; ss<num_svms; ss++)
						    {
						      offset = ss*svs.seqlen;
						      svm_value[ss]=svs.svm_values[offset+plen];
						    }

						  REAL input_value ;
						  REAL pen_val = lookup_penalty(penalty, pos[t]-pos[ts], svm_value, true, input_value) ;
						  for (short int diff=0; diff<nbest; diff++)
						    {
						      REAL  val        = DELTA(ts,ii,diff) + elem_val[i] ;
						      val             += pen_val ;
						      REAL mval = -val;

                                                      #if USEHEAP > 0
						      tempheap->Insert(mval,ii + diff*N + ts*N*nbest);
						      #endif

						      #if USEORIGINALLIST > 0
						      oldtempvv[old_list_len] = mval ;
						      oldtempii[old_list_len] = ii + diff*N + ts*N*nbest;
						      old_list_len++ ;
						      #endif

						      #if USEFIXEDLENLIST > 0
						      
						      /* only place -val in fixedtempvv if it is one of the nbest lowest values in there */
						      /* fixedtempvv[i], i=0:nbest-1, is sorted so that fixedtempvv[0] <= fixedtempvv[1] <= ...*/
						      /* fixed_list_len has the number of elements in fixedtempvv */

						      if ((fixed_list_len < nbest) || (mval < fixedtempvv[fixed_list_len-1]))
							{
							  if ( (fixed_list_len<nbest) && ((0==fixed_list_len) || (mval>fixedtempvv[fixed_list_len-1])) )
							    {
							      fixedtempvv[fixed_list_len] = mval ;
							      fixedtempii[fixed_list_len] = ii + diff*N + ts*N*nbest;
							      fixed_list_len++ ;
							    }
							  else  // must have mval < fixedtempvv[fixed_list_len-1]
							    {
							      int addhere = fixed_list_len;
							      while ((addhere > 0) && (mval < fixedtempvv[addhere-1]))
								addhere--;
							      
							      // move everything from addhere+1 one forward 
							      
							      for (int jj=fixed_list_len-1; jj>addhere; jj--)
								{
								  fixedtempvv[jj] = fixedtempvv[jj-1];
								  fixedtempii[jj] = fixedtempii[jj-1];
								}
							      
							      fixedtempvv[addhere] = mval;
							      fixedtempii[addhere] = ii + diff*N + ts*N*nbest;
							      
							      if (fixed_list_len < nbest)
								fixed_list_len++;
							    }
							}
						      #endif

						    }
						}
					}
				}
				
				#if USEORIGINALLIST > 0
				CMath::nmin<INT>(oldtempvv, oldtempii, old_list_len, nbest) ;
				#endif


				int numEnt = 0;
				#if USEHEAP == 2
				numEnt = tempheap->GetNumNodes();
				#elif USEORIGINALLIST == 2
				numEnt = old_list_len;
				#elif USEFIXEDLENLIST == 2
				numEnt = fixed_list_len;
                                #endif

				double minusscore;
				long int fromtjk;

				for (short int k=0; k<nbest; k++)
				{
					if (k<numEnt)
					{
#if (USEHEAP == 2)
					    tempheap->ExtractMin(minusscore,fromtjk);
#elif (USEORIGINALLIST == 2)
					    minusscore = oldtempvv[k];
					    fromtjk = oldtempii[k];
#elif (USEFIXEDLENLIST == 2)
					    minusscore = fixedtempvv[k];
					    fromtjk = fixedtempii[k];
#endif
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

				#if USEHEAP > 0
				delete tempheap;
				#endif
			}
		}
	}

	clear_svm_value(svs);


	
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
				my_pos_seq[i+k*seq_len]   = pos_seq[num_states-i-1] ;
			}
			my_state_seq[num_states+k*seq_len]=-1 ;
			my_pos_seq[num_states+k*seq_len]=-1 ;
		}

		REAL svm_value[num_svms] ;
		for (INT s=0; s<num_svms; s++)
			svm_value[s]=0 ;

		// one more for the emissions: the first
		num_PEN_id++ ;
        // allocate memory
		PEN_values = new REAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_values[s]=0 ;
		PEN_input_values = new REAL[num_PEN_id*seq_len*nbest] ;
		for (INT s=0; s<num_PEN_id*seq_len*nbest; s++)
			PEN_input_values[s]=0 ;
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
				INT last_svm_pos[num_degrees] ;
				for (INT qq=0; qq<num_degrees; qq++)
					last_svm_pos[qq]=-1 ;
				
				PEN_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] += SEQ(to_state, to_pos) ;
				PEN_input_values[num_PEN_id-1 + i*num_PEN_id + seq_len*num_PEN_id*k] = to_state + to_pos*1000 ;

				reset_svm_value(pos[to_pos], last_svm_pos, svm_value) ;
				extend_svm_value(wordstr, pos[from_pos], last_svm_pos, svm_value) ;

				struct penalty_struct *penalty = PEN(to_state, from_state) ;
				while (penalty)
				{
					REAL input_value=0 ;
					REAL pen_val = lookup_penalty(penalty, pos[to_pos]-pos[from_pos], svm_value, false, input_value) ;
					PEN_values[penalty->id + i*num_PEN_id + seq_len*num_PEN_id*k] += pen_val ;
					PEN_input_values[penalty->id + i*num_PEN_id + seq_len*num_PEN_id*k] += input_value ;
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

	for (INT j=0; j<num_degrees; j++)
		delete[] wordstr[j] ;

	delete[] delta ;
	delete[] psi ;
	delete[] ktable;
	delete[] ptable;

	delete[] ktable_end;
	delete[] path_end ;
	delete[] delta_end ;


	delete[] oldtempvv ;
	delete[] oldtempii ;


	#if USEFIXEDLENLIST
	delete[] fixedtempvv ;
	delete[] fixedtempii ;
	#endif

	delete[] state_seq ;
	delete[] pos_seq ;
	delete[] genestr_stop ;

}
