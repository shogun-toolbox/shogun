#include "lib/common.h"
#include "kernel/CommWordStringKernel.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCommWordStringKernel::CCommWordStringKernel(LONG size, bool use_sign_)
  : CStringKernel<WORD>(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	  dictionary_size(0), dictionary(NULL), dictionary_weights(NULL), use_sign(use_sign_)
{
}

CCommWordStringKernel::~CCommWordStringKernel() 
{
	if (get_is_initialized())
		delete_optimization() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
}
  
void CCommWordStringKernel::remove_lhs() 
{ 
	if (get_is_initialized())
		delete_optimization() ;
	if (lhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;

	lhs = NULL ; 
	rhs = NULL ; 
	initialized = false ;
	sqrtdiag_lhs = NULL ;
	sqrtdiag_rhs = NULL ;
} ;

void CCommWordStringKernel::remove_rhs()
{
	if (rhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs ;
	rhs = lhs ;
}

bool CCommWordStringKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CStringKernel<WORD>::init(l,r,do_init);
	initialized = false ;
	INT i;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
	  delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL ;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL ;

	sqrtdiag_lhs= new REAL[lhs->get_num_vectors()];

	for (i=0; i<lhs->get_num_vectors(); i++)
		sqrtdiag_lhs[i]=1;

	if (l==r)
		sqrtdiag_rhs=sqrtdiag_lhs;
	else {
		sqrtdiag_rhs= new REAL[rhs->get_num_vectors()];
		for (i=0; i<rhs->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;
	}

	assert(sqrtdiag_lhs);
	assert(sqrtdiag_rhs);

	this->lhs=(CStringFeatures<WORD>*) l;
	this->rhs=(CStringFeatures<WORD>*) l;

	//compute normalize to 1 values
	for (i=0; i<lhs->get_num_vectors(); i++)
	{
		sqrtdiag_lhs[i]=sqrt(compute(i,i));
		
		//trap divide by zero exception
		if (sqrtdiag_lhs[i]==0)
			sqrtdiag_lhs[i]=1e-16;
	}

	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if (sqrtdiag_lhs!=sqrtdiag_rhs)
	{
		this->lhs=(CStringFeatures<WORD>*) r;
		this->rhs=(CStringFeatures<WORD>*) r;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		{
		  sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
	}

	this->lhs=(CStringFeatures<WORD>*) l;
	this->rhs=(CStringFeatures<WORD>*) r;

	initialized = true ;
	return result;
}

void CCommWordStringKernel::cleanup()
{
}

bool CCommWordStringKernel::load_init(FILE* src)
{
	return false;
}

bool CCommWordStringKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CCommWordStringKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;

  WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(idx_a, alen);
  WORD* bvec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(idx_b, blen);

  REAL sqrt_a= 1 ;
  REAL sqrt_b= 1 ;
  if (initialized)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;

  REAL sqrt_both=sqrt_a*sqrt_b;

  INT result=0;

  INT left_idx=0;
  INT right_idx=0;

  if (use_sign)
  {
	  while (left_idx < alen && right_idx < blen)
	  {
		  if (avec[left_idx]==bvec[right_idx])
		  {
			  WORD sym=avec[left_idx];
			  
			  while (left_idx< alen && avec[left_idx]==sym)
				  left_idx++;
			  
			  while (right_idx< alen && bvec[right_idx]==sym)
				  right_idx++;
			  
			  result++ ;
		  }
		  else if (avec[left_idx]<bvec[right_idx])
			  left_idx++;
		  else
			  right_idx++;
	  }
  }
  else
  {
	  while (left_idx < alen && right_idx < blen)
	  {
		  if (avec[left_idx]==bvec[right_idx])
		  {
			  INT old_left_idx=left_idx;
			  INT old_right_idx=right_idx;
			  
			  WORD sym=avec[left_idx];
			  
			  while (left_idx< alen && avec[left_idx]==sym)
				  left_idx++;
			  
			  while (right_idx< alen && bvec[right_idx]==sym)
				  right_idx++;
			  
			  result+=(left_idx-old_left_idx)*(right_idx-old_right_idx);
		  }
		  else if (avec[left_idx]<bvec[right_idx])
			  left_idx++;
		  else
			  right_idx++;
	  }
  }
 
  return result/sqrt_both;
}



bool CCommWordStringKernel::init_optimization(INT count, INT *IDX, REAL * weights) 
{
	INT alen=-1 ;
	if (count<=0)
	{
		set_is_initialized(true) ;
		CIO::message(M_DEBUG, "empty set of SVs\n") ;
		return true ;
	} ;
	CIO::message(M_DEBUG, "initializing CCommWordStringKernel optimization\n") ;

	INT max_words=0 ;
	int i ;
	for (i=0; i<count; i++)
	{
		((CStringFeatures<WORD>*) lhs)->get_feature_vector(IDX[i], alen);
		max_words+=alen ;
	} ;
	WORD *words = new WORD[max_words] ;
	if (words==NULL)
		return false ;

	int num_words = 0 ;
	for (i=0; i<count; i++)
	{
		WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(IDX[i], alen);
		if (avec==NULL)
			return false ;
		int j;
		for (j=0; j<alen; j++)
		{
			assert(num_words<max_words) ;
			words[num_words++]=avec[j] ;
		}
	} ;
	CIO::message(M_DEBUG, "%i words\n", num_words) ;
	int num_unique_words = math.unique(words, num_words) ;
	CIO::message(M_DEBUG, "%i unique words\n", num_unique_words) ;
	
	{ // remove the memory overhead
		WORD* tmp = new WORD[num_unique_words] ;
		for (i=0; i<num_unique_words; i++)
			tmp[i]=words[i] ;
		delete[] words ;
		words = tmp ;
	}
	
	REAL* word_weights = new REAL[num_unique_words] ;
	if (word_weights==NULL)
	{
		CIO::message(M_ERROR, "out of memory\n") ;
		delete[] words ;
		return false ;
	}
	for (i=0; i<num_unique_words; i++)
		word_weights[i]=0 ;
	
	for (i=0; i<count; i++)
	{
		if ( (i % (count/10+1)) == 0)
			CIO::message(M_PROGRESS, "%3i%%  \r", 100*i/(count+1)) ;

		WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(IDX[i], alen);

		int j, last_j=0 ;
		if (use_sign)
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;
				int idx = math.fast_find(words, num_unique_words, avec[j-1]) ;
				assert(idx!=-1) ;
				word_weights[idx] += weights[i]/sqrtdiag_lhs[IDX[i]] ;
			}
			int idx = math.fast_find(words, num_unique_words, avec[alen-1]) ;
			assert(idx!=-1) ;
			word_weights[idx] += weights[i]/sqrtdiag_lhs[IDX[i]] ;
		}
		else
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;
				int idx = math.fast_find(words, num_unique_words, avec[j-1]) ;
				assert(idx!=-1) ;
				word_weights[idx] += weights[i]*(j-last_j)/sqrtdiag_lhs[IDX[i]] ;
				last_j = j ;
			}
			int idx = math.fast_find(words, num_unique_words, avec[alen-1]) ;
			assert(idx!=-1) ;
			word_weights[idx] += weights[i]*(alen-last_j)/sqrtdiag_lhs[IDX[i]] ;
		}
	}
	CIO::message(M_PROGRESS, "Done.         \n") ;
	
	dictionary         = words ;
	dictionary_weights = word_weights ;
	dictionary_size    = num_unique_words ;
	
	set_is_initialized(true) ;
	return true ;
} ;

void CCommWordStringKernel::delete_optimization() 
{
	if (get_is_initialized())
	{
		CIO::message(M_DEBUG, "deleting CCommWordStringKernel optimization\n") ;
		delete[] dictionary ;
		delete[] dictionary_weights;
		
		dictionary_size=0 ;
		dictionary=NULL ;
		dictionary_weights=NULL ;
		
		set_is_initialized(false) ;
	}
	else
		CIO::message(M_ERROR, "CCommWordStringKernel optimization not initialized\n") ;
} ;

REAL CCommWordStringKernel::compute_optimized(INT i) 
{ 
	if (!get_is_initialized())
	{
		CIO::message(M_ERROR, "CCommWordStringKernel optimization not initialized\n") ;
		return 0 ; 
	}

	REAL result = 0 ;
	INT alen = -1 ;
	WORD* avec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(i, alen);
	assert(avec!=NULL) ;
	assert(alen!=-1) ;

	int j, last_j=0 ;
	if (use_sign)
	{
		for (j=1; j<alen; j++)
		{
			if (avec[j]==avec[j-1])
				continue ;
			int idx = math.fast_find(dictionary, dictionary_size, avec[j-1]) ;
			if (idx!=-1)
				result += dictionary_weights[idx] ;
		}
		int idx = math.fast_find(dictionary, dictionary_size, avec[alen-1]) ;
		if (idx!=-1)
			result += dictionary_weights[idx] ;
	}
	else
	{
		for (j=1; j<alen; j++)
		{
			if (avec[j]==avec[j-1])
				continue ;
			int idx = math.fast_find(dictionary, dictionary_size, avec[j-1]) ;
			if (idx!=-1)
				result += dictionary_weights[idx]*(j-last_j) ;
			last_j = j ;
		}
		int idx = math.fast_find(dictionary, dictionary_size, avec[alen-1]) ;
		if (idx!=-1)
			result += dictionary_weights[idx]*(alen-last_j) ;
	}
	
	//((CWordFeatures*) rhs)->free_feature_vector(avec, i, afree);

	return result/sqrtdiag_rhs[i] ;
} ;
