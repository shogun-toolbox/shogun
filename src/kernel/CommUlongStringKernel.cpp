#include "lib/common.h"
#include "kernel/CommUlongStringKernel.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCommUlongStringKernel::CCommUlongStringKernel(LONG size, bool use_sign_, 
											 E_NormalizationType normalization_) 
  : CStringKernel<ULONG>(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	dictionary_size(0), dictionary(NULL), dictionary_weights(NULL), use_sign(use_sign_), 
	normalization(normalization_)
{
}

CCommUlongStringKernel::~CCommUlongStringKernel() 
{
	cleanup();
}
  
void CCommUlongStringKernel::remove_lhs() 
{ 
	delete_optimization();

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

void CCommUlongStringKernel::remove_rhs()
{
	if (rhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs ;
	rhs = lhs ;
}

bool CCommUlongStringKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CStringKernel<ULONG>::init(l,r,do_init);
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

	this->lhs=(CStringFeatures<ULONG>*) l;
	this->rhs=(CStringFeatures<ULONG>*) l;

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
		this->lhs=(CStringFeatures<ULONG>*) r;
		this->rhs=(CStringFeatures<ULONG>*) r;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		{
		  sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
	}

	this->lhs=(CStringFeatures<ULONG>*) l;
	this->rhs=(CStringFeatures<ULONG>*) r;

	initialized = true ;
	return result;
}

void CCommUlongStringKernel::cleanup()
{
	delete_optimization();

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;

	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	initialized=false;
}

bool CCommUlongStringKernel::load_init(FILE* src)
{
	return false;
}

bool CCommUlongStringKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CCommUlongStringKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;

  ULONG* avec=((CStringFeatures<ULONG>*) lhs)->get_feature_vector(idx_a, alen);
  ULONG* bvec=((CStringFeatures<ULONG>*) rhs)->get_feature_vector(idx_b, blen);

  REAL sqrt_both=1;
  if (initialized && normalization!=E_NO_NORMALIZATION)
    {
		REAL sqrt_a=sqrtdiag_lhs[idx_a] ;
		REAL sqrt_b=sqrtdiag_rhs[idx_b] ;
		sqrt_both=sqrt_a*sqrt_b;
    } ;


  INT result=0;

  INT left_idx=0;
  INT right_idx=0;

  if (use_sign)
  {
	  while (left_idx < alen && right_idx < blen)
	  {
		  if (avec[left_idx]==bvec[right_idx])
		  {
			  ULONG sym=avec[left_idx];
			  
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
			  
			  ULONG sym=avec[left_idx];
			  
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
  switch (normalization)
  {
  case E_NO_NORMALIZATION:
	  return result ;
  case E_SQRT_NORMALIZATION:
	  return result/sqrt(sqrt_both) ;
  case E_FULL_NORMALIZATION:
	  return result/sqrt_both ;
  case E_SQRTLEN_NORMALIZATION:
	  return result/sqrt(sqrt(alen*blen)) ;
  case E_LEN_NORMALIZATION:
	  return result/sqrt(alen*blen) ;
  case E_SQLEN_NORMALIZATION:
	  return result/(alen*blen) ;
  default:
	  assert(0) ;
  }
  return result ;
}



bool CCommUlongStringKernel::init_optimization(INT count, INT *IDX, REAL * weights) 
{
	INT alen=-1 ;
	if (count<=0)
	{
		set_is_initialized(true) ;
		CIO::message(M_DEBUG, "empty set of SVs\n") ;
		return true ;
	} ;
	CIO::message(M_DEBUG, "initializing CCommUlongStringKernel optimization\n") ;

	INT max_words=1000 ; // use a bit more ...
	int i ;
	for (i=0; i<count; i++)
	{
		ULONG* avec=((CStringFeatures<ULONG>*) lhs)->get_feature_vector(IDX[i], alen);
		if (avec==NULL)
			return false ;
		max_words+=alen ;
	} ;
	ULONG *words = new ULONG[max_words] ;
	if (words==NULL)
		return false ;
	CIO::message(M_DEBUG, "max_words=%i\n", max_words) ;
	
	int num_words = 0 ;
	for (i=0; i<count; i++)
	{
		ULONG* avec=((CStringFeatures<ULONG>*) lhs)->get_feature_vector(IDX[i], alen);
		if (avec==NULL)
			return false ;
		int j;
		for (j=0; j<alen; j++)
		{
			if (num_words>=max_words)
				CIO::message(M_DEBUG, "num_words=%i\n", num_words) ;
			//assert(num_words<max_words) ;
			words[num_words++]=avec[j] ;
		}
	} ;
	CIO::message(M_DEBUG, "%i words\n", num_words) ;
	int num_unique_words = CMath::unique(words, num_words) ;
	CIO::message(M_DEBUG, "%i unique words\n", num_unique_words) ;
	
	{ // remove the memory overhead
		ULONG* tmp = new ULONG[num_unique_words] ;
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
			CIO::progress(i, 0, count);

		ULONG* avec=((CStringFeatures<ULONG>*) lhs)->get_feature_vector(IDX[i], alen);

		int j, last_j=0 ;
		if (use_sign)
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;
				int idx = CMath::fast_find(words, num_unique_words, avec[j-1]) ;
				assert(idx!=-1) ;
				switch (normalization)
				{
				case E_NO_NORMALIZATION:
					word_weights[idx] += weights[i] ;
					break ;
				case E_SQRT_NORMALIZATION:
					word_weights[idx] += weights[i]/sqrt(sqrtdiag_lhs[IDX[i]]) ;
					break ;
				case E_FULL_NORMALIZATION:
					word_weights[idx] += weights[i]/sqrtdiag_lhs[IDX[i]] ;
					break ;
				case E_SQRTLEN_NORMALIZATION:
					word_weights[idx] += weights[i]/sqrt(sqrt(alen)) ;
					break ;
				case E_LEN_NORMALIZATION:
					word_weights[idx] += weights[i]/sqrt(alen) ;
					break ;
				case E_SQLEN_NORMALIZATION:
					word_weights[idx] += weights[i]/alen ;
					break ;
				default:
					assert(0) ;
				}
			}
			int idx = CMath::fast_find(words, num_unique_words, avec[alen-1]) ;
			assert(idx!=-1) ;
			switch (normalization)
			{
			case E_NO_NORMALIZATION:
				word_weights[idx] += weights[i] ;
				break ;
			case E_SQRT_NORMALIZATION:
				word_weights[idx] += weights[i]/sqrt(sqrtdiag_lhs[IDX[i]]) ;
				break ;
			case E_FULL_NORMALIZATION:
				word_weights[idx] += weights[i]/sqrtdiag_lhs[IDX[i]] ;
				break ;
			case E_SQRTLEN_NORMALIZATION:
				word_weights[idx] += weights[i]/sqrt(sqrt(alen)) ;
				break ;
			case E_LEN_NORMALIZATION:
				word_weights[idx] += weights[i]/sqrt(alen) ;
				break ;
			case E_SQLEN_NORMALIZATION:
				word_weights[idx] += weights[i]/alen ;
				break ;
			default:
				assert(0) ;
			}
		}
		else
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;
				int idx = CMath::fast_find(words, num_unique_words, avec[j-1]) ;
				assert(idx!=-1) ;
				switch (normalization)
				{
				case E_NO_NORMALIZATION:
					word_weights[idx] += weights[i]*(j-last_j) ;
					break ;
				case E_SQRT_NORMALIZATION:
					word_weights[idx] += weights[i]*(j-last_j)/sqrt(sqrtdiag_lhs[IDX[i]]) ;	
					break ;
				case E_FULL_NORMALIZATION:
					word_weights[idx] += weights[i]*(j-last_j)/sqrtdiag_lhs[IDX[i]] ;
					break ;
				case E_SQRTLEN_NORMALIZATION:
					word_weights[idx] += weights[i]*(j-last_j)/sqrt(sqrt(alen)) ;
					break ;
				case E_LEN_NORMALIZATION:
					word_weights[idx] += weights[i]*(j-last_j)/sqrt(alen) ;
					break ;
				case E_SQLEN_NORMALIZATION:
					word_weights[idx] += weights[i]*(j-last_j)/alen ;
					break ;
				default:
					assert(0) ;
				}
				last_j = j ;
			}
			int idx = CMath::fast_find(words, num_unique_words, avec[alen-1]) ;
			assert(idx!=-1) ;
			switch (normalization)
			{
			case E_NO_NORMALIZATION:
				word_weights[idx] += weights[i]*(alen-last_j) ;
				break ;
			case E_SQRT_NORMALIZATION:
				word_weights[idx] += weights[i]*(alen-last_j)/sqrt(sqrtdiag_lhs[IDX[i]]) ;
				break ;
			case E_FULL_NORMALIZATION:
				word_weights[idx] += weights[i]*(alen-last_j)/sqrtdiag_lhs[IDX[i]] ;
				break ;
			case E_SQRTLEN_NORMALIZATION:
				word_weights[idx] += weights[i]*(alen-last_j)/sqrt(sqrt(alen)) ;
				break ;
			case E_SQLEN_NORMALIZATION:
				word_weights[idx] += weights[i]*(alen-last_j)/alen ;
				break ;
			case E_LEN_NORMALIZATION:
				word_weights[idx] += weights[i]*(alen-last_j)/sqrt(alen) ;
				break ;
			default:
				assert(0) ;
			}
		}
	}
	CIO::message(M_MESSAGEONLY, "Done.         \n") ;
	
	dictionary         = words ;
	dictionary_weights = word_weights ;
	dictionary_size    = num_unique_words ;
	
	set_is_initialized(true) ;
	return true ;
} ;

bool CCommUlongStringKernel::delete_optimization() 
{
	CIO::message(M_DEBUG, "deleting CCommUlongStringKernel optimization\n") ;
	delete[] dictionary ;
	delete[] dictionary_weights;

	dictionary_size=0 ;
	dictionary=NULL ;
	dictionary_weights=NULL ;

	set_is_initialized(false) ;

	return true;
}

REAL CCommUlongStringKernel::compute_optimized(INT i) 
{ 
	if (!get_is_initialized())
	{
		CIO::message(M_ERROR, "CCommUlongStringKernel optimization not initialized\n") ;
		return 0 ; 
	}

	REAL result = 0 ;
	INT alen = -1 ;
	ULONG* avec=((CStringFeatures<ULONG>*) rhs)->get_feature_vector(i, alen);
	assert(avec!=NULL) ;
	assert(alen!=-1) ;

	int j, last_j=0 ;
	if (use_sign)
	{
		for (j=1; j<alen; j++)
		{
			if (avec[j]==avec[j-1])
				continue ;
			int idx = CMath::fast_find(dictionary, dictionary_size, avec[j-1]) ;
			if (idx!=-1)
				result += dictionary_weights[idx] ;
		}
		int idx = CMath::fast_find(dictionary, dictionary_size, avec[alen-1]) ;
		if (idx!=-1)
			result += dictionary_weights[idx] ;
	}
	else
	{
		for (j=1; j<alen; j++)
		{
			if (avec[j]==avec[j-1])
				continue ;
			int idx = CMath::fast_find(dictionary, dictionary_size, avec[j-1]) ;
			if (idx!=-1)
				result += dictionary_weights[idx]*(j-last_j) ;
			last_j = j ;
		}
		int idx = CMath::fast_find(dictionary, dictionary_size, avec[alen-1]) ;
		if (idx!=-1)
			result += dictionary_weights[idx]*(alen-last_j) ;
	}
	
	switch (normalization)
	{
	case E_NO_NORMALIZATION:
		return result ;
	case E_SQRT_NORMALIZATION:
		return result/sqrt(sqrtdiag_rhs[i]) ;
	case E_FULL_NORMALIZATION:
		return result/sqrtdiag_rhs[i] ;
	case E_SQRTLEN_NORMALIZATION:
		return result/sqrt(sqrt(alen)) ;
	case E_LEN_NORMALIZATION:
		return result/sqrt(alen) ;
	case E_SQLEN_NORMALIZATION:
		return result/alen ;
	default:
		assert(0) ;
	}
	return result ;
}
