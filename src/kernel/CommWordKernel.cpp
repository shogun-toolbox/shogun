#include "lib/common.h"
#include "kernel/CommWordKernel.h"
#include "features/Features.h"
#include "features/WordFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCommWordKernel::CCommWordKernel(LONG size, bool use_sign_)
	: CWordKernel(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	  dictionary_size(0), dictionary(NULL), dictionary_weights(NULL), use_sign(use_sign_)
{
}

CCommWordKernel::~CCommWordKernel() 
{
	cleanup();
}
  
void CCommWordKernel::remove_lhs() 
{ 
	delete_optimization();

	if (lhs)
		cache_reset();

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;

	lhs = NULL ; 
	rhs = NULL ; 
	initialized = false ;
	sqrtdiag_lhs = NULL ;
	sqrtdiag_rhs = NULL ;
} ;

void CCommWordKernel::remove_rhs()
{
	if (rhs)
		cache_reset() ;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs = sqrtdiag_lhs ;
	rhs = lhs ;
}

bool CCommWordKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CWordKernel::init(l,r,do_init);
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
	else
	{
		sqrtdiag_rhs= new REAL[rhs->get_num_vectors()];
		for (i=0; i<rhs->get_num_vectors(); i++)
			sqrtdiag_rhs[i]=1;
	}

	assert(sqrtdiag_lhs);
	assert(sqrtdiag_rhs);

	this->lhs=(CWordFeatures*) l;
	this->rhs=(CWordFeatures*) l;

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
		this->lhs=(CWordFeatures*) r;
		this->rhs=(CWordFeatures*) r;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		{
			sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
	}

	this->lhs=(CWordFeatures*) l;
	this->rhs=(CWordFeatures*) r;

	initialized = true ;
	return result;
}

void CCommWordKernel::cleanup()
{
	delete_optimization();

	initialized=false;
	dictionary_size=0;
	dictionary=NULL;
	dictionary_weights=NULL;
	
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;

	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;
}

bool CCommWordKernel::load_init(FILE* src)
{
	return false;
}

bool CCommWordKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CCommWordKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen==blen);

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
  
  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
  
  return result/sqrt_both;
}

void CCommWordKernel::add_to_normal(INT vec_idx, REAL weight)
{
	INT alen=-1;
	bool afree;
	WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(vec_idx, alen, afree);

	int j, last_j=0 ;
	if (use_sign)
	{
		for (j=1; j<alen; j++)
		{
			if (avec[j]==avec[j-1])
				continue ;
			int idx = CMath::fast_find(dictionary, dictionary_size, avec[j-1]) ;
			assert(idx!=-1) ;
			dictionary_weights[idx] += weight/sqrtdiag_lhs[vec_idx] ;
		}
		int idx = CMath::fast_find(dictionary, dictionary_size, avec[alen-1]) ;
		assert(idx!=-1) ;
		dictionary_weights[idx] += weight/sqrtdiag_lhs[vec_idx] ;
	}
	else
	{
		for (j=1; j<alen; j++)
		{
			if (avec[j]==avec[j-1])
				continue ;
			int idx = CMath::fast_find(dictionary, dictionary_size, avec[j-1]) ;
			assert(idx!=-1) ;
			dictionary_weights[idx] += weight*(j-last_j)/sqrtdiag_lhs[vec_idx] ;
			last_j = j ;
		}
		int idx = CMath::fast_find(dictionary, dictionary_size, avec[alen-1]) ;
		assert(idx!=-1) ;
		dictionary_weights[idx] += weight*(alen-last_j)/sqrtdiag_lhs[vec_idx] ;
	}
	((CWordFeatures*) lhs)->free_feature_vector(avec, vec_idx, afree);
}

void CCommWordKernel::clear_normal()
{
	for (int i=0; i<dictionary_size; i++)
		dictionary_weights[i]=0;
}


bool CCommWordKernel::init_optimization(INT count, INT *IDX, REAL * weights) 
{
	INT alen=-1 ;
	bool afree ;
	if (count<=0)
	{
		set_is_initialized(true) ;
		CIO::message(M_DEBUG, "empty set of SVs\n") ;
		return true ;
	} ;
	CIO::message(M_DEBUG, "initializing CCommWordKernel optimization\n") ;
	
	WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(0, alen, afree);
	if (avec==NULL)
		return false ;
	((CWordFeatures*) lhs)->free_feature_vector(avec, 0, afree);
	if (alen==-1) 
		return false ;
	WORD *words = new WORD[count*alen] ;
	if (words==NULL)
		return false ;

	int i ;
	int num_words = 0 ;
	for (i=0; i<count; i++)
	{
		WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(IDX[i], alen, afree);
		if (avec==NULL)
			return false ;
		int j;
		for (j=0; j<alen; j++)
			words[num_words++]=avec[j] ;
		((CWordFeatures*) lhs)->free_feature_vector(avec, IDX[i], afree); ;
	} ;
	CIO::message(M_DEBUG, "%i words\n", num_words) ;
	int num_unique_words = CMath::unique(words, num_words) ;
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
			CIO::progress(i, 0, count);

		WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(IDX[i], alen, afree);

		int j, last_j=0 ;
		if (use_sign)
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;
				int idx = CMath::fast_find(words, num_unique_words, avec[j-1]) ;
				assert(idx!=-1) ;
				word_weights[idx] += weights[i]/sqrtdiag_lhs[IDX[i]] ;
			}
			int idx = CMath::fast_find(words, num_unique_words, avec[alen-1]) ;
			assert(idx!=-1) ;
			word_weights[idx] += weights[i]/sqrtdiag_lhs[IDX[i]] ;
		}
		else
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;
				int idx = CMath::fast_find(words, num_unique_words, avec[j-1]) ;
				assert(idx!=-1) ;
				word_weights[idx] += weights[i]*(j-last_j)/sqrtdiag_lhs[IDX[i]] ;
				last_j = j ;
			}
			int idx = CMath::fast_find(words, num_unique_words, avec[alen-1]) ;
			assert(idx!=-1) ;
			word_weights[idx] += weights[i]*(alen-last_j)/sqrtdiag_lhs[IDX[i]] ;
		}
		((CWordFeatures*) lhs)->free_feature_vector(avec, IDX[i], afree);
	}
	CIO::message(M_MESSAGEONLY, "Done.         \n") ;
	
	dictionary         = words ;
	dictionary_weights = word_weights ;
	dictionary_size    = num_unique_words ;
	
	set_is_initialized(true) ;
	return true ;
}

bool CCommWordKernel::delete_optimization() 
{
	CIO::message(M_DEBUG, "deleting CCommWordKernel optimization\n");
	delete[] dictionary;
	delete[] dictionary_weights;

	dictionary_size=0;
	dictionary=NULL;
	dictionary_weights=NULL;

	set_is_initialized(false);

	return true;
}

REAL CCommWordKernel::compute_optimized(INT i) 
{ 
	if (!get_is_initialized())
	{
		CIO::message(M_ERROR, "CCommWordKernel optimization not initialized\n") ;
		return 0 ; 
	}

	REAL result = 0 ;
	INT alen = -1 ;
	bool afree ;
	WORD* avec=((CWordFeatures*) rhs)->get_feature_vector(i, alen, afree);
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
	
	((CWordFeatures*) rhs)->free_feature_vector(avec, i, afree);

	return result/sqrtdiag_rhs[i] ;
}
