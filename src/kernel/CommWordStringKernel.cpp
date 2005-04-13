#include "lib/common.h"
#include "kernel/CommWordStringKernel.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCommWordStringKernel::CCommWordStringKernel(LONG size, bool use_sign_, 
											 E_NormalizationType normalization_) 
  : CStringKernel<WORD>(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	dictionary_size(0), dictionary_weights(NULL), use_sign(use_sign_), 
	normalization(normalization_)
{
	properties |= KP_LINADD;
	dictionary_size= 1<<(sizeof(WORD)*8);
	dictionary_weights = new REAL[dictionary_size];
	CIO::message(M_DEBUG, "using dictionary of %d bytes\n", dictionary_size);
	clear_normal();
}

CCommWordStringKernel::~CCommWordStringKernel() 
{
	cleanup();
	delete[] dictionary_weights;
}
  
void CCommWordStringKernel::remove_lhs() 
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
	delete_optimization();

	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;

	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	initialized=false;
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

void CCommWordStringKernel::add_to_normal(INT vec_idx, REAL weight)
{
	int j, last_j=0 ;
	INT alen=-1;
	WORD* avec=((CStringFeatures<WORD>*) lhs)->get_feature_vector(vec_idx, alen);

	if (avec && alen>0)
	{
		if (use_sign)
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;

				dictionary_weights[(int) avec[j-1]] += normalize_weight(weight, vec_idx, alen, normalization);
			}

			dictionary_weights[(int) avec[alen-1]] += normalize_weight(weight, vec_idx, alen, normalization);
		}
		else
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;

				dictionary_weights[(int) avec[j-1]] += normalize_weight(weight*(j-last_j), vec_idx, alen, normalization);
				last_j = j ;
			}

			dictionary_weights[(int) avec[alen-1]] += normalize_weight(weight*(alen-last_j), vec_idx, alen, normalization);

		}
	}

	set_is_initialized(true);
}

void CCommWordStringKernel::clear_normal()
{
	memset(dictionary_weights, 0, dictionary_size*sizeof(REAL));
	set_is_initialized(false);
}

bool CCommWordStringKernel::init_optimization(INT count, INT *IDX, REAL * weights) 
{
	if (count<=0)
	{
		set_is_initialized(true) ;
		CIO::message(M_DEBUG, "empty set of SVs\n") ;
		return true ;
	} ;

	CIO::message(M_DEBUG, "initializing CCommWordStringKernel optimization\n") ;

	for (int i=0; i<count; i++)
	{
		if ( (i % (count/10+1)) == 0)
			CIO::progress(i, 0, count);

		add_to_normal(IDX[i], weights[i]);
	}

	CIO::message(M_MESSAGEONLY, "Done.         \n") ;
	
	set_is_initialized(true) ;
	return true ;
}

bool CCommWordStringKernel::delete_optimization() 
{
	CIO::message(M_DEBUG, "deleting CCommWordStringKernel optimization\n");

	clear_normal();
	return true;
}

REAL CCommWordStringKernel::compute_optimized(INT i) 
{ 
	if (!get_is_initialized())
	{
		CIO::message(M_ERROR, "CCommWordStringKernel optimization not initialized\n") ;
		return 0 ; 
	}

	REAL result = 0 ;
	INT alen = -1 ;
	int j, last_j=0 ;
	WORD* avec=((CStringFeatures<WORD>*) rhs)->get_feature_vector(i, alen);

	if (avec && alen>0)
	{
		if (use_sign)
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;
				result += dictionary_weights[(int) avec[j-1]] ;
			}
			result += dictionary_weights[(int) avec[alen-1]] ;
		}
		else
		{
			for (j=1; j<alen; j++)
			{
				if (avec[j]==avec[j-1])
					continue ;

				result += dictionary_weights[(int) avec[j-1]]*(j-last_j) ;
				last_j = j ;
			}

			result += dictionary_weights[(int) avec[alen-1]]*(alen-last_j) ;
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
	}
	return result ;
}
