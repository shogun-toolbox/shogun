#include "lib/common.h"
#include "kernel/CommWordKernel.h"
#include "features/Features.h"
#include "features/WordFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCommWordKernel::CCommWordKernel(LONG size, bool use_sign_)
	: CWordKernel(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false),
	  use_sign(use_sign_)
{
	properties |= KP_LINADD;
	dictionary_size= 1<<(sizeof(WORD)*8);
	dictionary_weights = new REAL[dictionary_size];
	CIO::message(M_DEBUG, "using dictionary of %d bytes\n", dictionary_size);
	clear_normal();
}

CCommWordKernel::~CCommWordKernel() 
{
	cleanup();

	delete[] dictionary_weights;
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
	clear_normal();

	initialized=false;
	
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

			dictionary_weights[(int) avec[j-1]] += weight/sqrtdiag_lhs[vec_idx] ;
		}

		dictionary_weights[(int) avec[alen-1]] += weight/sqrtdiag_lhs[vec_idx] ;
	}
	else
	{
		for (j=1; j<alen; j++)
		{
			if (avec[j]==avec[j-1])
				continue ;

			dictionary_weights[(int) avec[j-1]] += weight*(j-last_j)/sqrtdiag_lhs[vec_idx] ;
			last_j = j ;
		}

		dictionary_weights[(int) avec[alen-1]] += weight*(alen-last_j)/sqrtdiag_lhs[vec_idx] ;
	}
	((CWordFeatures*) lhs)->free_feature_vector(avec, vec_idx, afree);

	set_is_initialized(true);
}

void CCommWordKernel::clear_normal()
{
	memset(dictionary_weights, 0, dictionary_size*sizeof(REAL));
	set_is_initialized(false);
}


bool CCommWordKernel::init_optimization(INT count, INT *IDX, REAL * weights) 
{
	if (count<=0)
	{
		set_is_initialized(true) ;
		CIO::message(M_DEBUG, "empty set of SVs\n") ;
		return true ;
	}


	CIO::message(M_DEBUG, "initializing CCommWordKernel optimization\n") ;
	
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

bool CCommWordKernel::delete_optimization() 
{
	CIO::message(M_DEBUG, "deleting CCommWordKernel optimization\n");

	clear_normal();
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
	
	((CWordFeatures*) rhs)->free_feature_vector(avec, i, afree);

	return result/sqrtdiag_rhs[i] ;
}
