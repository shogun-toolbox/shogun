#include "lib/common.h"
#include "kernel/CommWordKernel.h"
#include "features/Features.h"
#include "features/WordFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCommWordKernel::CCommWordKernel(LONG size)
  : CWordKernel(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false)
{
}

CCommWordKernel::~CCommWordKernel() 
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
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
		sqrtdiag_lhs[i]=sqrt(compute(i,i));

	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if (sqrtdiag_lhs!=sqrtdiag_rhs)
	{
		this->lhs=(CWordFeatures*) r;
		this->rhs=(CWordFeatures*) r;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		  sqrtdiag_rhs[i]=sqrt(compute(i,i));
	}

	this->lhs=(CWordFeatures*) l;
	this->rhs=(CWordFeatures*) r;

	initialized = true ;
	return result;
}

void CCommWordKernel::cleanup()
{
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

  while (left_idx < alen && right_idx < alen)
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

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
  
  return result/sqrt_both;
}
