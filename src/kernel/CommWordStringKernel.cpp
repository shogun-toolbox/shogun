#include "lib/common.h"
#include "kernel/CommWordStringKernel.h"
#include "features/StringFeatures.h"
#include "features/WordFeatures.h"
#include "lib/io.h"

#include <assert.h>

CCommWordStringKernel::CCommWordStringKernel(LONG size)
  : CWordKernel(size), sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false)
{
}

CCommWordStringKernel::~CCommWordStringKernel() 
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
}
  
bool CCommWordStringKernel::init(CFeatures* l, CFeatures* r, bool do_init)
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
		sqrtdiag_lhs[i]=sqrt(compute(i,i));

	// if lhs is different from rhs (train/test data)
	// compute also the normalization for rhs
	if (sqrtdiag_lhs!=sqrtdiag_rhs)
	{
		this->lhs=(CStringFeatures<WORD>*) r;
		this->rhs=(CStringFeatures<WORD>*) r;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		  sqrtdiag_rhs[i]=sqrt(compute(i,i));
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

  while (left_idx < alen && right_idx < blen)
  {
	  if (avec[left_idx]==bvec[right_idx])
	  {
		  INT old_left_idx=left_idx;
		  INT old_right_idx=right_idx;

		  WORD sym=avec[left_idx];

		  while (left_idx< alen && avec[left_idx]==sym)
			  left_idx++;

		  while (right_idx< blen && bvec[right_idx]==sym)
			  right_idx++;

		  result+=(left_idx-old_left_idx)*(right_idx-old_right_idx);
	  }
	  else if (avec[left_idx]<bvec[right_idx])
		  left_idx++;
	  else
		  right_idx++;
  }

  return result/sqrt_both;
}
