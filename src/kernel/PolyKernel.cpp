#include "lib/config.h"

#ifdef HAVE_ATLAS
extern "C" {
#include <atlas_level1.h>
}
#endif

#include "lib/common.h"
#include "kernel/PolyKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

#include <assert.h>

CPolyKernel::CPolyKernel(LONG size, INT d, bool inhom)
  : CRealKernel(size),degree(d),inhomogene(inhom),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false)
{
}

CPolyKernel::~CPolyKernel() 
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	delete[] sqrtdiag_lhs;
}
  
bool CPolyKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CRealKernel::init(l,r,do_init);

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

	this->lhs=(CRealFeatures*) l;
	this->rhs=(CRealFeatures*) l;

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
		this->lhs=(CRealFeatures*) r;
		this->rhs=(CRealFeatures*) r;

		//compute normalize to 1 values
		for (i=0; i<rhs->get_num_vectors(); i++)
		{
			sqrtdiag_rhs[i]=sqrt(compute(i,i));

			//trap divide by zero exception
			if (sqrtdiag_rhs[i]==0)
				sqrtdiag_rhs[i]=1e-16;
		}
	}

	this->lhs=(CRealFeatures*) l;
	this->rhs=(CRealFeatures*) r;

	initialized = true ;
	return result;
}

void CPolyKernel::cleanup()
{
}

bool CPolyKernel::load_init(FILE* src)
{
	return false;
}

bool CPolyKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CPolyKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  assert(alen==blen);

  REAL sqrt_a= 1 ;
  REAL sqrt_b= 1 ;
  if (initialized)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;

  REAL sqrt_both=sqrt_a*sqrt_b;

  INT ialen=(int) alen;

#ifndef HAVE_ATLAS
  REAL result=0;
  {
    for (INT i=0; i<ialen; i++)
	{
      result+=avec[i]*bvec[i];
	}

  }
#else
  INT skip=1;
  REAL result=ATL_ddot(ialen, avec, skip, bvec, skip);
#endif // HAVE_ATLAS

  if (inhomogene)
	  result+=1;

  REAL re=result;

  for (INT j=1; j<degree; j++)
	  result*=re;

  result/=sqrt_both;

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
