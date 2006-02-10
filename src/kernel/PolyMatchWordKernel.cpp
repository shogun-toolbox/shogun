#include "lib/common.h"
#include "lib/io.h"
#include "kernel/PolyMatchWordKernel.h"
#include "features/Features.h"
#include "features/WordFeatures.h"

CPolyMatchWordKernel::CPolyMatchWordKernel(LONG size, INT d, bool inhom, bool use_norm)
  : CWordKernel(size),degree(d),inhomogene(inhom),
	sqrtdiag_lhs(NULL), sqrtdiag_rhs(NULL), initialized(false), use_normalization(use_norm)
{
}

CPolyMatchWordKernel::~CPolyMatchWordKernel() 
{
	cleanup();
}
  
bool CPolyMatchWordKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CWordKernel::init(l,r,do_init);

	initialized = false ;
	INT i;

	if (sqrtdiag_lhs != sqrtdiag_rhs)
	  delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL ;
	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL ;

	if (use_normalization)
	{
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

		ASSERT(sqrtdiag_lhs);
		ASSERT(sqrtdiag_rhs);

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
	}

	this->lhs=(CWordFeatures*) l;
	this->rhs=(CWordFeatures*) r;

	initialized = true ;
	return result;
}

void CPolyMatchWordKernel::cleanup()
{
	if (sqrtdiag_lhs != sqrtdiag_rhs)
		delete[] sqrtdiag_rhs;
	sqrtdiag_rhs=NULL;

	delete[] sqrtdiag_lhs;
	sqrtdiag_lhs=NULL;

	initialized=false;
}

bool CPolyMatchWordKernel::load_init(FILE* src)
{
	return false;
}

bool CPolyMatchWordKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CPolyMatchWordKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  ASSERT(alen==blen);

  REAL sqrt_a= 1 ;
  REAL sqrt_b= 1 ;

  if (initialized && use_normalization)
    {
      sqrt_a=sqrtdiag_lhs[idx_a] ;
      sqrt_b=sqrtdiag_rhs[idx_b] ;
    } ;

  REAL sqrt_both=sqrt_a*sqrt_b;

  INT ialen=(int) alen;

  INT sum=0;
  {
    for (INT i=0; i<ialen; i++)
	{
	  sum+= (avec[i]==bvec[i]) ? 1 : 0;
	}

  }

  if (inhomogene)
	  sum+=1;

  REAL result=sum;

  for (INT j=1; j<degree; j++)
	  result*=sum;

  result/=sqrt_both;

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
