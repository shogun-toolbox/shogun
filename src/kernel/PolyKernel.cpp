#include "lib/common.h"
#include "kernel/PolyKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

#include <assert.h>

CPolyKernel::CPolyKernel(long size, int d, bool inhom)
  : CRealKernel(size),degree(d),inhomogene(inhom)
{
}

CPolyKernel::~CPolyKernel() 
{
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
  
REAL CPolyKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  assert(alen==blen);

  int skip=1;
  int ialen=(int) alen;

#ifdef NO_LAPACK
  REAL result=0;
  REAL anormalize=0;
  REAL bnormalize=0;
  {
    for (int i=0; i<ialen; i++)
	{
      result+=avec[i]*bvec[i];
	  anormalize+=avec[i]*avec[i];
	  bnormalize+=bvec[i]*bvec[i];
	}

  }
#else
  REAL result=ddot_(&ialen, avec, &skip, bvec, &skip);
  REAL anormalize=ddot_(&ialen, avec, &skip, avec, &skip);
  REAL bnormalize=ddot_(&ialen, bvec, &skip, bvec, &skip);
#endif // NO_LAPACK

  if (inhomogene)
  {
	  result+=1;
	  anormalize+=1;
	  bnormalize+=1;
  }

  REAL re=result;
  REAL ano=anormalize;
  REAL bno=bnormalize;

  for (int j=1; j<degree; j++)
  {
	  result*=re;
	  ano*=anormalize;
	  bno*=bnormalize;
  }

  result/=sqrt(ano*bno);

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
