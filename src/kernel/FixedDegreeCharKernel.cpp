#include "lib/common.h"
#include "kernel/FixedDegreeCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CFixedDegreeCharKernel::CFixedDegreeCharKernel(long size, int d)
  : CCharKernel(size),degree(d)
{
}

CFixedDegreeCharKernel::~CFixedDegreeCharKernel() 
{
}
  
void CFixedDegreeCharKernel::cleanup()
{
}

bool CFixedDegreeCharKernel::load_init(FILE* src)
{
	return false;
}

bool CFixedDegreeCharKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CFixedDegreeCharKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen==blen);

  long sum=0;

  for (int i=0; i<alen-degree; i++)
  {
	  bool match=true;

	  for (int j=i; j<i+degree && match; j++)
	  {
		  match= avec[j]==bvec[j];
	  }

	  if (match)
		  sum++;
  }

  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (double) sum;
}
