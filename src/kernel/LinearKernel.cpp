#include "lib/common.h"
#include "kernel/LinearKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

#include <assert.h>

CLinearKernel::CLinearKernel(bool rescale_) 
  : CKernel(),rescale(rescale_),scale(1.0)
{
}

CLinearKernel::~CLinearKernel() 
{
}
  
void CLinearKernel::init(CFeatures* l, CFeatures* r)
{
	CKernel::init(l,r); 

	if (rescale)
		init_rescale(l) ;
}

void CLinearKernel::init_rescale(CFeatures* f)
{
	double sum=0;
	scale=1.0;

	for (long i=0; i<f->get_num_vectors(); i++)
		sum+=compute(i, i);

	scale=sum/f->get_num_vectors();
}

void CLinearKernel::cleanup()
{
}
  
bool CLinearKernel::check_features(CFeatures* f) 
{
  return (f->get_feature_type()==F_REAL);
}

REAL CLinearKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  assert(alen==blen);
  //fprintf(stderr, "LinKernel.compute(%ld,%ld) %d\n", idx_a, idx_b, alen) ;

//  double sum=0;
//  for (long i=0; i<alen; i++)
//	  sum+=avec[i]*bvec[i];

//  CIO::message("%ld,%ld -> %f\n",idx_a, idx_b, sum);

  int skip=1;
  int ialen=(int) alen;
  //REAL result=F77CALL(ddot)(REF ialen, avec, REF skip, bvec, REF skip)/scale;

  REAL result=ddot_(&ialen, avec, &skip, bvec, &skip)/scale;
//  REAL result=sum/scale;
  ((CRealFeatures*) lhs)->free_feature_vector(avec, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, bfree);

  return result;
}

