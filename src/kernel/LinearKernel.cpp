#include "lib/common.h"
#include "kernel/LinearKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

#include <assert.h>

CLinearKernel::CLinearKernel(bool rescale_) 
  : CKernel(),rescale(rescale_),scale(1.0)
{
}

CLinearKernel::~CLinearKernel() 
{
}
  
void CLinearKernel::init(CFeatures* f)
{
  if (rescale)
    init_rescale(f) ;
}

void CLinearKernel::init_rescale(CFeatures* f)
{
	double sum=0;
	scale=1.0;

	for (long i=0; i<f->get_number_of_examples(); i++)
		sum+=compute(f, i, f, i);

	scale=sum/f->get_number_of_examples();
}

void CLinearKernel::cleanup()
{
}
  
bool CLinearKernel::check_features(CFeatures* f) 
{
  return (f->get_feature_type()==F_REAL);
}

REAL CLinearKernel::compute(CFeatures* a, long idx_a, CFeatures* b, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  double* avec=((CRealFeatures*) a)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) b)->get_feature_vector(idx_b, blen, bfree);
  
  assert(alen==blen);
  //fprintf(stderr, "LinKernel.compute(%ld,%ld) %d\n", idx_a, idx_b, alen) ;

  //double sum=0;
  //for (long i=0; i<alen; i++)
//	  sum+=avec[i]*bvec[i];

//  CIO::message("%ld,%ld -> %f\n",idx_a, idx_b, sum);

  int skip=1;
  int ialen=(int) alen;
  //REAL result=F77CALL(ddot)(REF ialen, avec, REF skip, bvec, REF skip)/scale;

  REAL result=ddot_(&ialen, avec, &skip, bvec, &skip)/scale;
  ((CRealFeatures*) a)->free_feature_vector(avec, afree);
  ((CRealFeatures*) b)->free_feature_vector(bvec, bfree);

  return result;
}

