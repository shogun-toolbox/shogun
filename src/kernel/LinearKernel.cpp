#include "lib/common.h"
#include "kernel/LinearKernel.h"

CLinearKernel::CLinearKernel(bool rescale_) 
  : CKernel(),rescale(rescale_),scale(1.0)
{
} ;
CLinearKernel::~CLinearKernel() 
{
}
  
void CLinearKernel::init(CFeatures* f)
{
  if (rescale)
    init_scale(f) ;
} ;

void CLinearKernel::init_rescale(CFeatures* f)
{
  fprintf(stderr,"CLinearKernel::init_rescale not implemented yet\n") ;
} ;

void CLinearKernel::cleanup()
{
} ;
  
REAL CLinearKernel::compute(CFeatures* a, int idx_a, CFeatures* b, int idx_b)
{
  int alen, blen, afree, bfree ;
  REAL* avec=a->get_feature_vector(idx_a, alen, afree);
  REAL* avec=b->get_feature_vector(idx_b, blen, bfree);
  assert(alen==blen) ;

  REAL result=ddot_(alen, avec, 1, bvec, 1) ;
  a->free_feature_vector(avec, afree);
  b->free_feature_vector(bvec, bfree);

  return result ;

}

