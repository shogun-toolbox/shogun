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
  
}

