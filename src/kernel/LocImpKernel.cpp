#include "lib/common.h"
#include "kernel/LocImpKernel.h"

CLocImpKernel::CLocImpKernel(int width_, int degree1_, 
			     int degree2_, int degree3_) 
  : CKernel(), width(width_), degree1(degree1_), degree2(degree2_),
    degree3(degree3_)
{
} ;

CLocImpKernel::~CLocImpKernel() 
{
}
  
void CLocImpKernel::init(CFeatures* f)
{
} ;

void CLocImpKernel::cleanup()
{
} ;
  
bool CLocImpKernel::check_features(CFeatures* f) 
{
  return (f->get_feature_type()==CFeatures::SHORT) ;
} ;

REAL CLocImpKernel::compute(CFeatures* a, int idx_a, CFeatures* b, int idx_b)
{
  
}

