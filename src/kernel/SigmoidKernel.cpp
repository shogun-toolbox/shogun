#include "lib/config.h"

#ifdef HAVE_ATLAS
extern "C" {
#include <cblas.h>
}
#endif

#include "lib/common.h"
#include "kernel/SigmoidKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

#include <assert.h>

CSigmoidKernel::CSigmoidKernel(LONG size, REAL g, REAL c)
  : CRealKernel(size),gamma(g), coef0(c)
{
}

CSigmoidKernel::~CSigmoidKernel() 
{
	cleanup();
}
  
bool CSigmoidKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CRealKernel::init(l, r, do_init); 
	return true;
}

void CSigmoidKernel::cleanup()
{
}

bool CSigmoidKernel::load_init(FILE* src)
{
	return false;
}

bool CSigmoidKernel::save_init(FILE* dest)
{
	return false;
}

REAL CSigmoidKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  assert(alen==blen);

  INT ialen=(int) alen;

#ifndef HAVE_ATLAS
  REAL result=0;
  {
    for (INT i=0; i<ialen; i++)
      result+=avec[i]*bvec[i];
  }
#else
  INT skip=1;
  REAL result = cblas_ddot(ialen, avec, skip, bvec, skip);
#endif // HAVE_ATLAS

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return tanh(gamma*result+coef0);
}
