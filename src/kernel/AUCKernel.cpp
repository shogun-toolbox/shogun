#include "lib/common.h"
#include "lib/Mathmatics.h"
#include "kernel/AUCKernel.h"
#include "kernel/WordKernel.h"
#include "features/WordFeatures.h"
#include "lib/io.h"

#include <assert.h>

CAUCKernel::CAUCKernel(INT size, CKernel * subkernel_)
	: CWordKernel(size),subkernel(subkernel_)
{
}

CAUCKernel::~CAUCKernel() 
{
	cleanup();
}
  
bool CAUCKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CWordKernel::init(l, r, do_init); 
	return true;
}


void CAUCKernel::cleanup()
{
}

bool CAUCKernel::load_init(FILE* src)
{
	return false;
}

bool CAUCKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CAUCKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  assert(alen==2);
  assert(blen==2);

  assert(subkernel!=NULL) ;
  REAL k11,k12,k21,k22 ;
  INT idx_a1=avec[0], idx_a2=avec[1], idx_b1=bvec[0], idx_b2=bvec[1] ;
  
  k11 = subkernel->kernel(idx_a1,idx_b1) ;
  k12 = subkernel->kernel(idx_a1,idx_b2) ;
  k21 = subkernel->kernel(idx_a2,idx_b1) ;  
  k22 = subkernel->kernel(idx_a2,idx_b2) ;

  REAL result = k11+k22-k21-k12 ;

  //CIO::message(M_DEBUG, "k(%i,%i)=%1.2f = k(%i,%i)+k(%i,%i)-k(%i,%i)-k(%i,%i)=%1.2f+%1.2f-%1.2f-%1.2f\n", idx_a, idx_b, result,idx_a1, idx_b1, idx_a1, idx_b2, idx_a2, idx_b1, idx_a2, idx_b2, k11, k22, k21, k12) ;
  
  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
