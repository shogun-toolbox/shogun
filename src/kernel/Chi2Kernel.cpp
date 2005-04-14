#include "lib/common.h"
#include "kernel/Chi2Kernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

#include <assert.h>

CChi2Kernel::CChi2Kernel(LONG size)
  : CRealKernel(size)
{
}

CChi2Kernel::~CChi2Kernel() 
{
	cleanup();
}
  
bool CChi2Kernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CRealKernel::init(l,r,do_init);
	initialized = true ;
	return result;
}

void CChi2Kernel::cleanup()
{
	initialized = false ;
}

bool CChi2Kernel::load_init(FILE* src)
{
	return false;
}

bool CChi2Kernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CChi2Kernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;

	double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	assert(alen==blen);

	REAL result=0;
	for (INT i=0; i<alen; i++)
	{
		REAL n=avec[i]-bvec[i];
		REAL d=avec[i]+bvec[i];
		result+=n*n/d;
	}

	((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
