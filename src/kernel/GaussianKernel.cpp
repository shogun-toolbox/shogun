#include "lib/common.h"
#include "kernel/GaussianKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
//#include "lib/f77blas.h"

#include <assert.h>

CGaussianKernel::CGaussianKernel(long size, double w, bool rescale_) 
  : CRealKernel(size),width(w),rescale(rescale_),scale(1.0)
{
}

CGaussianKernel::~CGaussianKernel() 
{
}
  
void CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	CRealKernel::init((CRealFeatures*) l, (CRealFeatures*) r); 

	if (rescale)
		init_rescale() ;
}

void CGaussianKernel::init_rescale()
{
	double sum=0;
	scale=1.0;
	for (long i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	scale=sum/math.min(lhs->get_num_vectors(), rhs->get_num_vectors());
	CIO::message("rescaling kernel by %g (sum:%g num:%d)\n",scale, sum, math.min(lhs->get_num_vectors(), rhs->get_num_vectors()));
}

void CGaussianKernel::cleanup()
{
}

bool CGaussianKernel::load_init(FILE* src)
{
	return false;
}

bool CGaussianKernel::save_init(FILE* dest)
{
	return false;
}
  
bool CGaussianKernel::check_features(CFeatures* f) 
{
  return (f->get_feature_type()==F_REAL);
}

REAL CGaussianKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  assert(alen==blen);
  int ialen=(int) alen;

  REAL result=0;
  {
    for (int i=0; i<ialen; i++)
      result+=(avec[i]-bvec[i])*(avec[i]-bvec[i]);
  }

  result/=-width;
  result=exp(result)/scale;

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
