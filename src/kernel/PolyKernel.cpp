#include "lib/common.h"
#include "kernel/PolyKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
//#include "lib/Time.h"
//#include "lib/f77blas.h"

#include <assert.h>

CPolyKernel::CPolyKernel(long size, int d, bool inhom)
  : CRealKernel(size),degree(d),inhomogene(inhom),scale(1.0)
{
}

CPolyKernel::~CPolyKernel() 
{
}
  
void CPolyKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CRealKernel::init((CRealFeatures*) l, (CRealFeatures*) r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message("rescaling kernel by %g (num:%d)\n",scale, math.min(l->get_num_vectors(), r->get_num_vectors()));
}

void CPolyKernel::init_rescale()
{
//	CTime t;
//
//	long maxx=math.min(5000l,lhs->get_num_vectors());
//	long maxy=math.min(5000l,rhs->get_num_vectors());
//
//	for (long x=0; x<maxx; x++)
//	{
//		for (long y=0; y<maxy; y++)
//		{
//			compute(x, y);
//		}
//	}
//	t.cur_time_diff_sec(true);

	double sum=0;
	scale=1.0;
	for (long i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	scale=sum/math.min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CPolyKernel::cleanup()
{
}

bool CPolyKernel::load_init(FILE* src)
{
	return false;
}

bool CPolyKernel::save_init(FILE* dest)
{
	return false;
}
  
bool CPolyKernel::check_features(CFeatures* f) 
{
  return (f->get_feature_type()==F_REAL);
}

REAL CPolyKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  double* avec=((CRealFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  
  assert(alen==blen);

  int skip=1;
  int ialen=(int) alen;

#ifdef NO_LAPACK
  REAL result=0;
  {
    for (int i=0; i<ialen; i++)
      result+=avec[i]*bvec[i];
  }
#else
  REAL result=ddot_(&ialen, avec, &skip, bvec, &skip);
#endif // NO_LAPACK

  if (inhomogene)
	  result+=1;

  REAL re=result/scale;
  result=re;
  
  for (int j=0; j<degree; j++)
	  result*=re;

//  result=pow(result,(double) degree)/scale;

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
