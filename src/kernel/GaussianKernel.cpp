#include "lib/common.h"
#include "kernel/GaussianKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"
//#include "lib/Time.h"

#include <assert.h>

CGaussianKernel::CGaussianKernel(long size, double w, bool rescale_) 
  : CRealKernel(size),width(w),rescale(rescale_),scale(1.0),vec(NULL),vec_len(0)
{
}

CGaussianKernel::~CGaussianKernel() 
{
}
  
void CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	CRealKernel::init((CRealFeatures*) l, (CRealFeatures*) r); 

	long l_len=((CRealFeatures*) l)->get_num_features();
	long r_len=((CRealFeatures*) r)->get_num_features();

	assert(l_len==r_len);
	CIO::message("%ld %ld\n",l_len, r_len);
	delete[] vec;

	vec_len=l_len;
	vec=new REAL[vec_len];

	assert(vec);

	if (rescale)
		init_rescale() ;
}

void CGaussianKernel::init_rescale()
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
	CIO::message("rescaling kernel by %g (sum:%g num:%d)\n",scale, sum, math.min(lhs->get_num_vectors(), rhs->get_num_vectors()));
}

void CGaussianKernel::cleanup()
{
	delete[] vec;
	vec=NULL;
	vec_len=0;
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
  assert(alen==vec_len);
  int ialen=(int) alen;

  REAL result=0;
  for (int i=0; i<ialen; i++)
	  result+=(avec[i]-bvec[i])*(avec[i]-bvec[i]);

  result=exp(-result/width)/scale;

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
