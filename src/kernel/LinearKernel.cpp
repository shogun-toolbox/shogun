#include "lib/config.h"

#ifdef HAVE_ATLAS
extern "C" {
#include <atlas_level1.h>
}
#endif

#include "lib/common.h"
#include "kernel/LinearKernel.h"
#include "features/Features.h"
#include "features/RealFeatures.h"
#include "lib/io.h"

#include <assert.h>

CLinearKernel::CLinearKernel(LONG size)
  : CRealKernel(size),scale(1.0)
{
}

CLinearKernel::~CLinearKernel() 
{
}
  
bool CLinearKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CRealKernel::init(l, r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message(M_INFO, "rescaling kernel by %g (num:%d)\n",scale, math.min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CLinearKernel::init_rescale()
{
	double sum=0;
	scale=1.0;
	for (INT i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	scale=sum/math.min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CLinearKernel::cleanup()
{
}

bool CLinearKernel::load_init(FILE* src)
{
    assert(src!=NULL);
    UINT intlen=0;
    UINT endian=0;
    UINT fourcc=0;
    UINT doublelen=0;
    double s=1;

    assert(fread(&intlen, sizeof(BYTE), 1, src)==1);
    assert(fread(&doublelen, sizeof(BYTE), 1, src)==1);
    assert(fread(&endian, (UINT) intlen, 1, src)== 1);
    assert(fread(&fourcc, (UINT) intlen, 1, src)==1);
    assert(fread(&s, (UINT) doublelen, 1, src)==1);
    CIO::message(M_INFO, "detected: intsize=%d, doublesize=%d, scale=%g\n", intlen, doublelen, s);

	scale=s;
	return true;
}

bool CLinearKernel::save_init(FILE* dest)
{
    BYTE intlen=sizeof(UINT);
    BYTE doublelen=sizeof(double);
    UINT endian=0x12345678;
    BYTE fourcc[5]="LINK"; //id for linear kernel

    assert(fwrite(&intlen, sizeof(BYTE), 1, dest)==1);
    assert(fwrite(&doublelen, sizeof(BYTE), 1, dest)==1);
    assert(fwrite(&endian, sizeof(UINT), 1, dest)==1);
    assert(fwrite(&fourcc, sizeof(UINT), 1, dest)==1);
    assert(fwrite(&scale, sizeof(double), 1, dest)==1);
    CIO::message(M_INFO, "wrote: intsize=%d, doublesize=%d, scale=%g\n", intlen, doublelen, scale);

	return true;
}
  
REAL CLinearKernel::compute(INT idx_a, INT idx_b)
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
  result/=scale;
#else
  INT skip=1;
  REAL result = ATL_ddot(ialen, avec, skip, bvec, skip)/scale;
#endif // HAVE_ATLAS

  ((CRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
