#include "lib/common.h"
#include "kernel/LinearByteKernel.h"
#include "kernel/ByteKernel.h"
#include "features/ByteFeatures.h"
#include "lib/io.h"

#include <assert.h>
#include <math.h>

CLinearByteKernel::CLinearByteKernel(long size)
  : CByteKernel(size),scale(1.0)
{
}

CLinearByteKernel::~CLinearByteKernel() 
{
}
  
void CLinearByteKernel::init(CByteFeatures* l, CByteFeatures* r, bool do_init)
{
	CByteKernel::init(l, r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message("rescaling kernel by %g (num:%d)\n",scale, math.min(l->get_num_vectors(), r->get_num_vectors()));
}

void CLinearByteKernel::init_rescale()
{
	long double sum=0;
	scale=1.0;
	for (long i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	if ( sum > (pow((double) 2, (double) 8*sizeof(long))) )
		CIO::message("the sum %lf does not fit into integer of %d bits expect bogus results.\n", sum, 8*sizeof(long));
	scale=sum/math.min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CLinearByteKernel::cleanup()
{
}

bool CLinearByteKernel::load_init(FILE* src)
{
    assert(src!=NULL);
    unsigned int intlen=0;
    unsigned int endian=0;
    unsigned int fourcc=0;
    unsigned int r=0;
    unsigned int doublelen=0;
    double s=1;

    assert(fread(&intlen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&doublelen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&endian, (unsigned int) intlen, 1, src)== 1);
    assert(fread(&fourcc, (unsigned int) intlen, 1, src)==1);
    assert(fread(&r, (unsigned int) intlen, 1, src)==1);
    assert(fread(&s, (unsigned int) doublelen, 1, src)==1);
    CIO::message("detected: intsize=%d, doublesize=%d, r=%d, scale=%g\n", intlen, doublelen, r, s);
	scale=s;
	return true;
}

bool CLinearByteKernel::save_init(FILE* dest)
{
    unsigned char intlen=sizeof(unsigned int);
    unsigned char doublelen=sizeof(double);
    unsigned int endian=0x12345678;
    unsigned int fourcc='LINK'; //id for linear kernel

    assert(fwrite(&intlen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&doublelen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&endian, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&fourcc, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&scale, sizeof(double), 1, dest)==1);
    CIO::message("wrote: intsize=%d, doublesize=%d, scale=%g\n", intlen, doublelen, scale);

	return true;
}
  
REAL CLinearByteKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;
//
//  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  BYTE* avec=((CByteFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  BYTE* bvec=((CByteFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
//  
  assert(alen==blen);
//  //fprintf(stderr, "LinKernel.compute(%ld,%ld) %d\n", idx_a, idx_b, alen) ;
//
  double sum=0;
  for (long i=0; i<alen; i++)
	  sum+=((long) avec[i])*((long) bvec[i]);
//
////  CIO::message("%ld,%ld -> %f\n",idx_a, idx_b, sum);
//
//  int skip=1;
//  int ialen=(int) alen;
//  //REAL result=F77CALL(ddot)(REF ialen, avec, REF skip, bvec, REF skip)/scale;
//
//#ifdef NO_LAPACK
//  REAL result=0;
//  {
//    for (int i=0; i<ialen; i++)
//      result+=avec[i]*bvec[i];
//  }
//#else
//  REAL result=ddot_(&ialen, avec, &skip, bvec, &skip)/scale;
//#endif // NO_LAPACK
//
  REAL result=sum/scale;
//  ((CByteFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
//  ((CByteFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);
//
  return result;
}
