#include "lib/common.h"
#include "kernel/FixedDegreeKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

#include <assert.h>

CFixedDegreeKernel::CFixedDegreeKernel(long size, int d)
  : CKernel(size),degree(d)
{
}

CFixedDegreeKernel::~CFixedDegreeKernel() 
{
}
  
void CFixedDegreeKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l,r, do_init); 
}

void CFixedDegreeKernel::cleanup()
{
}

bool CFixedDegreeKernel::load_init(FILE* src)
{
    assert(src!=NULL);
    unsigned int intlen=0;
    unsigned int endian=0;
    unsigned int fourcc=0;
    unsigned int doublelen=0;
    int d=1;

    assert(fread(&intlen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&doublelen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&endian, (unsigned int) intlen, 1, src)== 1);
    assert(fread(&fourcc, (unsigned int) intlen, 1, src)==1);
    assert(fread(&d, (unsigned int) intlen, 1, src)==1);
    CIO::message("detected: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, d);

	degree=d;
	return true;
}

bool CFixedDegreeKernel::save_init(FILE* dest)
{
    unsigned char intlen=sizeof(unsigned int);
    unsigned char doublelen=sizeof(double);
    unsigned int endian=0x12345678;
    unsigned int fourcc='FDGK'; //id for fixed degree kernel

    assert(fwrite(&intlen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&doublelen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&endian, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&fourcc, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&degree, sizeof(int), 1, dest)==1);
    CIO::message("wrote: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, degree);

	return true;
}
  
REAL CFixedDegreeKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;

  CHAR* avec=((CStringFeatures*) lhs)->get_feature_vector(idx_a, alen);
  CHAR* bvec=((CStringFeatures*) rhs)->get_feature_vector(idx_b, blen);

  // can only deal with strings of same length
  assert(alen==blen);

  long sum=0;

  for (int i=0; i<alen-degree; i++)
  {
	  bool match=true;

	  for (int j=i; j<i+degree && match; j++)
	  {
		  match= avec[j]==bvec[j];
	  }

	  if (match)
		  sum++;
  }

  return (double) sum;
}
