#include "lib/common.h"
#include "kernel/WeightedDegreeKernel.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWeightedDegreeKernel::CWeightedDegreeKernel(long size, double* w, int d)
  : CKernel(size),weights(NULL),degree(d)
{
	weights=new REAL[d];
	assert(weights!=NULL);
	for (int i=0; i<d; i++)
		weights[i]=w[i];
}

CWeightedDegreeKernel::~CWeightedDegreeKernel() 
{
	cleanup();
}
  
void CWeightedDegreeKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CKernel::init(l,r, do_init); 
}

void CWeightedDegreeKernel::cleanup()
{
	delete[] weights;
	weights=NULL;
}

bool CWeightedDegreeKernel::load_init(FILE* src)
{
    assert(src!=NULL);
    unsigned int intlen=0;
    unsigned int endian=0;
    unsigned int fourcc=0;
    unsigned int doublelen=0;
	double* w=NULL;
    int d=1;

    assert(fread(&intlen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&doublelen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&endian, (unsigned int) intlen, 1, src)== 1);
    assert(fread(&fourcc, (unsigned int) intlen, 1, src)==1);
    assert(fread(&d, (unsigned int) intlen, 1, src)==1);
	double* weights= new double[d];
    assert(fread(w, sizeof(double), d, src)==(unsigned int) d) ;

	for (int i=0; i<d; i++)
		weights[i]=w[i];

    CIO::message("detected: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, d);

	degree=d;
	return true;
}

bool CWeightedDegreeKernel::save_init(FILE* dest)
{
    unsigned char intlen=sizeof(unsigned int);
    unsigned char doublelen=sizeof(double);
    unsigned int endian=0x12345678;
    unsigned int fourcc='FDGK'; //id for fixed degree kernel
	double* w= new double[degree];

	for (int i=0; i<degree; i++)
		w[i]=weights[i];

    assert(fwrite(&intlen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&doublelen, sizeof(unsigned char), 1, dest)==1);
    assert(fwrite(&endian, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&fourcc, sizeof(unsigned int), 1, dest)==1);
    assert(fwrite(&degree, sizeof(int), 1, dest)==1);
    assert(fwrite(w, sizeof(double), degree, dest)==(unsigned int) degree) ;

    CIO::message("wrote: intsize=%d, doublesize=%d, degree=%d\n", intlen, doublelen, degree);

	return true;
}
  
REAL CWeightedDegreeKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;

  CHAR* avec=((CStringFeatures*) lhs)->get_feature_vector(idx_a, alen);
  CHAR* bvec=((CStringFeatures*) rhs)->get_feature_vector(idx_b, blen);

  // can only deal with strings of same length
  assert(alen==blen);

  double sum=0;

  for (int i=0; i<alen-degree; i++)
  {
	  bool match=true;

	  for (int j=0; j<degree && match; j++)
	  {
		  match= avec[i+j]==bvec[i+j];

		  if (match)
			  sum+=weights[j];
	  }

  }

  return (double) sum;
}
