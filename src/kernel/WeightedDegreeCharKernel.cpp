#include "lib/common.h"
#include "kernel/WeightedDegreeCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CWeightedDegreeCharKernel::CWeightedDegreeCharKernel(long size, double* w, int d)
  : CCharKernel(size),weights(NULL),degree(d)
{
	weights=new REAL[d];
	assert(weights!=NULL);
	for (int i=0; i<d; i++)
		weights[i]=w[i];
}

CWeightedDegreeCharKernel::~CWeightedDegreeCharKernel() 
{
	cleanup();
}
  
void CWeightedDegreeCharKernel::cleanup()
{
	delete[] weights;
	weights=NULL;
}

bool CWeightedDegreeCharKernel::load_init(FILE* src)
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

bool CWeightedDegreeCharKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CWeightedDegreeCharKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

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

  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (double) sum;
}
