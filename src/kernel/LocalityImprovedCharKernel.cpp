#include "lib/common.h"
#include "kernel/LocalityImprovedCharKernel.h"
#include "features/Features.h"
#include "features/CharFeatures.h"
#include "lib/io.h"

#include <assert.h>

CLocalityImprovedCharKernel::CLocalityImprovedCharKernel(LONG size, INT l, INT d1, INT d2)
  : CCharKernel(size),length(l),inner_degree(d1),outer_degree(d2),match(NULL)
{
	CIO::message(M_INFO, "LIK with parms: l=%d, d1=%d, d2=%d created!\n", l, d1, d2);
}

CLocalityImprovedCharKernel::~CLocalityImprovedCharKernel() 
{
	cleanup();
}

bool CLocalityImprovedCharKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	bool result=CCharKernel::init(l,r,do_init);

	if (result)
		match=new CHAR[((CCharFeatures*) l)->get_num_features()];

	return (match!=NULL && result==true);
}
  
void CLocalityImprovedCharKernel::cleanup()
{
	delete[] match;
	match = NULL;
}

bool CLocalityImprovedCharKernel::load_init(FILE* src)
{
	return false;
}

bool CLocalityImprovedCharKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CLocalityImprovedCharKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  CHAR* avec=((CCharFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  CHAR* bvec=((CCharFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  // can only deal with strings of same length
  assert(alen==blen);

  INT i,j,t;

  // initialize match table 1 -> match;  0 -> no match
  for (i=0; i<alen; i++)
  {
	  if (avec[i]==bvec[i])
		  match[i]=1;
	  else
		  match[i]=0;
  }


  REAL outer_sum=0;

  for (t=0; t<alen-length; t++)
  {
	  INT sum=0;
	  for (i=0; i<length; i++)
		  sum+=(i+1)*match[t+i]+(length-i)*match[t+i+length+1];

	  //add middle element + normalize with sum_i=0^2l+1 i = (2l+1)(l+1)
	  REAL inner_sum= ((REAL) sum + (length+1)*match[t+length]) / ((2*length+1)*(length+1));
	  REAL s=inner_sum;

	  for (j=1; j<inner_degree; j++)
		  inner_sum*=s;

	  outer_sum+=inner_sum;
  }

  double result=outer_sum;

  for (i=1; i<outer_degree; i++)
	  result*=outer_sum;

  ((CCharFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CCharFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return (double) result;
}
