#include "lib/common.h"
#include "kernel/SparseGaussianKernel.h"
#include "features/Features.h"
#include "features/SparseRealFeatures.h"
#include "features/SparseFeatures.h"
#include "lib/io.h"

#include <assert.h>

CSparseGaussianKernel::CSparseGaussianKernel(long size, double w)
  : CSparseRealKernel(size),width(w)
{
}

CSparseGaussianKernel::~CSparseGaussianKernel() 
{
}
  
bool CSparseGaussianKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CSparseRealKernel::init(l, r, do_init); 

	return true;
}

void CSparseGaussianKernel::cleanup()
{
}

bool CSparseGaussianKernel::load_init(FILE* src)
{
	return false;
}

bool CSparseGaussianKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CSparseGaussianKernel::compute(long idx_a, long idx_b)
{
  long alen, blen;
  bool afree, bfree;

  TSparseEntry<REAL>* avec=((CSparseRealFeatures*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<REAL>* bvec=((CSparseRealFeatures*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);
  
  REAL result=0;

  long i;
  for (i=0; i<alen; i++)
	  result+= avec[i].entry * avec[i].entry;

  for (i=0; i<blen; i++)
	  result+= bvec[i].entry * bvec[i].entry;


  if (alen<=blen)
  {
	  long j=0;
	  for (long i=0; i<alen; i++)
	  {
		  int a_feat_idx=avec[i].feat_index;

		  while ( (j<blen) && (bvec[j].feat_index < a_feat_idx) )
			  j++;

		  if ( (j<blen) && (bvec[j].feat_index == a_feat_idx) )
		  {
			  result-= 2*(avec[i].entry*bvec[j].entry);
			  j++;
		  }
	  }
  }
  else
  {
	  long j=0;
	  for (long i=0; i<blen; i++)
	  {
		  int b_feat_idx=bvec[i].feat_index;

		  while ( (j<alen) && (avec[j].feat_index < b_feat_idx) )
			  j++;

		  if ( (j<alen) && (avec[j].feat_index == b_feat_idx) )
		  {
			  result-= 2*(bvec[i].entry*avec[j].entry);
			  j++;
		  }
	  }
  }
  result=exp(-result/width);

  ((CSparseRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

