#include "lib/common.h"
#include "kernel/SparsePolyKernel.h"
#include "features/Features.h"
#include "features/SparseRealFeatures.h"
#include "features/SparseFeatures.h"
#include "kernel/SparseKernel.h"
#include "lib/io.h"

#include <assert.h>

CSparsePolyKernel::CSparsePolyKernel(long size, int d, bool inhom)
  : CSparseRealKernel(size),degree(d),inhomogene(inhom)
{
}

CSparsePolyKernel::~CSparsePolyKernel() 
{
}
  
void CSparsePolyKernel::cleanup()
{
}

bool CSparsePolyKernel::load_init(FILE* src)
{
	return false;
}

bool CSparsePolyKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CSparsePolyKernel::compute(long idx_a, long idx_b)
{
  long alen=0;
  long blen=0;
  bool afree=false;
  bool bfree=false;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  TSparseEntry<REAL>* avec=((CSparseRealFeatures*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<REAL>* bvec=((CSparseRealFeatures*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);
  
  REAL result=0;
  REAL anormalize=0;
  REAL bnormalize=0;

  //result remains zero when one of the vectors is non existent
  if (avec && bvec)
  {
	  long i;
	  for (i=0; i<alen; i++)
		  anormalize+= avec[i].entry*avec[i].entry;

	  for (i=0; i<blen; i++)
		  bnormalize+= bvec[i].entry*bvec[i].entry;


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
	    		  result+= avec[i].entry * bvec[j].entry;
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
	    		  result+= bvec[i].entry * avec[j].entry;
	    		  j++;
	    	  }
	      }
	  }

	  if (inhomogene)
		  result+=1;

	  REAL re=result;
	  REAL ano=anormalize;
	  REAL bno=bnormalize;

	  for (int j=1; j<degree; j++)
	  {
		  result*=re;
		  ano*=anormalize;
		  bno*=bnormalize;
	  }

	  result/=sqrt(ano*bno);
  }
  else
  {
	  if (inhomogene)
		  result=1.0;
  }

  ((CSparseRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
