#include "lib/common.h"
#include "kernel/SparseLinearKernel.h"
#include "features/Features.h"
#include "features/SparseRealFeatures.h"
#include "features/SparseFeatures.h"
#include "kernel/SparseKernel.h"
#include "lib/io.h"

#include <assert.h>


CSparseLinearKernel::CSparseLinearKernel(long size)
  : CSparseRealKernel(size),scale(1.0)
{
}

CSparseLinearKernel::~CSparseLinearKernel() 
{
}
  
bool CSparseLinearKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CSparseRealKernel::init(l, r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message("rescaling kernel by %g (num:%d)\n",scale, math.min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CSparseLinearKernel::init_rescale()
{
	double sum=0;
	scale=1.0;
	for (long i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	scale=sum/math.min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CSparseLinearKernel::cleanup()
{
}

bool CSparseLinearKernel::load_init(FILE* src)
{
    assert(src!=NULL);
    unsigned int intlen=0;
    unsigned int endian=0;
    unsigned int fourcc=0;
    unsigned int doublelen=0;
    double s=1;

    assert(fread(&intlen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&doublelen, sizeof(unsigned char), 1, src)==1);
    assert(fread(&endian, (unsigned int) intlen, 1, src)== 1);
    assert(fread(&fourcc, (unsigned int) intlen, 1, src)==1);
    assert(fread(&s, (unsigned int) doublelen, 1, src)==1);
    CIO::message("detected: intsize=%d, doublesize=%d, scale=%g\n", intlen, doublelen, s);

	scale=s;
	return true;
}

bool CSparseLinearKernel::save_init(FILE* dest)
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
  
REAL CSparseLinearKernel::compute(long idx_a, long idx_b)
{
  long alen=0;
  long blen=0;
  bool afree=false;
  bool bfree=false;

  //fprintf(stderr, "LinKernel.compute(%ld,%ld)\n", idx_a, idx_b) ;
  TSparseEntry<REAL>* avec=((CSparseRealFeatures*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<REAL>* bvec=((CSparseRealFeatures*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);
  
  REAL result=0;

  //result remains zero when one of the vectors is non existent
  if (avec && bvec)
  {
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

	  result/=scale;
  }

  ((CSparseRealFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
