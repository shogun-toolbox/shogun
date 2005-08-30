#include "lib/common.h"
#include "kernel/SparseLinearKernel.h"
#include "features/Features.h"
#include "features/SparseRealFeatures.h"
#include "features/SparseFeatures.h"
#include "kernel/SparseKernel.h"
#include "lib/io.h"

#include <assert.h>


CSparseLinearKernel::CSparseLinearKernel(LONG size, bool do_rescale_, REAL scale_)
  : CSparseRealKernel(size),scale(scale_),do_rescale(do_rescale_), normal_length(0), normal(NULL)
{
	properties |= KP_LINADD | KP_KERNCOMBINATION ;
}

CSparseLinearKernel::~CSparseLinearKernel() 
{
	cleanup();
}
  
bool CSparseLinearKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CSparseRealKernel::init(l, r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message(M_INFO, "rescaling kernel by %g (num:%d)\n",scale, CMath::min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CSparseLinearKernel::init_rescale()
{
	if (!do_rescale)
		return ;
	double sum=0;
	scale=1.0;
	for (INT i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	scale=sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CSparseLinearKernel::cleanup()
{
	delete_optimization();
}

bool CSparseLinearKernel::load_init(FILE* src)
{
	return false;
}

bool CSparseLinearKernel::save_init(FILE* dest)
{
	return false;
}

void CSparseLinearKernel::clear_normal()
{
	assert(normal_length);

	if (normal==NULL)
	{
		normal = new REAL[normal_length];
	}

	for (int i=0; i<normal_length; i++)
		normal[i] = 0;
}

void CSparseLinearKernel::add_to_normal(INT idx, REAL weight) 
{
	//INT vlen;
	//bool vfree;
	//double* vec=((CRealFeatures*) lhs)->get_feature_vector(idx, vlen, vfree);

	//for (int i=0; i<vlen; i++)
	//	normal[i]+= weight*vec[i];

	//((CRealFeatures*) lhs)->free_feature_vector(vec, idx, vfree);
}
  
REAL CSparseLinearKernel::compute(INT idx_a, INT idx_b)
{
  INT alen=0;
  INT blen=0;
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
	      INT j=0;
	      for (INT i=0; i<alen; i++)
	      {
	    	  INT a_feat_idx=avec[i].feat_index;

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
	      INT j=0;
	      for (INT i=0; i<blen; i++)
	      {
	    	  INT b_feat_idx=bvec[i].feat_index;

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

bool CSparseLinearKernel::init_optimization(INT num_suppvec, INT* sv_idx, REAL* alphas) 
{
	//INT alen;
	//bool afree;
	//int i;

	//int num_feat=((CRealFeatures*) lhs)->get_num_features();
	//assert(num_feat);

	//normal=new REAL[num_feat];
	//assert(normal);

	//for (i=0; i<num_feat; i++)
	//	normal[i]=0;

	//for (int i=0; i<num_suppvec; i++)
	//{
	//	REAL* avec=((CRealFeatures*) lhs)->get_feature_vector(sv_idx[i], alen, afree);
	//	assert(avec);

	//	for (int j=0; j<num_feat; j++)
	//		normal[j]+=alphas[i]*avec[j];

	//	((CRealFeatures*) lhs)->free_feature_vector(avec, 0, afree);
	//}

	//set_is_initialized(true);
	return false;
}

bool CSparseLinearKernel::delete_optimization()
{
	//delete[] normal;
	//normal=NULL;
	//set_is_initialized(false);

	//return true;
	return false;
}

REAL CSparseLinearKernel::compute_optimized(INT idx_b) 
{
	/*
	INT blen;
	bool bfree;

	double* bvec=((CRealFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

	INT ialen=(int) blen;

#ifndef HAVE_ATLAS
	REAL result=0;
	{
		for (INT i=0; i<ialen; i++)
			result+=normal[i]*bvec[i];
	}
	result/=scale;
#else
	INT skip=1;
	REAL result = cblas_ddot(ialen, normal, skip, bvec, skip)/scale;
#endif

	((CRealFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
	*/
	CIO::message(M_ERROR, "not impl.\n");
	return 0;
}
