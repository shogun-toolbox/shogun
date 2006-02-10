#include "lib/common.h"
#include "lib/io.h"
#include "features/Features.h"
#include "features/SparseRealFeatures.h"
#include "features/SparseFeatures.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparseKernel.h"

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
	int num = ((CSparseRealFeatures*) lhs)->get_num_features();
	if (normal==NULL)
		normal = new REAL[num] ;
	for (int i=0; i<num; i++)
		normal[i]=0;
}

void CSparseLinearKernel::add_to_normal(INT idx, REAL weight) 
{
	INT vlen;
	bool vfree;
	TSparseEntry<REAL>* vec=((CSparseRealFeatures*) rhs)->get_sparse_feature_vector(idx, vlen, vfree);

	for (int i=0; i<vlen; i++)
		normal[vec[i].feat_index]+= weight*vec[i].entry;

	((CSparseRealFeatures*) lhs)->free_feature_vector(vec, idx, vfree);
}
  
REAL CSparseLinearKernel::compute(INT idx_a, INT idx_b)
{
  INT alen=0;
  INT blen=0;
  bool afree=false;
  bool bfree=false;

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
	clear_normal();

	for (int i=0; i<num_suppvec; i++)
		add_to_normal(sv_idx[i], alphas[i]);

	set_is_initialized(true);
	return true;;
}

bool CSparseLinearKernel::delete_optimization()
{
	delete[] normal;
	normal=NULL;
	set_is_initialized(false);

	return true;
}

REAL CSparseLinearKernel::compute_optimized(INT idx) 
{
	INT vlen;
	bool vfree;

	TSparseEntry<REAL>* vec=((CSparseRealFeatures*) rhs)->get_sparse_feature_vector(idx, vlen, vfree);

	REAL result=0;
	for (INT i=0; i<vlen; i++)
		result+=normal[vec[i].feat_index]*vec[i].entry;
	result/=scale;

	((CSparseRealFeatures*) rhs)->free_sparse_feature_vector(vec, idx, vfree);

	return result;
}
