#include "lib/common.h"
#include "lib/Mathmatics.h"
#include "lib/io.h"
#include "kernel/WordMatchKernel.h"
#include "kernel/WordKernel.h"
#include "features/WordFeatures.h"

CWordMatchKernel::CWordMatchKernel(LONG size, INT d)
  : CWordKernel(size),scale(1.0),degree(d)
{
}

CWordMatchKernel::~CWordMatchKernel() 
{
	cleanup();
}
  
bool CWordMatchKernel::init(CFeatures* l, CFeatures* r, bool do_init)
{
	CWordKernel::init(l, r, do_init); 

	if (do_init)
		init_rescale() ;

	CIO::message(M_INFO, "rescaling kernel by %g (num:%d)\n",scale, CMath::min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CWordMatchKernel::init_rescale()
{
	LONGREAL sum=0;
	scale=1.0;
	for (INT i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	if ( sum > (pow((double) 2, (double) 8*sizeof(LONG))) )
		CIO::message(M_ERROR, "the sum %lf does not fit into integer of %d bits expect bogus results.\n", sum, 8*sizeof(LONG));
	scale=sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
}

void CWordMatchKernel::cleanup()
{
}

bool CWordMatchKernel::load_init(FILE* src)
{
	return false;
}

bool CWordMatchKernel::save_init(FILE* dest)
{
	return false;
}
  
REAL CWordMatchKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);

  ASSERT(alen==blen);
  double sum=0;
  for (INT i=0; i<alen; i++)
	  sum+= (avec[i]==bvec[i]) ? 1 : 0;

  REAL result=sum;

  for (INT j=1; j<degree; j++)
	  result*=sum;
  sum/=scale;

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
