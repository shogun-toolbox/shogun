/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/Mathematics.h"
#include "lib/io.h"
#include "kernel/WordMatchKernel.h"
#include "features/WordFeatures.h"

CWordMatchKernel::CWordMatchKernel(INT size, INT d, bool dr, DREAL s)
: CSimpleKernel<WORD>(size),scale(s),do_rescale(dr), initialized(false),
	degree(d)
{
}

CWordMatchKernel::CWordMatchKernel(
	CWordFeatures* l, CWordFeatures* r, INT d, bool dr, DREAL s)
: CSimpleKernel<WORD>(10), scale(s), do_rescale(dr), initialized(false),
	degree(d)
{
	init(l, r);
}

CWordMatchKernel::~CWordMatchKernel()
{
	cleanup();
}

bool CWordMatchKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<WORD>::init(l, r);

	if (!initialized)
		init_rescale() ;

	SG_INFO( "rescaling kernel by %g (num:%d)\n",scale, CMath::min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CWordMatchKernel::init_rescale()
{
	if (!do_rescale)
		return ;
	LONGREAL sum=0;
	scale=1.0;
	for (INT i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	if ( sum > (pow((double) 2, (double) 8*sizeof(LONG))) ) {
      SG_ERROR( "the sum %lf does not fit into integer of %d bits expect bogus results.\n", sum, 8*sizeof(LONG));
   }
	scale=sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
	initialized=true;
}

bool CWordMatchKernel::load_init(FILE* src)
{
	return false;
}

bool CWordMatchKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CWordMatchKernel::compute(INT idx_a, INT idx_b)
{
  INT alen, blen;
  bool afree, bfree;

  WORD* avec=((CWordFeatures*) lhs)->get_feature_vector(idx_a, alen, afree);
  WORD* bvec=((CWordFeatures*) rhs)->get_feature_vector(idx_b, blen, bfree);
  ASSERT(alen==blen);

  double sum=0;
  for (INT i=0; i<alen; i++)
	  sum+= (avec[i]==bvec[i]) ? 1 : 0;

  DREAL result=sum;

  for (INT j=1; j<degree; j++)
	  result*=sum;
  sum/=scale;

  ((CWordFeatures*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CWordFeatures*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
