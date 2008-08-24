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
#include "lib/io.h"
#include "kernel/SparsePolyKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/SparseFeatures.h"

CSparsePolyKernel::CSparsePolyKernel(INT size, INT d, bool i)
: CSparseKernel<DREAL>(size), degree(d), inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CSparsePolyKernel::CSparsePolyKernel(
	CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r, INT size, INT d, bool i)
: CSparseKernel<DREAL>(size),degree(d),inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l,r);
}

CSparsePolyKernel::~CSparsePolyKernel()
{
	cleanup();
}

bool CSparsePolyKernel::init(CFeatures* l, CFeatures* r)
{
	CSparseKernel<DREAL>::init(l,r);
	return init_normalizer();
}
  
void CSparsePolyKernel::cleanup()
{
	CKernel::cleanup();
}

bool CSparsePolyKernel::load_init(FILE* src)
{
	return false;
}

bool CSparsePolyKernel::save_init(FILE* dest)
{
	return false;
}
  
DREAL CSparsePolyKernel::compute(INT idx_a, INT idx_b)
{
  INT alen=0;
  INT blen=0;
  bool afree=false;
  bool bfree=false;

  TSparseEntry<DREAL>* avec=((CSparseFeatures<DREAL>*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<DREAL>* bvec=((CSparseFeatures<DREAL>*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);

  DREAL result=((CSparseFeatures<DREAL>*) lhs)->sparse_dot(1.0,avec, alen, bvec, blen);

  if (inhomogene)
	  result+=1;

  result=CMath::pow(result, degree);

  ((CSparseFeatures<DREAL>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseFeatures<DREAL>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
