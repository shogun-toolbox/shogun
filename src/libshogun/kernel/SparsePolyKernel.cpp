/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "kernel/SparsePolyKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/SparseFeatures.h"

CSparsePolyKernel::CSparsePolyKernel(int32_t size, int32_t d, bool i)
: CSparseKernel<float64_t>(size), degree(d), inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CSparsePolyKernel::CSparsePolyKernel(
	CSparseFeatures<float64_t>* l, CSparseFeatures<float64_t>* r,
	int32_t size, int32_t d, bool i)
: CSparseKernel<float64_t>(size),degree(d),inhomogene(i)
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
	CSparseKernel<float64_t>::init(l,r);
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

float64_t CSparsePolyKernel::compute(int32_t idx_a, int32_t idx_b)
{
  int32_t alen=0;
  int32_t blen=0;
  bool afree=false;
  bool bfree=false;

  TSparseEntry<float64_t>* avec=((CSparseFeatures<float64_t>*) lhs)->
  	get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<float64_t>* bvec=((CSparseFeatures<float64_t>*) rhs)->
  	get_sparse_feature_vector(idx_b, blen, bfree);

  float64_t result=((CSparseFeatures<float64_t>*) lhs)->sparse_dot(1.0,avec, alen, bvec, blen);

  if (inhomogene)
	  result+=1;

  result=CMath::pow(result, degree);

  ((CSparseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
