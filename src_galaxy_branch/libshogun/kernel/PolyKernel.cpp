/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"
#include "lib/common.h"
#include "lib/io.h"
#include "kernel/PolyKernel.h"
#include "kernel/SqrtDiagKernelNormalizer.h"
#include "features/SimpleFeatures.h"

CPolyKernel::CPolyKernel(int32_t size, int32_t d, bool i)
: CSimpleKernel<float64_t>(size), degree(d), inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
}

CPolyKernel::CPolyKernel(
	CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r, int32_t d, bool i, int32_t size)
: CSimpleKernel<float64_t>(size), degree(d), inhomogene(i)
{
	set_normalizer(new CSqrtDiagKernelNormalizer());
	init(l,r);
}

CPolyKernel::~CPolyKernel()
{
	cleanup();
}

bool CPolyKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<float64_t>::init(l,r);
	return init_normalizer();
}

void CPolyKernel::cleanup()
{
	CKernel::cleanup();
}

bool CPolyKernel::load_init(FILE* src)
{
	return false;
}

bool CPolyKernel::save_init(FILE* dest)
{
	return false;
}

float64_t CPolyKernel::compute(int32_t idx_a, int32_t idx_b)
{
  int32_t alen=0;
  int32_t blen=0;
  bool afree=false;
  bool bfree=false;

  float64_t* avec=
	((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
  float64_t* bvec=
	((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
  ASSERT(alen==blen);

  float64_t result=CMath::dot(avec, bvec, alen);

  if (inhomogene)
	  result+=1;

  result=CMath::pow(result, degree);

  ((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}
