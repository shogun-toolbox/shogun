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
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "kernel/LinearKernel.h"

CLinearKernel::CLinearKernel()
: CSimpleKernel<float64_t>(0), normal(NULL), normal_length(0)
{
	properties |= KP_LINADD;
}

CLinearKernel::CLinearKernel(CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r)
: CSimpleKernel<float64_t>(0), normal(NULL), normal_length(0)
{
	properties |= KP_LINADD;
	init(l,r);
}

CLinearKernel::~CLinearKernel()
{
	cleanup();
}

bool CLinearKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<float64_t>::init(l, r);

	return init_normalizer();
}

void CLinearKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
}

void CLinearKernel::clear_normal()
{
	int32_t num = ((CSimpleFeatures<float64_t>*) lhs)->get_num_features();
	if (normal==NULL)
	{
		normal = new float64_t[num];
		normal_length=num;
	}

	memset(normal, 0, sizeof(float64_t)*normal_length);

	set_is_initialized(true);
}

void CLinearKernel::add_to_normal(int32_t idx, float64_t weight) 
{
	int32_t vlen;
	bool vfree;
	float64_t* vec=((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx, vlen, vfree);

	for (int32_t i=0; i<vlen; i++)
		normal[i]+= weight*normalizer->normalize_lhs(vec[i], idx);

	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(vec, idx, vfree);

	set_is_initialized(true);
}

float64_t CLinearKernel::compute(int32_t idx_a, int32_t idx_b)
{
  int32_t alen, blen;
  bool afree, bfree;

  float64_t* avec=
	((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
  float64_t* bvec=
	((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);

  ASSERT(alen==blen);

  float64_t result=CMath::dot(avec, bvec, alen);

  ((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

bool CLinearKernel::init_optimization(
	int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas)
{
	clear_normal();

	for (int32_t i=0; i<num_suppvec; i++)
		add_to_normal(sv_idx[i], alphas[i]);

	set_is_initialized(true);
	return true;
}

bool CLinearKernel::delete_optimization()
{
	delete[] normal;
	normal_length=0;
	normal=NULL;
	set_is_initialized(false);

	return true;
}

float64_t CLinearKernel::compute_optimized(int32_t idx)
{
	ASSERT(get_is_initialized());

	int32_t vlen;
	bool vfree;
	float64_t* vec=((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx, vlen, vfree);
	ASSERT(vlen==normal_length);
	float64_t result=CMath::dot(normal,vec, vlen);
	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(vec, idx, vfree);

	return normalizer->normalize_rhs(result, idx);
}
