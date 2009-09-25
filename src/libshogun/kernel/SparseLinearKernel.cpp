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
#include "features/SparseFeatures.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparseKernel.h"

CSparseLinearKernel::CSparseLinearKernel()
: CSparseKernel<float64_t>(0), normal(NULL), normal_length(0)
{
	properties |= KP_LINADD;
}

CSparseLinearKernel::CSparseLinearKernel(
	CSparseFeatures<float64_t>* l, CSparseFeatures<float64_t>* r)
: CSparseKernel<float64_t>(0), normal(NULL), normal_length(0)
{
	properties |= KP_LINADD;
	init(l,r);
}

CSparseLinearKernel::~CSparseLinearKernel()
{
	cleanup();
}

bool CSparseLinearKernel::init(CFeatures* l, CFeatures* r)
{
	CSparseKernel<float64_t>::init(l, r);
	return init_normalizer();
}

void CSparseLinearKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
}

void CSparseLinearKernel::clear_normal()
{
	int32_t num=((CSparseFeatures<float64_t>*) lhs)->get_num_features();
	if (normal==NULL)
	{
		normal=new float64_t[num];
		normal_length=num;
	}

	memset(normal, 0, sizeof(float64_t)*normal_length);
	set_is_initialized(true);
}

void CSparseLinearKernel::add_to_normal(int32_t idx, float64_t weight)
{
	((CSparseFeatures<float64_t>*) rhs)->add_to_dense_vec(
		normalizer->normalize_lhs(weight, idx), idx, normal, normal_length);
	set_is_initialized(true);
}
  
float64_t CSparseLinearKernel::compute(int32_t idx_a, int32_t idx_b)
{
  int32_t alen=0;
  int32_t blen=0;
  bool afree=false;
  bool bfree=false;

  TSparseEntry<float64_t>* avec=((CSparseFeatures<float64_t>*) lhs)->
  	get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<float64_t>* bvec=((CSparseFeatures<float64_t>*) rhs)->
  	get_sparse_feature_vector(idx_b, blen, bfree);

  float64_t result=((CSparseFeatures<float64_t>*) lhs)->
  	sparse_dot(1.0, avec,alen, bvec,blen);

  ((CSparseFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

bool CSparseLinearKernel::init_optimization(
	int32_t num_suppvec, int32_t* sv_idx, float64_t* alphas)
{
	clear_normal();

	for (int32_t i=0; i<num_suppvec; i++)
		add_to_normal(sv_idx[i], alphas[i]);

	set_is_initialized(true);
	return true;
}

bool CSparseLinearKernel::delete_optimization()
{
	delete[] normal;
	normal_length=0;
	normal=NULL;
	set_is_initialized(false);

	return true;
}

float64_t CSparseLinearKernel::compute_optimized(int32_t idx)
{
	ASSERT(get_is_initialized());
	float64_t result = ((CSparseFeatures<float64_t>*) rhs)->
		dense_dot(1.0, idx, normal, normal_length, 0.0);
	return normalizer->normalize_rhs(result, idx);
}
