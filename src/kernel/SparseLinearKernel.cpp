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
#include "features/Features.h"
#include "features/SparseFeatures.h"
#include "kernel/SparseLinearKernel.h"
#include "kernel/SparseKernel.h"

CSparseLinearKernel::CSparseLinearKernel(INT size, DREAL s)
: CSparseKernel<DREAL>(size), scale(s), initialized(false), normal(NULL),
	normal_length(0)
{
	properties |= KP_LINADD;
}

CSparseLinearKernel::CSparseLinearKernel(
	CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r, DREAL s, INT size)
: CSparseKernel<DREAL>(size), scale(s), initialized(false), normal(NULL),
	normal_length(0)
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
	CSparseKernel<DREAL>::init(l, r);

	if (!initialized)
		init_rescale();

	SG_INFO( "rescaling kernel by %g (num:%d)\n",scale, CMath::min(l->get_num_vectors(), r->get_num_vectors()));

	return true;
}

void CSparseLinearKernel::init_rescale()
{
	if (scale!=0.0)
		return;
	double sum=0;
	scale=1.0;
	for (INT i=0; (i<lhs->get_num_vectors() && i<rhs->get_num_vectors()); i++)
			sum+=compute(i, i);

	scale=sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
	initialized=true;
}

void CSparseLinearKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
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
	int num = ((CSparseFeatures<DREAL>*) lhs)->get_num_features();
	if (normal==NULL)
	{
		normal = new DREAL[num];
		normal_length=num;
	}

	memset(normal, 0, sizeof(DREAL)*normal_length);

	set_is_initialized(true);
}

void CSparseLinearKernel::add_to_normal(INT idx, DREAL weight) 
{
	((CSparseFeatures<DREAL>*) rhs)->add_to_dense_vec(weight, idx, normal, normal_length);
	set_is_initialized(true);
}
  
DREAL CSparseLinearKernel::compute(INT idx_a, INT idx_b)
{
  INT alen=0;
  INT blen=0;
  bool afree=false;
  bool bfree=false;

  TSparseEntry<DREAL>* avec=((CSparseFeatures<DREAL>*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
  TSparseEntry<DREAL>* bvec=((CSparseFeatures<DREAL>*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);
  
  DREAL result=((CSparseFeatures<DREAL>*) lhs)->sparse_dot(1.0/scale, avec,alen, bvec,blen);

  ((CSparseFeatures<DREAL>*) lhs)->free_feature_vector(avec, idx_a, afree);
  ((CSparseFeatures<DREAL>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

  return result;
}

bool CSparseLinearKernel::init_optimization(INT num_suppvec, INT* sv_idx, DREAL* alphas) 
{
	clear_normal();

	for (int i=0; i<num_suppvec; i++)
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

DREAL CSparseLinearKernel::compute_optimized(INT idx) 
{
	ASSERT(get_is_initialized());
	return ((CSparseFeatures<DREAL>*) rhs)->dense_dot(1.0/scale, idx, normal, normal_length, 0.0);
}
