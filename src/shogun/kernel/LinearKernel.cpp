/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <lib/common.h>
#include <io/SGIO.h>
#include <features/Features.h>
#include <features/DotFeatures.h>
#include <kernel/LinearKernel.h>

using namespace shogun;

CLinearKernel::CLinearKernel()
: CDotKernel(0)
{
	properties |= KP_LINADD;
}

CLinearKernel::CLinearKernel(CDotFeatures* l, CDotFeatures* r)
: CDotKernel(0)
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
	CDotKernel::init(l, r);

	return init_normalizer();
}

void CLinearKernel::cleanup()
{
	delete_optimization();

	CKernel::cleanup();
}

void CLinearKernel::add_to_normal(int32_t idx, float64_t weight)
{
	((CDotFeatures*) lhs)->add_to_dense_vec(
		normalizer->normalize_lhs(weight, idx), idx, normal.vector, normal.size());
	set_is_initialized(true);
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

bool CLinearKernel::init_optimization(CKernelMachine* km)
{
	clear_normal();

	int32_t num_suppvec=km->get_num_support_vectors();

	for (int32_t i=0; i<num_suppvec; i++)
		add_to_normal(km->get_support_vector(i), km->get_alpha(i));

	set_is_initialized(true);
	return true;
}

bool CLinearKernel::delete_optimization()
{
	normal = SGVector<float64_t>();
	set_is_initialized(false);

	return true;
}

float64_t CLinearKernel::compute_optimized(int32_t idx)
{
	ASSERT(get_is_initialized())
	float64_t result = ((CDotFeatures*) rhs)->
		dense_dot(idx, normal.vector, normal.size());
	return normalizer->normalize_rhs(result, idx);
}
