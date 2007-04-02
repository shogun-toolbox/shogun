/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/Mathematics.h"
#include "kernel/LinearStringKernel.h"
#include "features/StringFeatures.h"

CLinearStringKernel::CLinearStringKernel(INT size, bool do_rescale_, DREAL scale_)
: CStringKernel<CHAR>(size), scale(scale_), do_rescale(do_rescale_),
	initialized(false), normal(NULL)
{
}

CLinearStringKernel::~CLinearStringKernel()
{
	cleanup();
}

bool CLinearStringKernel::init(CFeatures *l, CFeatures *r)
{
	CStringKernel<CHAR>::init(l, r);

	if (!initialized)
		init_rescale();
	SG_INFO("rescaling kernel by %g (num:%d)\n", scale,
		CMath::min(l->get_num_vectors(), r->get_num_vectors()));
	return true;
}

void CLinearStringKernel::init_rescale()
{
	if (!do_rescale)
		return ;
	LONGREAL sum = 0;
	scale = 1.0;
	for (LONG i = 0; i<lhs->get_num_vectors() && i<rhs->get_num_vectors(); i++)
		sum += compute(i, i);

	if (sum > pow(2, 8*sizeof(LONG)))
		SG_ERROR("the sum %lf does not fit into integer of %d bits "
			"expect bogus results.\n", sum, 8*sizeof(LONG));
	scale = sum/CMath::min(lhs->get_num_vectors(), rhs->get_num_vectors());
	initialized = true;
}

void CLinearStringKernel::cleanup()
{
	delete_optimization();
}

bool CLinearStringKernel::load_init(FILE *src)
{
	return false;
}

bool CLinearStringKernel::save_init(FILE *dest)
{
	return false;
}

void CLinearStringKernel::clear_normal()
{
	memset(normal, 0, lhs->get_num_vectors()*sizeof(DREAL));
}

void CLinearStringKernel::add_to_normal(INT idx, DREAL weight)
{
	INT vlen;
	CHAR *vec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx, vlen);

	for (INT i = 0; i<vlen; i++)
		normal[i] += weight*vec[i];
}

DREAL CLinearStringKernel::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;

	CHAR *avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(idx_a, alen);
	CHAR *bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	ASSERT(alen==blen);
	double sum = 0;
	for (INT i = 0; i<alen; i++) /* FIXME: use dot from Mathematics.h */
		sum += ((LONG) avec[i])*((LONG) bvec[i]);
	return sum/scale;
}

bool CLinearStringKernel::init_optimization(INT num_suppvec, INT *sv_idx,
		DREAL *alphas)
{
	SG_DEBUG("drin gelandet yeah\n");
	INT i, alen;

	int num_feat = ((CStringFeatures<CHAR>*) lhs)->get_max_vector_length();
	ASSERT(num_feat);

	normal = new DREAL[num_feat];
	ASSERT(normal);
	clear_normal();

	for (i = 0; i<num_suppvec; i++)
	{
		CHAR *avec = ((CStringFeatures<CHAR>*) lhs)->get_feature_vector(sv_idx[i], alen);
		ASSERT(avec);

		for (INT j = 0; j<num_feat; j++)
			normal[j] += alphas[i]*((double) avec[j]);
	}
	set_is_initialized(true);
	return true;
}

bool CLinearStringKernel::delete_optimization()
{
	delete[] normal;
	normal = NULL;
	set_is_initialized(false);
	return true;
}

DREAL CLinearStringKernel::compute_optimized(INT idx_b)
{
	INT blen;

	CHAR *bvec = ((CStringFeatures<CHAR>*) rhs)->get_feature_vector(idx_b, blen);

	double result = 0;
	for (INT i = 0; i<blen; i++) /* FIXME: Use dot() from Mathematics.h */
		result += normal[i]*((double) bvec[i]);
	return result/scale;
}
