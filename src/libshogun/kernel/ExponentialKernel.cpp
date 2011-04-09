/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gaussian Kernel used as template, attribution:
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 * 
 * Slightly edited by Justin Patera 2011
 */

#include "lib/common.h"
#include "base/Parameter.h"
#include "kernel/ExponentialKernel.h"
#include "features/DotFeatures.h"
#include "lib/io.h"

using namespace shogun;

CExponentialKernel::CExponentialKernel()
	: CDotKernel()
{
	init();
}


CExponentialKernel::CExponentialKernel(int32_t size, float64_t w)
: CDotKernel(size)
{
	init();
	width=w;
}

CExponentialKernel::CExponentialKernel(
	CDotFeatures* l, CDotFeatures* r, float64_t w, int32_t size)
: CDotKernel(size)
{
	init();
	width=w;

	init(l,r);
}

CExponentialKernel::~CExponentialKernel()
{
	cleanup();
}

void CExponentialKernel::cleanup()
{
	if (sq_lhs != sq_rhs)
		delete[] sq_rhs;
	sq_rhs = NULL;

	delete[] sq_lhs;
	sq_lhs = NULL;

	CKernel::cleanup();
}
//probably wont need this...
void CExponentialKernel::precompute_squared_helper(float64_t* &buf, CDotFeatures* df)
{
	ASSERT(df);
	int32_t num_vec=df->get_num_vectors();
	buf=new float64_t[num_vec];

	for (int32_t i=0; i<num_vec; i++)
		buf[i]=df->dot(i,df, i);
}

bool CExponentialKernel::init(CFeatures* l, CFeatures* r)
{
	///free sq_{r,l}hs first
	cleanup();

	CDotKernel::init(l, r);
	precompute_squared();
	return init_normalizer();
}

float64_t CExponentialKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=sq_lhs[idx_a]+sq_rhs[idx_b]-2*CDotKernel::compute(idx_a,idx_b);
	return exp(-result/width);
}

void CExponentialKernel::load_serializable_post(void) throw (ShogunException)
{
	CKernel::load_serializable_post();
	precompute_squared();
}
//probably wont need this either
void CExponentialKernel::precompute_squared()
{
	if (!lhs || !rhs)
		return;

	precompute_squared_helper(sq_lhs, (CDotFeatures*) lhs);

	if (lhs==rhs)
		sq_rhs=sq_lhs;
	else
		precompute_squared_helper(sq_rhs, (CDotFeatures*) rhs);
}

void CExponentialKernel::init()
{
	width=1;
	sq_lhs=NULL;
	sq_rhs=NULL;
	m_parameters->add(&width, "width", "Kernel width.");
}
