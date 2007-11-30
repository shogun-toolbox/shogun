/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Soeren Sonnenburg
 * Copyright (C) 2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "distance/SparseEuclidianDistance.h"
#include "features/Features.h"
#include "features/SparseFeatures.h"

CSparseEuclidianDistance::CSparseEuclidianDistance()
: CSparseDistance<DREAL>()
{
}

CSparseEuclidianDistance::CSparseEuclidianDistance(
	CSparseFeatures<DREAL>* l, CSparseFeatures<DREAL>* r)
: CSparseDistance<DREAL>()
{
	init(l, r);
}

CSparseEuclidianDistance::~CSparseEuclidianDistance()
{
	cleanup();
}

bool CSparseEuclidianDistance::init(CFeatures* l, CFeatures* r)
{
	CSparseDistance<DREAL>::init(l, r);

	return true;
}

void CSparseEuclidianDistance::cleanup()
{
}

bool CSparseEuclidianDistance::load_init(FILE* src)
{
	return false;
}

bool CSparseEuclidianDistance::save_init(FILE* dest)
{
	return false;
}

DREAL CSparseEuclidianDistance::compute(INT idx_a, INT idx_b)
{
	INT alen, blen;
	bool afree, bfree;
	TSparseEntry<DREAL>* avec=((CSparseFeatures<DREAL>*) lhs)->get_sparse_feature_vector(idx_a, alen, afree);
	TSparseEntry<DREAL>* bvec=((CSparseFeatures<DREAL>*) rhs)->get_sparse_feature_vector(idx_b, blen, bfree);
	DREAL result=((CSparseFeatures<DREAL>*) lhs)->compute_squared_norm(avec, alen, bvec, blen);

	((CSparseFeatures<DREAL>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSparseFeatures<DREAL>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return CMath::sqrt(result);
}
