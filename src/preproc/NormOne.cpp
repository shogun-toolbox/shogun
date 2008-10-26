/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "preproc/NormOne.h"
#include "preproc/SimplePreProc.h"
#include "lib/Mathematics.h"
#include "features/Features.h"
#include "features/RealFeatures.h"

CNormOne::CNormOne()
: CSimplePreProc<DREAL>("NormOne", "NRM1")
{
}

CNormOne::~CNormOne()
{
}

/// initialize preprocessor from features
bool CNormOne::init(CFeatures* f)
{
	ASSERT(f->get_feature_class()==C_SIMPLE);
	ASSERT(f->get_feature_type()==F_DREAL);

	return true;
}

/// clean up allocated memory
void CNormOne::cleanup()
{
}

/// initialize preprocessor from file
bool CNormOne::load(FILE* f)
{
	return false;
}

/// save preprocessor init-data to file
bool CNormOne::save(FILE* f)
{
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
DREAL* CNormOne::apply_to_feature_matrix(CFeatures* f)
{
	int32_t num_vec;
	int32_t num_feat;
	DREAL* matrix=((CRealFeatures*) f)->get_feature_matrix(num_feat, num_vec);

	for (int32_t i=0; i<num_vec; i++)
	{
		DREAL* vec=&matrix[i*num_feat];
		DREAL norm=CMath::sqrt(CMath::dot(vec, vec, num_feat));
		CMath::scale_vector(1.0/norm, vec, num_feat);
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
DREAL* CNormOne::apply_to_feature_vector(DREAL* f, int32_t& len)
{
	DREAL* vec=new DREAL[len];
	DREAL norm=CMath::sqrt(CMath::dot(f, f, len));

	for (int32_t i=0; i<len; i++)
		vec[i]=f[i]/norm;

	return vec;
}

/// initialize preprocessor from file
bool CNormOne::load_init_data(FILE* src)
{
	return true;
}

/// save init-data (like transforamtion matrices etc) to file
bool CNormOne::save_init_data(FILE* dst)
{
	return true;
}
