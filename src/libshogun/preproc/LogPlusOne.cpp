/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "preproc/LogPlusOne.h"
#include "preproc/SimplePreProc.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/Mathematics.h"

using namespace shogun;

CLogPlusOne::CLogPlusOne()
: CSimplePreProc<float64_t>("LogPlusOne", "LG+1")
{
}


CLogPlusOne::~CLogPlusOne()
{
}

/// initialize preprocessor from features
bool CLogPlusOne::init(CFeatures* f)
{
	ASSERT(f->get_feature_class()==C_SIMPLE);
	ASSERT(f->get_feature_type()==F_DREAL);

	return true;
}

/// clean up allocated memory
void CLogPlusOne::cleanup()
{
}

/// initialize preprocessor from file
bool CLogPlusOne::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CLogPlusOne::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
float64_t* CLogPlusOne::apply_to_feature_matrix(CFeatures* f)
{
	int32_t i,j;
	int32_t num_vec;
	int32_t num_feat;
	float64_t* matrix=((CSimpleFeatures<float64_t>*) f)->get_feature_matrix(num_feat, num_vec);

	for (i=0; i<num_vec; i++)
	{
		float64_t* vec=&matrix[i*num_feat];

		for (j=0; j<num_feat; j++)
			vec[j]=log(vec[j]+1);
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
float64_t* CLogPlusOne::apply_to_feature_vector(float64_t* f, int32_t& len)
{
	float64_t* vec=new float64_t[len];
	int32_t i=0;

	for (i=0; i<len; i++)
		vec[i]=log(f[i]+1);

	return vec;
}
