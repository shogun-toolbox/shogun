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
	INT i,j;
	INT num_vec;
	INT num_feat;
	DREAL* matrix=((CRealFeatures*) f)->get_feature_matrix(num_feat, num_vec);

	for (i=0; i<num_vec; i++)
	{
		DREAL sqnorm=0;
		DREAL norm=0;
		DREAL* vec=&matrix[i*num_feat];

		for (j=0; j<num_feat; j++)
		{
			if (vec[j]>1e100)
				vec[j]=0;
			sqnorm+=vec[j]*vec[j];
		}

		norm=sqrt(sqnorm);

		for (j=0; j<num_feat; j++)
			vec[j]/=norm;
	}
	return matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
DREAL* CNormOne::apply_to_feature_vector(DREAL* f, INT& len)
{
	DREAL* vec=new DREAL[len];
	DREAL sqnorm=0;
	DREAL norm=0;
	INT i=0;

	for (i=0; i<len; i++)
		sqnorm+=f[i]*f[i];

	norm=sqrt(sqnorm);

	for (i=0; i<len; i++)
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
