/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <preprocessor/LogPlusOne.h>
#include <preprocessor/DensePreprocessor.h>
#include <features/Features.h>
#include <mathematics/Math.h>

using namespace shogun;

CLogPlusOne::CLogPlusOne()
: CDensePreprocessor<float64_t>()
{
}


CLogPlusOne::~CLogPlusOne()
{
}

/// initialize preprocessor from features
bool CLogPlusOne::init(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_DENSE)
	ASSERT(features->get_feature_type()==F_DREAL)

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
SGMatrix<float64_t> CLogPlusOne::apply_to_feature_matrix(CFeatures* features)
{
	SGMatrix<float64_t> feature_matrix =
			((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	for (int32_t i=0; i<feature_matrix.num_cols; i++)
	{
		for (int32_t j=0; j<feature_matrix.num_rows; j++)
			feature_matrix.matrix[i*feature_matrix.num_rows+j] =
					CMath::log(feature_matrix.matrix[i*feature_matrix.num_rows+j]+1.0);
	}
	return feature_matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CLogPlusOne::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* log_vec = SG_MALLOC(float64_t, vector.vlen);

	for (int32_t i=0; i<vector.vlen; i++)
		log_vec[i]=CMath::log(vector.vector[i]+1.0);

	return SGVector<float64_t>(log_vec,vector.vlen);
}
