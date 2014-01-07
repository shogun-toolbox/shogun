/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <preprocessor/NormOne.h>
#include <preprocessor/DensePreprocessor.h>
#include <mathematics/Math.h>
#include <features/Features.h>

using namespace shogun;

CNormOne::CNormOne()
: CDensePreprocessor<float64_t>()
{
}

CNormOne::~CNormOne()
{
}

/// initialize preprocessor from features
bool CNormOne::init(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_DENSE)
	ASSERT(features->get_feature_type()==F_DREAL)

	return true;
}

/// clean up allocated memory
void CNormOne::cleanup()
{
}

/// initialize preprocessor from file
bool CNormOne::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CNormOne::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CNormOne::apply_to_feature_matrix(CFeatures* features)
{
	SGMatrix<float64_t> feature_matrix=((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	for (int32_t i=0; i<feature_matrix.num_cols; i++)
	{
		float64_t* vec= &(feature_matrix.matrix[i*feature_matrix.num_rows]);
		float64_t norm=CMath::sqrt(SGVector<float64_t>::dot(vec, vec, feature_matrix.num_rows));
		SGVector<float64_t>::scale_vector(1.0/norm, vec, feature_matrix.num_rows);
	}
	return feature_matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CNormOne::apply_to_feature_vector(SGVector<float64_t> vector)
{
	float64_t* normed_vec = SG_MALLOC(float64_t, vector.vlen);
	float64_t norm=CMath::sqrt(SGVector<float64_t>::dot(vector.vector, vector.vector, vector.vlen));

	for (int32_t i=0; i<vector.vlen; i++)
		normed_vec[i]=vector.vector[i]/norm;

	return SGVector<float64_t>(normed_vec,vector.vlen);
}
