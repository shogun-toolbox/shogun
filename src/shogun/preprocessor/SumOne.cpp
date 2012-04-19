/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Sergey Lisitsyn
 * Copyright (C) 2012 Sergey Lisitsyn
 */

#include <shogun/preprocessor/SumOne.h>
#include <shogun/preprocessor/SimplePreprocessor.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/Features.h>
#include <shogun/features/SimpleFeatures.h>

using namespace shogun;

CSumOne::CSumOne()
: CSimplePreprocessor<float64_t>()
{
}

CSumOne::~CSumOne()
{
}

/// initialize preprocessor from features
bool CSumOne::init(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_SIMPLE);
	ASSERT(features->get_feature_type()==F_DREAL);

	return true;
}

/// clean up allocated memory
void CSumOne::cleanup()
{
}

/// initialize preprocessor from file
bool CSumOne::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// save preprocessor init-data to file
bool CSumOne::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

/// apply preproc on feature matrix
/// result in feature matrix
/// return pointer to feature_matrix, i.e. f->get_feature_matrix();
SGMatrix<float64_t> CSumOne::apply_to_feature_matrix(CFeatures* features)
{
	SGMatrix<float64_t> feature_matrix=((CSimpleFeatures<float64_t>*)features)->get_feature_matrix();

	for (int32_t i=0; i<feature_matrix.num_cols; i++)
	{
		float64_t* vec= &(feature_matrix.matrix[i*feature_matrix.num_rows]);
		float64_t sum = CMath::sum(vec,feature_matrix.num_rows);
		CMath::scale_vector(1.0/sum, vec, feature_matrix.num_rows);
	}
	return feature_matrix;
}

/// apply preproc on single feature vector
/// result in feature matrix
SGVector<float64_t> CSumOne::apply_to_feature_vector(const SGVector<float64_t>&w vector)
{
	float64_t* normed_vec = SG_MALLOC(float64_t, vector.vlen);
	float64_t sum = CMath::sum(vector.vector, vector.vlen);

	for (int32_t i=0; i<vector.vlen; i++)
		normed_vec[i]=vector.vector[i]/sum;

	return SGVector<float64_t>(normed_vec,vector.vlen);
}
