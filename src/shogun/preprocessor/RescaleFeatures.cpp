/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 20013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <shogun/preprocessor/RescaleFeatures.h>

using namespace shogun;

CRescaleFeatures::CRescaleFeatures()
 : CDensePreprocessor<float64_t>()
{

}

CRescaleFeatures::~CRescaleFeatures()
{

}

bool CRescaleFeatures::init(CFeatures* features)
{
	ASSERT(features->get_feature_class()==C_DENSE);
	ASSERT(features->get_feature_type()==F_DREAL);
	return true;
}

void CRescaleFeatures::cleanup()
{

}

bool CRescaleFeatures::load(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

bool CRescaleFeatures::save(FILE* f)
{
	SG_SET_LOCALE_C;
	SG_RESET_LOCALE;
	return false;
}

SGMatrix<float64_t> CRescaleFeatures::apply_to_feature_matrix(CFeatures* features)
{
	SGMatrix<float64_t> feature_matrix=((CDenseFeatures<float64_t>*)features)->get_feature_matrix();

	/* Need at least 2 feature vectors to apply this preprocessor */
	if (feature_matrix.num_cols < 2)
		return feature_matrix;
	
	for (index_t i = 0; i < feature_matrix.num_rows; i++)
	{
		SGVector<float64_t> vec = feature_matrix.get_row_vector(i);
		float64_t min = vec[0];
		float64_t max = vec[0];

		/* find the max and min values in one loop */
		for (index_t j = 1; j < vec.vlen; j++)
		{
			min = CMath::min(vec[j], min);
			max = CMath::max(vec[j], max);
		}
		float64_t range = max-min;

		if (range > 0)
		{
			for (index_t j = 0; j < feature_matrix.num_cols; j++)
			{
				float64_t& k = feature_matrix(i, j);
				k = (k-min)/range;
			}
		}
	}

	return feature_matrix;
}

SGVector<float64_t> CRescaleFeatures::apply_to_feature_vector(SGVector<float64_t> vector)
{
	/* nothing to do here */
	return vector;
}
