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
 : CDensePreprocessor<float64_t>(),
 m_initialized(false)
{
	register_parameters();
}

CRescaleFeatures::~CRescaleFeatures()
{
	cleanup();
}

bool CRescaleFeatures::init(CFeatures* features)
{
	if (!m_initialized)
	{
		ASSERT(features->get_feature_class()==C_DENSE);
		ASSERT(features->get_feature_type()==F_DREAL);

		CDenseFeatures<float64_t>* simple_features=(CDenseFeatures<float64_t>*) features;
		int32_t num_examples = simple_features->get_num_vectors();
		int32_t num_features = simple_features->get_num_features();
		REQUIRE(num_examples > 1,
                        "number of feature vectors should be at least 2!\n");

		SG_INFO("Extracting min and range values for each feature\n")

		m_min = SGVector<float64_t>(num_features);
		m_range = SGVector<float64_t>(num_features);
		SGMatrix<float64_t> feature_matrix=((CDenseFeatures<float64_t>*)features)->get_feature_matrix();
		for (index_t i = 0; i < num_features; i++)
		{
			SGVector<float64_t> vec = feature_matrix.get_row_vector(i);
			float64_t cur_min = vec[0];
			float64_t cur_max = vec[0];

			/* find the max and min values in one loop */
			for (index_t j = 1; j < vec.vlen; j++)
			{
				cur_min = CMath::min(vec[j], cur_min);
				cur_max = CMath::max(vec[j], cur_max);
			}

			/* only rescale if range > 0 */
			if ((cur_max - cur_min) > 0) {
				m_min[i] = cur_min;
				m_range[i] = 1.0/(cur_max - cur_min);
			}
			else {
				m_min[i] = 0.0;
				m_range[i] = 1.0;
			}
		}

		m_initialized = true;

		return true;
	}

	return false;
}

void CRescaleFeatures::cleanup()
{
	m_initialized = false;
}

SGMatrix<float64_t> CRescaleFeatures::apply_to_feature_matrix(CFeatures* features)
{
	ASSERT(m_initialized);

	SGMatrix<float64_t> feature_matrix=((CDenseFeatures<float64_t>*)features)->get_feature_matrix();
	ASSERT(feature_matrix.num_rows == m_min.vlen);

	for (index_t i = 0; i < feature_matrix.num_cols; i++)
	{
		float64_t* vec = feature_matrix.get_column_vector(i);
		SGVector<float64_t>::vec1_plus_scalar_times_vec2(vec, -1.0, m_min.vector, feature_matrix.num_rows);
		for (index_t j = 0; j < feature_matrix.num_rows; j++) {
			vec[j] *= m_range[j];
		}
	}

	return feature_matrix;
}

SGVector<float64_t> CRescaleFeatures::apply_to_feature_vector(SGVector<float64_t> vector)
{
	ASSERT(m_initialized);
	ASSERT(m_min.vlen == vector.vlen);

	float64_t* ret = SG_MALLOC(float64_t, vector.vlen);
	SGVector<float64_t>::add(ret, 1.0, vector.vector, -1.0, m_min.vector, vector.vlen);
	for (index_t i = 0; i < vector.vlen; i++) {
		ret[i] *= m_range[i];
	}

	return SGVector<float64_t>(ret,vector.vlen);
}

void CRescaleFeatures::register_parameters()
{
	SG_ADD(&m_min, "min", "minimum values of each feature", MS_NOT_AVAILABLE);
	SG_ADD(&m_range, "range", "Reciprocal of the range of each feature", MS_NOT_AVAILABLE);
	SG_ADD(&m_initialized, "initialized", "Indicator of the state of the preprocessor.", MS_NOT_AVAILABLE);
}
