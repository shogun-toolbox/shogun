/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (C) 2013 Soeren Sonnenburg
 */

#include <mathematics/Math.h>
#include <preprocessor/LogPlusOne.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LogPlusOne, apply_to_feature_matrix)
{
	float64_t data[6] = {1,2,3,4,5,6};
	int32_t num_vectors = 2;
	int32_t num_features = 3;
	SGMatrix<float64_t> orig(data, num_features, num_vectors, false);
	SGMatrix<float64_t> m = orig.clone();

	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(m);
	CLogPlusOne* preproc = new CLogPlusOne();
	preproc->init(feats);

	feats->add_preprocessor(preproc);
	feats->apply_preprocessor();

	for (index_t i = 0; i < num_vectors; i++)
	{
		SGVector<float64_t> v = feats->get_feature_vector(i);
		float64_t* v_orig = orig.get_column_vector(i);
		for (index_t j = 0; j < num_features; j++) {
			float64_t e = CMath::log(v_orig[j]+1.0);
			EXPECT_DOUBLE_EQ(e, v[j]);
		}
	}

	SG_UNREF(feats);
}
