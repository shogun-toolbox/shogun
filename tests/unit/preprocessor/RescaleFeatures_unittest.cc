/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 */

#include <preprocessor/RescaleFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(RescaleFeatures, apply_to_feature_matrix)
{
	index_t num_features = 3;
	index_t num_vectors = 10;
	SGVector<float64_t> min(num_features), range(num_features);
	SGVector<float64_t> v(num_features*num_vectors), ev;
	sg_rand->set_seed(12345);
	v.random(-1024, 1024);
	ev = v.clone();

	SGMatrix<float64_t> m(v.vector, num_features, num_vectors, false);
	SGMatrix<float64_t> em(ev.vector, num_features, num_vectors, false);
	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(m);
	CRescaleFeatures* rescaler = new CRescaleFeatures();
	rescaler->init(feats);

	/* find the min and range for each feature among all the vectors */
	for (index_t i = 0; i < num_features; i++)
	{
		SGVector<float64_t> t = em.get_row_vector(i);
		min[i] = SGVector<float64_t>::min(t.vector, t.vlen);
		range[i] = SGVector<float64_t>::max(t.vector, t.vlen) - min[i];
	}

	feats->add_preprocessor(rescaler);
	feats->apply_preprocessor();
	for (index_t i = 0; i < num_vectors; i++)
	{
		SGVector<float64_t> vec = feats->get_feature_vector(i);
		float64_t* v_orig = em.get_column_vector(i);
		for (index_t j = 0; j < num_features; j++) {
			float64_t e = (v_orig[j]-min[j])/range[j];
			EXPECT_DOUBLE_EQ(e, vec[j]);
		}
	}

	SG_UNREF(feats);
}
