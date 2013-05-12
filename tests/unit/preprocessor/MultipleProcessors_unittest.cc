/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (C) 2013 Soeren Sonnenburg
 */

#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/LogPlusOne.h>
#include <shogun/preprocessor/SumOne.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MultipleProcessors, apply_to_feature_matrix)
{
	float64_t data[6] = {1,2,3,4,5,6};
	int32_t num_vectors = 2;
	int32_t num_features = 3;
	SGMatrix<float64_t> orig(data, num_features, num_vectors, false);
	SGMatrix<float64_t> m = orig.clone();

	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(m);
	CSumOne* sum1 = new CSumOne();
	CLogPlusOne* logp1 = new CLogPlusOne();
	sum1->init(feats);
	feats->add_preprocessor(sum1);

	logp1->init(feats);
	feats->add_preprocessor(logp1);
	feats->apply_preprocessor();

	EXPECT_EQ(2, feats->get_num_preprocessors());

	for (index_t i = 0; i < num_vectors; i++)
	{
		SGVector<float64_t> v = feats->get_feature_vector(i);
		float64_t* v_orig = orig.get_column_vector(i);
		float64_t sum = SGVector<float64_t>::sum(v_orig, num_features);
		for (index_t j = 0; j < num_features; j++) {
			float64_t e = CMath::log(v_orig[j]/sum+1.0);
			EXPECT_DOUBLE_EQ(e, v[j]);
		}
	}

	SG_UNREF(feats);
}
