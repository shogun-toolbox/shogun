/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (C) 2013 Soeren Sonnenburg
 */

#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/preprocessor/LogPlusOne.h>
#include <shogun/preprocessor/SumOne.h>
#include <shogun/preprocessor/PruneVarSubMean.h>
#include <shogun/preprocessor/NormOne.h>
#include <shogun/base/some.h>
#include <gtest/gtest.h>

using namespace shogun;

const float64_t eps = 1e-15;

class MultipleProcessors : public ::testing::TestWithParam<bool> {};

TEST_P(MultipleProcessors, preprocess) {
	const index_t num_vectors = 2;
	const index_t num_features = 3;
	SGMatrix<float64_t> orig({{1,2},{3,4},{5,6}});
	SGMatrix<float64_t> m = orig.clone();

	auto feats = some<CDenseFeatures<float64_t> >(m);
	auto sum1 = some<CSumOne>();
	auto logp1 = some<CLogPlusOne>();

	sum1->init(feats);
	logp1->init(feats);
	auto pre_feats = feats->preprocess(sum1)->preprocess(logp1);

	if (GetParam())
		pre_feats->eval();

	for (index_t i = 0; i < num_vectors; i++)
	{
		auto v = pre_feats->get_feature_vector(i);
		auto v_orig = orig.get_column(i);
		auto sum = linalg::sum(v_orig);

		for (index_t j = 0; j < num_features; j++) {
			auto e = CMath::log(v_orig[j]/sum+1.0);
			EXPECT_DOUBLE_EQ(e, v[j]);
		}
	}
}

INSTANTIATE_TEST_CASE_P(Eval, MultipleProcessors, ::testing::Values(false, true));
