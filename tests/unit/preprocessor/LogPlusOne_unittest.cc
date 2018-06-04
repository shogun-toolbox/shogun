/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Shubham Shukla, Soeren Sonnenburg
 */

#include <shogun/mathematics/Math.h>
#include <shogun/preprocessor/LogPlusOne.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LogPlusOne, apply)
{
	float64_t data[6] = {1,2,3,4,5,6};
	int32_t num_vectors = 2;
	int32_t num_features = 3;
	SGMatrix<float64_t> orig(data, num_features, num_vectors, false);
	SGMatrix<float64_t> m = orig.clone();

	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(m);
	CLogPlusOne* preproc = new CLogPlusOne();
	preproc->fit(feats);

	feats = preproc->apply(feats)->as<CDenseFeatures<float64_t>>();

	for (index_t i = 0; i < num_vectors; i++)
	{
		SGVector<float64_t> v = feats->get_feature_vector(i);
		float64_t* v_orig = orig.get_column_vector(i);
		for (index_t j = 0; j < num_features; j++) {
			float64_t e = std::log(v_orig[j] + 1.0);
			EXPECT_DOUBLE_EQ(e, v[j]);
		}
	}

	SG_UNREF(feats);
}
