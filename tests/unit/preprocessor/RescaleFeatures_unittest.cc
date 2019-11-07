/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Sanuj Sharma, Soeren Sonnenburg
 */

#include <gtest/gtest.h>
#include <shogun/preprocessor/RescaleFeatures.h>
#include <shogun/mathematics/RandomNamespace.h>

#include <random>

using namespace shogun;

TEST(RescaleFeatures, transform)
{
	int32_t seed = 12345;
	index_t num_features = 3;
	index_t num_vectors = 10;
	SGVector<float64_t> min(num_features), range(num_features);
	SGVector<float64_t> v(num_features*num_vectors), ev;
	std::mt19937_64 prng(seed);
	random::fill_array(v, -1024, 1024, prng);
	ev = v.clone();

	SGMatrix<float64_t> m(v.vector, num_features, num_vectors, false);
	SGMatrix<float64_t> em(ev.vector, num_features, num_vectors, false);
	auto feats = std::make_shared<DenseFeatures<float64_t>>(m);
	auto rescaler = std::make_shared<RescaleFeatures>();
	rescaler->fit(feats);

	/* find the min and range for each feature among all the vectors */
	for (index_t i = 0; i < num_features; i++)
	{
		SGVector<float64_t> t = em.get_row_vector(i);
		min[i] = Math::min(t.vector, t.vlen);
		range[i] = Math::max(t.vector, t.vlen) - min[i];
	}

	feats = rescaler->transform(feats)->as<DenseFeatures<float64_t>>();

	for (index_t i = 0; i < num_vectors; i++)
	{
		SGVector<float64_t> vec = feats->get_feature_vector(i);
		float64_t* v_orig = em.get_column_vector(i);
		for (index_t j = 0; j < num_features; j++) {
			float64_t e = (v_orig[j]-min[j])/range[j];
			EXPECT_DOUBLE_EQ(e, vec[j]);
		}
	}


}
