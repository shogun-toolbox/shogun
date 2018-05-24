/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Sanuj Sharma, Soeren Sonnenburg
 */

#include <shogun/preprocessor/RescaleFeatures.h>
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
	rescaler->fit(feats);

	/* find the min and range for each feature among all the vectors */
	for (index_t i = 0; i < num_features; i++)
	{
		SGVector<float64_t> t = em.get_row_vector(i);
		min[i] = CMath::min(t.vector, t.vlen);
		range[i] = CMath::max(t.vector, t.vlen) - min[i];
	}

	feats = rescaler->apply(feats)->as<CDenseFeatures<float64_t>>();

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
