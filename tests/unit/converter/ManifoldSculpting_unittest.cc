/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/converter/ManifoldSculpting.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;

#ifdef HAVE_LAPACK
/* Basic test for manifold sculpting, that just checks that it works anyhow */
TEST(ManifoldSculptingTest,DISABLED_basic)
{
	const index_t n_samples = 15;
	const index_t n_dimensions = 3;
	const index_t n_target_dimensions = 2;
	CDenseFeatures<float64_t>* high_dimensional_features =
		new CDenseFeatures<float64_t>(CDataGenerator::generate_gaussians(n_samples, 1, n_dimensions));

	CManifoldSculpting* embedder =
		new CManifoldSculpting();

	embedder->set_target_dim(n_target_dimensions);
	EXPECT_EQ(n_target_dimensions, embedder->get_target_dim());

	embedder->set_k(5);

	auto low_dimensional_features =
	    embedder->transform(high_dimensional_features)
	        ->as<CDenseFeatures<float64_t>>();

	EXPECT_EQ(n_target_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());

	SG_UNREF(embedder);
	SG_UNREF(high_dimensional_features);
}
#endif // HAVE_LAPACK
