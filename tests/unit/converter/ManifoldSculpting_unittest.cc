#include <converter/ManifoldSculpting.h>
#include <features/DenseFeatures.h>
#include <features/DataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_EIGEN3

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

	CDenseFeatures<float64_t>* low_dimensional_features =
		embedder->embed(high_dimensional_features);

	EXPECT_EQ(n_target_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());

	SG_UNREF(embedder);
	SG_UNREF(high_dimensional_features);
}
#endif // HAVE_LAPACK

#endif
