#include <shogun/converter/TDistributedStochasticNeighborEmbedding.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_LAPACK
/* Basic test for t-SNE, that just checks that it works anyhow */
TEST(TDistributedStochasticNeighborEmbeddingTest,basic)
{
	const index_t n_samples = 15;
	const index_t n_dimensions = 3;
	const index_t n_target_dimensions = 2;
	CDenseFeatures<float64_t>* high_dimensional_features =
		new CDenseFeatures<float64_t>(CDataGenerator::generate_gaussians(n_samples, 1, n_dimensions));

	CTDistributedStochasticNeighborEmbedding* embedder =
		new CTDistributedStochasticNeighborEmbedding();

	embedder->set_target_dim(n_target_dimensions);
	EXPECT_EQ(n_target_dimensions, embedder->get_target_dim());

	/* Set perplexity so that it is in range
	 * 0<=perplexity<=(n_samples - 1)/3.
	 */
	embedder->set_perplexity(n_samples / 5.0);

	CDenseFeatures<float64_t>* low_dimensional_features =
		embedder->embed(high_dimensional_features);

	EXPECT_EQ(n_target_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());

	SG_UNREF(embedder);
	SG_UNREF(high_dimensional_features);
	SG_UNREF(low_dimensional_features);
}
#endif // HAVE_LAPACK

