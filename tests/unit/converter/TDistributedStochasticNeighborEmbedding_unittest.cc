/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn, Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/converter/TDistributedStochasticNeighborEmbedding.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;

#ifdef HAVE_LAPACK
/* Basic test for t-SNE, that just checks that it works anyhow */
TEST(TDistributedStochasticNeighborEmbeddingTest,basic)
{
	std::mt19937_64 prng(24);

	const index_t n_samples = 15;
	const index_t n_dimensions = 3;
	const index_t n_target_dimensions = 2;
	auto high_dimensional_features =
		std::make_shared<DenseFeatures<float64_t>>(DataGenerator::generate_gaussians(n_samples, 1, n_dimensions, prng));

	auto embedder =
		std::make_shared<TDistributedStochasticNeighborEmbedding>();

	embedder->set_target_dim(n_target_dimensions);
	EXPECT_EQ(n_target_dimensions, embedder->get_target_dim());

	/* Set perplexity so that it is in range
	 * 0<=perplexity<=(n_samples - 1)/3.
	 */
	embedder->set_perplexity(n_samples / 5.0);

	auto low_dimensional_features =
	    embedder->transform(high_dimensional_features)
	        ->as<DenseFeatures<float64_t>>();

	EXPECT_EQ(n_target_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());
}
#endif // HAVE_LAPACK

