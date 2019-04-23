#include <gtest/gtest.h>
#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;

#ifdef HAVE_LAPACK
TEST(MultidimensionaScalingTest,distance_preserving)
{
	std::mt19937_64 prng(24);

	const index_t n_samples = 10;
	const index_t n_gaussians = 5;
	const index_t n_dimensions = 5;
	auto high_dimensional_features =
		std::make_shared<DenseFeatures<float64_t>>(DataGenerator::generate_gaussians(n_samples, n_gaussians, n_dimensions, prng));

	auto euclidean_distance =
		std::make_shared<EuclideanDistance>(high_dimensional_features, high_dimensional_features);

	auto mds_converter =
		std::make_shared<MultidimensionalScaling>();

	mds_converter->set_target_dim(n_dimensions);
	EXPECT_EQ(n_dimensions,mds_converter->get_target_dim());

	auto low_dimensional_features =
		mds_converter->embed_distance(euclidean_distance);
	EXPECT_EQ(n_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());

	auto euclidean_distance_for_embedding =
		std::make_shared<EuclideanDistance>(low_dimensional_features, low_dimensional_features);

	SGMatrix<float64_t> euclidean_distance_matrix =
		euclidean_distance->get_distance_matrix();
	SGMatrix<float64_t> euclidean_distance_for_embedding_matrix =
		euclidean_distance_for_embedding->get_distance_matrix();

	for (index_t i=0; i<euclidean_distance_matrix.num_rows; i++)
	{
		for (index_t j=0; j<euclidean_distance_matrix.num_cols; j++)
		{
			ASSERT_NEAR(euclidean_distance_matrix(i,j), euclidean_distance_for_embedding_matrix(i,j), 1e-9);
		}
	}




}
#endif // HAVE_LAPACK

