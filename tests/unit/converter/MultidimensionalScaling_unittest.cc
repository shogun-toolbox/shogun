#include <shogun/converter/MultidimensionalScaling.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_EIGEN3

#ifdef HAVE_LAPACK
TEST(MultidimensionaScalingTest,distance_preserving)
{
	const index_t n_samples = 10;
	const index_t n_gaussians = 5;
	const index_t n_dimensions = 5;
	CDenseFeatures<float64_t>* high_dimensional_features =
		new CDenseFeatures<float64_t>(CDataGenerator::generate_gaussians(n_samples, n_gaussians, n_dimensions));

	CDistance* euclidean_distance =
		new CEuclideanDistance(high_dimensional_features, high_dimensional_features);

	CMultidimensionalScaling* mds_converter =
		new CMultidimensionalScaling();

	mds_converter->set_target_dim(n_dimensions);
	EXPECT_EQ(n_dimensions,mds_converter->get_target_dim());

	CDenseFeatures<float64_t>* low_dimensional_features =
		mds_converter->embed_distance(euclidean_distance);
	EXPECT_EQ(n_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());

	CDistance* euclidean_distance_for_embedding =
		new CEuclideanDistance(low_dimensional_features, low_dimensional_features);

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

	SG_UNREF(mds_converter);
	SG_UNREF(euclidean_distance);
	SG_UNREF(euclidean_distance_for_embedding);
}
#endif // HAVE_LAPACK

#endif
