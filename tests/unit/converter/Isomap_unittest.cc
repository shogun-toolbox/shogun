#include <vector>
#include <set>
#include <algorithm>

#include <shogun/converter/Isomap.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/SGMatrix.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_EIGEN3

#ifdef HAVE_LAPACK
TEST(IsomapTest,DISABLED_distance_preserving_max_k)
{
	const index_t n_samples = 5;
	const index_t n_gaussians = 5;
	const index_t n_dimensions = 5;
	CDenseFeatures<float64_t>* high_dimensional_features =
		new CDenseFeatures<float64_t>(CDataGenerator::generate_gaussians(n_samples, n_gaussians, n_dimensions));

	CDistance* euclidean_distance =
		new CEuclideanDistance(high_dimensional_features, high_dimensional_features);

	CIsomap* isomap_converter =
		new CIsomap();

	isomap_converter->set_target_dim(n_dimensions);
	EXPECT_EQ(n_dimensions,isomap_converter->get_target_dim());

	isomap_converter->set_k(n_samples*n_gaussians-1);
	EXPECT_EQ(n_samples*n_gaussians-1,isomap_converter->get_k());

	CDenseFeatures<float64_t>* low_dimensional_features =
		isomap_converter->embed_distance(euclidean_distance);
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

	SG_UNREF(isomap_converter);
	SG_UNREF(euclidean_distance);
	SG_UNREF(euclidean_distance_for_embedding);
}
#endif // HAVE_LAPACK

struct index_and_distance_struct
{
	float64_t distance;
	index_t neighbor_index;
} ;

struct heap_comparator
{
	bool operator() (const index_and_distance_struct& first, const index_and_distance_struct& second)
	{
		return first.distance > second.distance;
	}
} comparator;

std::set<index_t> get_neighbors_indices(CDistance* distance_object, index_t feature_vector_index, index_t n_neighbors);

void check_similarity_of_sets(const std::set<index_t>& first_set, const std::set<index_t>& second_set, float64_t min_similarity_level);

/* Fills passed-in matrix with special test data, that is suitable
 * for dimensionality reduction - basically, a set of points, randomly
 * sampled from a hyperplane.
 */
void fill_matrix_with_test_data(SGMatrix<float64_t>& matrix_to_fill);

TEST(IsomapTest,neighbors_preserving)
{
	const index_t n_samples = 30;
	const index_t n_dimensions = 3;
	const index_t n_target_dimensions = 2;
	const index_t n_neighbors = 10;
	const float64_t required_similarity_level = 0.8;

	SGMatrix<float64_t> high_dimensional_matrix = SGMatrix<float64_t>::get_allocated_matrix(n_dimensions, n_samples);

	fill_matrix_with_test_data(high_dimensional_matrix);

	CDenseFeatures<float64_t>* high_dimensional_features =
		new CDenseFeatures<float64_t>(high_dimensional_matrix);

	CDistance* high_dimensional_dist =
		new CEuclideanDistance(high_dimensional_features, high_dimensional_features);

	std::vector<std::set<index_t> > high_dimensional_neighbors_for_vectors;
	/* Find n_neighbors nearest neighbors for each vector */
	for (index_t i=0; i<n_samples; ++i)
	{
		high_dimensional_neighbors_for_vectors.push_back(get_neighbors_indices(high_dimensional_dist, i, n_neighbors));
	}

	CIsomap* isoEmbedder = new CIsomap();

	isoEmbedder->set_k(n_neighbors);

	isoEmbedder->set_target_dim(n_target_dimensions);
	EXPECT_EQ(n_target_dimensions, isoEmbedder->get_target_dim());

	CDenseFeatures<float64_t>* low_dimensional_features =
		isoEmbedder->embed(high_dimensional_features);

	EXPECT_EQ(n_target_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());

	CDistance* low_dimensional_dist =
		new CEuclideanDistance(low_dimensional_features, low_dimensional_features);

	for (index_t i=0; i<n_samples; ++i)
	{
		std::set<index_t> low_dimensional_neighbors = get_neighbors_indices(low_dimensional_dist, i, n_neighbors);
		check_similarity_of_sets(high_dimensional_neighbors_for_vectors[i], low_dimensional_neighbors, required_similarity_level);
	}

	SG_UNREF(isoEmbedder);
	SG_UNREF(high_dimensional_dist);
	SG_UNREF(low_dimensional_dist);
}

std::set<index_t> get_neighbors_indices(CDistance* distance_object, index_t feature_vector_index, index_t n_neighbors)
{
	index_t n_vectors = distance_object->get_num_vec_lhs();
	EXPECT_EQ(n_vectors, distance_object->get_num_vec_rhs());
	EXPECT_LE(n_neighbors, n_vectors - 1) <<
	"Number of neigbors can not be greater than total number of vectors minus 1";
	EXPECT_LE(feature_vector_index, n_vectors - 1);

	std::vector<index_and_distance_struct> distances_and_indices;

	for (index_t j = 0; j<n_vectors; ++j)
	{
		if (j == feature_vector_index)
			/* To avoid adding itself to the neighbors list */
			continue;
		index_and_distance_struct current;
		current.distance = distance_object->distance(feature_vector_index, j);
		current.neighbor_index = j;
		distances_and_indices.push_back(current);
	}

	/* Heapify, and then extract n_neighbors nearest neighbors*/
	std::make_heap(distances_and_indices.begin(), distances_and_indices.end(), comparator);
	std::set<index_t> neighbors_for_current_vector;
	for (index_t j = 0; j < n_neighbors; ++j)
	{
		neighbors_for_current_vector.insert(distances_and_indices[0].neighbor_index);
		std::pop_heap(distances_and_indices.begin(), distances_and_indices.end(), comparator);
		distances_and_indices.pop_back();
	}
	return neighbors_for_current_vector;
}

void check_similarity_of_sets(const std::set<index_t>& first_set,const std::set<index_t>& second_set, float64_t min_similarity_level)
{
	size_t total_elements_count = first_set.size();
	ASSERT_EQ(total_elements_count, second_set.size()) << "Can not compare sets of different size.";
	ASSERT_LE(min_similarity_level, 1.0) << "Similarity level can not be greater than 1.";
	ASSERT_GE(min_similarity_level, 0) << "Similarity level can not be less than 0.";
	if (min_similarity_level == 0)
		/*Nothing to do*/
		return;
	index_t similar_elements_count = 0;
	std::set<index_t>::iterator first_iter = first_set.begin(), second_iter = second_set.begin();
	while (first_iter != first_set.end() && second_iter != second_set.end())
	{
		if (*first_iter < *second_iter)
			++first_iter;
		else if (*second_iter < *first_iter)
			++second_iter;
		else
		{
			++similar_elements_count; ++first_iter; ++second_iter;
		}
	}
	EXPECT_GE((float64_t) similar_elements_count /(float64_t) total_elements_count, min_similarity_level) <<
	"#similarElements/#total < minimal similarity level.";
}

void fill_matrix_with_test_data(SGMatrix<float64_t>& matrix_to_fill)
{
	index_t num_cols = matrix_to_fill.num_cols, num_rows = matrix_to_fill.num_rows;
	for (index_t i = 0; i < num_cols; ++i)
	{
		for (index_t j = 0; j < num_rows - 1; ++j)
		{
			matrix_to_fill(j, i) = i;
		}
		matrix_to_fill(num_rows - 1, i) = CMath::randn_double();
	}
}

#endif
