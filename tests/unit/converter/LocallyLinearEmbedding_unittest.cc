#include <vector>
#include <set>
#include <algorithm> /* heap operations, std::sort */
#include <iostream>

#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <gtest/gtest.h>

using namespace shogun;

#ifdef HAVE_EIGEN3

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

void check_similarity_of_sets(const std::set<index_t>& first_set,const std::set<index_t>& second_set, float min_similarity_level);

TEST(LocallyLinearEmbeddingTest,neighbors_preserving)
{
	const index_t n_samples = 100;
	const index_t n_gaussians = 1;
	const index_t n_dimensions = 4;
	const index_t n_target_dimensions = 3;
	const index_t n_neighbors = 40;
	const float required_similarity_level = 0.5; /*hope we will get rid of this*/
	CDenseFeatures<float64_t>* high_dimensional_features = 
		new CDenseFeatures<float64_t>(CDataGenerator::generate_gaussians(n_samples, n_gaussians, n_dimensions)); 
	
	CDistance* high_dimensional_dist = 
		new CEuclideanDistance(high_dimensional_features, high_dimensional_features);

	std::vector<std::set<index_t> > high_dimensional_neighbors_for_vectors;
	/* Find n_neighbors nearest eighbours for each vector */
	for (index_t i=0; i<n_samples; ++i)
	{
		high_dimensional_neighbors_for_vectors.push_back(get_neighbors_indices(high_dimensional_dist, i, n_neighbors));
	}

	CLocallyLinearEmbedding* lleEmbedder =
		new CLocallyLinearEmbedding();
	lleEmbedder->set_k(n_neighbors);

	lleEmbedder->set_target_dim(n_target_dimensions);
	EXPECT_EQ(n_target_dimensions, lleEmbedder->get_target_dim());

	CDenseFeatures<float64_t>* low_dimensional_features = 
		lleEmbedder->embed(high_dimensional_features);

	EXPECT_EQ(n_target_dimensions,low_dimensional_features->get_dim_feature_space());
	EXPECT_EQ(high_dimensional_features->get_num_vectors(),low_dimensional_features->get_num_vectors());

	CDistance* low_dimensional_dist =
		new CEuclideanDistance(low_dimensional_features, low_dimensional_features);
	
	for (index_t i=0; i<n_samples; ++i) 
	{
		std::set<index_t> low_dimensional_neighbors = get_neighbors_indices(low_dimensional_dist, i, n_neighbors);
		check_similarity_of_sets(high_dimensional_neighbors_for_vectors[i], low_dimensional_neighbors, required_similarity_level);
	}

	SG_UNREF(lleEmbedder);
	SG_UNREF(high_dimensional_dist);
	SG_UNREF(low_dimensional_dist);
}

std::set<index_t> get_neighbors_indices(CDistance* distance_object, index_t feature_vector_index, index_t n_neighbors)
{
	index_t n_vectors = distance_object->get_num_vec_lhs();
	EXPECT_EQ(n_vectors, distance_object->get_num_vec_rhs());
	EXPECT_LE(n_neighbors, n_vectors - 1) << "Number of neigbors can not be greater than total number of vectors minus 1";
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

void check_similarity_of_sets(const std::set<index_t>& first_set,const std::set<index_t>& second_set, float min_similarity_level)
{
	index_t total_elements_count = first_set.size();
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
	EXPECT_GE((float) similar_elements_count /(float) total_elements_count, min_similarity_level)<<"#similarElements/#total < minimal similarity level.";
}
#endif
