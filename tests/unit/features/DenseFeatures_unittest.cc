/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saurabh Mahindre, Soumyajit De, Thoralf Klein, Heiko Strathmann,
 *          Viktor Gal
 */

#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <shogun/features/DenseFeatures.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/lib/View.h>

#include <random>

namespace shogun
{

class DenseFeaturesMock : public DenseFeatures<float64_t>
{
public:
	DenseFeaturesMock(SGMatrix<float64_t> data) : DenseFeatures<float64_t>(data)
	{
	}

	void copy_feature_matrix_public(SGMatrix<float64_t> target, index_t column_offset)
	{
		copy_feature_matrix(target, column_offset);
	}
};

}

using namespace shogun;

TEST(DenseFeaturesTest,create_merged_copy)
{
	/* create two matrices, feature objects for them, call create_merged_copy,
	 * and check if it worked */
	int32_t seed = 10;
	index_t n_1=3;
	index_t n_2=4;
	index_t dim=2;

	SGMatrix<float64_t> data_1(dim,n_1);
	for (index_t i=0; i<dim*n_1; ++i)
		data_1.matrix[i]=i;

	//data_1.display_matrix("data_1");
	std::mt19937_64 prng(seed);
	NormalDistribution<float64_t> normal_dist;
	SGMatrix<float64_t> data_2(dim,n_2);
	for (index_t i=0; i<dim*n_2; ++i)
		data_2.matrix[i]=normal_dist(prng);

	//data_2.display_matrix("data_2");

	auto features_1=std::make_shared<DenseFeatures<float64_t>>(data_1);
	auto features_2=std::make_shared<DenseFeatures<float64_t>>(data_2);

	auto concatenation=features_1->create_merged_copy(features_2);

	SGMatrix<float64_t> concat_data=
			concatenation->as<DenseFeatures<float64_t>>()->get_feature_matrix();
	//concat_data.display_matrix("concat_data");

	/* check for equality with data_1 */
	for (index_t i=0; i<dim*n_1; ++i)
		EXPECT_EQ(data_1.matrix[i], concat_data.matrix[i]);

	/* check for equality with data_2 */
	for (index_t i=0; i<dim*n_2; ++i)
		EXPECT_NEAR(data_2.matrix[i], concat_data.matrix[n_1*dim+i], 1E-15);




}

TEST(DenseFeaturesTest, create_merged_copy_with_subsets)
{
	const index_t n_1=10;
	const index_t n_2=15;
	const index_t dim=2;
	std::mt19937_64 prng(57);

	SGMatrix<float64_t> data_1(dim, n_1);
	std::iota(data_1.matrix, data_1.matrix + data_1.size(), 1);

	SGMatrix<float64_t> data_2(dim, n_2);
	std::iota(data_2.matrix, data_2.matrix + data_2.size(), data_2.size());

	auto features_1=std::make_shared<DenseFeatures<float64_t> >(data_1);
	auto features_2=std::make_shared<DenseFeatures<float64_t> >(data_2);

	SGVector<index_t> subset_1(n_1/2);
	random::fill_array(subset_1, 0, n_1-1, prng);
	features_1->add_subset(subset_1);
	auto active_data_1=features_1->get_feature_matrix();

	SGVector<index_t> subset_2(n_2/3);
	random::fill_array(subset_2, 0, n_2-1, prng);
	features_2->add_subset(subset_2);
	auto active_data_2=features_2->get_feature_matrix();

	SGMatrix<float64_t> expected_merged_mat(dim, active_data_1.num_cols+active_data_2.num_cols);
	std::copy(active_data_1.matrix, active_data_1.matrix+active_data_1.size(),
			expected_merged_mat.matrix);
	std::copy(active_data_2.matrix, active_data_2.matrix+active_data_2.size(),
			expected_merged_mat.matrix+active_data_1.size());

	auto merged=features_1->create_merged_copy(features_2)->as<DenseFeatures<float64_t>>();
	SGMatrix<float64_t> merged_mat = merged->get_feature_matrix();

	ASSERT_EQ(expected_merged_mat.num_rows, merged_mat.num_rows);
	ASSERT_EQ(expected_merged_mat.num_cols, merged_mat.num_cols);
	for (index_t j=0; j<expected_merged_mat.num_cols; ++j)
	{
		for (index_t i=0; i<expected_merged_mat.num_rows; ++i)
			EXPECT_NEAR(expected_merged_mat(i, j), merged_mat(i, j), 1E-15);
	}

}

TEST(DenseFeaturesTest, copy_dimension_subset)
{
	int32_t seed = 12;
	index_t dim=5;
	index_t n=10;

	SGMatrix<float64_t> data(dim, n);
	for (index_t i=0; i<dim*n; ++i)
		data.matrix[i]=i;

	auto features=std::make_shared<DenseFeatures<float64_t>>(data);

	SGVector<index_t> dims(dim/2);
	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	for (index_t i=0; i<dims.vlen; ++i)
		dims[i]=uniform_int_dist(prng, {0, dim-1});

	auto f_reduced=
		features->copy_dimension_subset(dims)->as<DenseFeatures<float64_t>>();

	SGMatrix<float64_t> data_reduced=f_reduced->get_feature_matrix();

	for (index_t i=0; i<data_reduced.num_rows; ++i)
	{
		for (index_t j=0; j<data_reduced.num_cols; ++j)
			EXPECT_NEAR(data(dims[i], j), data_reduced(i, j), 1E-16);
	}



}

TEST(DenseFeaturesTest, copy_dimension_subset_with_subsets)
{
	int32_t seed = 12;
	index_t dim=5;
	index_t n=10;

	SGMatrix<float64_t> data(dim, n);
	for (index_t i=0; i<dim*n; ++i)
		data.matrix[i]=i;

	auto features=std::make_shared<DenseFeatures<float64_t>>(data);

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	SGVector<index_t> inds(n/2);
	for (index_t i=0; i<inds.vlen; ++i)
		inds[i]=uniform_int_dist(prng, {0, n-1});

	features->add_subset(inds);

	SGVector<index_t> dims(dim/2);
	for (index_t i=0; i<dims.vlen; ++i)
		dims[i]=uniform_int_dist(prng, {0, dim-1});

	auto f_reduced=
		features->copy_dimension_subset(dims)->as<DenseFeatures<float64_t>>();

	SGMatrix<float64_t> data_reduced=f_reduced->get_feature_matrix();
	for (index_t i=0; i<data_reduced.num_rows; ++i)
	{
		for (index_t j=0; j<data_reduced.num_cols; ++j)
			EXPECT_NEAR(data(dims[i], inds[j]), data_reduced(i, j), 1E-16);
	}



}

TEST(DenseFeaturesTest, shallow_copy_subset_data)
{
	index_t dim=5;
	index_t n=10;
	std::mt19937_64 prng(57);

	SGMatrix<float64_t> data(dim, n);
	std::iota(data.data(), data.data()+data.size(), 1);
	SGVector<index_t> inds(n/2);
	random::fill_array(inds, 0, n-1, prng);

	auto features=std::make_shared<DenseFeatures<float64_t>>(data);
	features->add_subset(inds);
	auto features_copy = features->shallow_subset_copy();

	SGMatrix<float64_t> orig_matrix=features->get_feature_matrix();
	SGMatrix<float64_t> copy_matrix=features_copy->as<DenseFeatures<float64_t>>()->get_feature_matrix();


	for (index_t i=0; i<dim; ++i)
	{
		for (index_t j=0; j<inds.size(); ++j)
			EXPECT_NEAR(orig_matrix(i,j), copy_matrix(i,j), 1E-15);
	}

}

TEST(DenseFeaturesTest, copy_feature_matrix)
{
	index_t dim=5;
	index_t n=10;
	std::mt19937_64 prng(57);

	SGMatrix<float64_t> data(dim, n);
	std::iota(data.data(), data.data()+data.size(), 1);
	SGVector<index_t> inds(n/2);
	random::fill_array(inds, 0, n-1, prng);

	auto features=std::make_shared<DenseFeaturesMock>(data);
	features->add_subset(inds);

	index_t offset=3;
	SGMatrix<float64_t> copy(dim, inds.vlen+offset);
	auto data_ptr=copy.matrix;
	std::fill(copy.data(), copy.data()+copy.size(), 0);
	features->copy_feature_matrix_public(copy, offset);

	ASSERT_EQ(copy.num_rows, dim);
	ASSERT_EQ(copy.num_cols, inds.vlen+offset);
	ASSERT_EQ(copy.matrix, data_ptr);

	for (index_t j=0; j<offset; ++j)
	{
		for (index_t i=0; i<dim; ++i)
			EXPECT_NEAR(copy(i, j), 0, 1E-15);
	}

	for (index_t j=0; j<inds.vlen; ++j)
	{
		for (index_t i=0; i<dim; ++i)
			EXPECT_NEAR(copy(i, j+offset), data(i, inds[j]), 1E-15);
	}
}

TEST(DenseFeaturesTest, view)
{
	auto num_feats = 2;
	auto num_vectors = 4;
	SGMatrix<float64_t> data(num_feats, num_vectors);

	for (auto i : range(num_feats * num_vectors))
	{
		data[i] = i;
	}
	auto feats_original = std::make_shared<DenseFeatures<float64_t>>(data);

	SGVector<index_t> subset1{0, 2, 3};
	auto feats_subset1 = view(feats_original, subset1);
	ASSERT_EQ(feats_subset1->get_num_features(), num_feats);
	ASSERT_EQ(feats_subset1->get_num_vectors(), subset1.vlen);
	auto feature_matrix_subset1 = feats_subset1->get_feature_matrix();

	// check feature_matrix(i, j) == feature_matrix_original(i, subset(j))
	for (auto j : range(subset1.vlen))
	{
		for (auto i : range(num_feats))
			EXPECT_EQ(feature_matrix_subset1(i, j), data(i, subset1[j]));
	}

	SGVector<index_t> subset2{
	    0, 2}; // subset2 is column 0 & 3 of original features
	auto feats_subset2 = view(feats_subset1, subset2);
	ASSERT_EQ(feats_subset2->get_num_features(), num_feats);
	ASSERT_EQ(feats_subset2->get_num_vectors(), subset2.vlen);
	auto feature_matrix_subset2 = feats_subset2->get_feature_matrix();

	for (auto j : range(subset2.size()))
	{
		for (auto i : range(num_feats))
			EXPECT_EQ(
			    feature_matrix_subset2(i, j), data(i, subset1[subset2[j]]));
	}
}
