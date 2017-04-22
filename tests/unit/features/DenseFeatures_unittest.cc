/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <numeric>
#include <algorithm>
#include <shogun/base/some.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>

namespace shogun
{

class CDenseFeaturesMock : public CDenseFeatures<float64_t>
{
public:
	CDenseFeaturesMock(SGMatrix<float64_t> data) : CDenseFeatures<float64_t>(data)
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

	index_t n_1=3;
	index_t n_2=4;
	index_t dim=2;

	SGMatrix<float64_t> data_1(dim,n_1);
	for (index_t i=0; i<dim*n_1; ++i)
		data_1.matrix[i]=i;

	//data_1.display_matrix("data_1");

	SGMatrix<float64_t> data_2(dim,n_2);
	for (index_t i=0; i<dim*n_2; ++i)
		data_2.matrix[i]=CMath::randn_double();

	//data_2.display_matrix("data_2");

	CDenseFeatures<float64_t>* features_1=new CDenseFeatures<float64_t>(data_1);
	CDenseFeatures<float64_t>* features_2=new CDenseFeatures<float64_t>(data_2);

	CFeatures* concatenation=features_1->create_merged_copy(features_2);

	SGMatrix<float64_t> concat_data=
			((CDenseFeatures<float64_t>*)concatenation)->get_feature_matrix();
	//concat_data.display_matrix("concat_data");

	/* check for equality with data_1 */
	for (index_t i=0; i<dim*n_1; ++i)
		EXPECT_EQ(data_1.matrix[i], concat_data.matrix[i]);

	/* check for equality with data_2 */
	for (index_t i=0; i<dim*n_2; ++i)
		EXPECT_NEAR(data_2.matrix[i], concat_data.matrix[n_1*dim+i], 1E-15);

	SG_UNREF(concatenation);
	SG_UNREF(features_1);
	SG_UNREF(features_2);
}

TEST(DenseFeaturesTest, create_merged_copy_with_subsets)
{
	const index_t n_1=10;
	const index_t n_2=15;
	const index_t dim=2;

	SGMatrix<float64_t> data_1(dim, n_1);
	std::iota(data_1.matrix, data_1.matrix + data_1.size(), 1);

	SGMatrix<float64_t> data_2(dim, n_2);
	std::iota(data_2.matrix, data_2.matrix + data_2.size(), data_2.size());

	auto features_1=some<CDenseFeatures<float64_t> >(data_1);
	auto features_2=some<CDenseFeatures<float64_t> >(data_2);

	SGVector<index_t> subset_1(n_1/2);
	subset_1.random(0, n_1-1);
	features_1->add_subset(subset_1);
	auto active_data_1=features_1->get_feature_matrix();

	SGVector<index_t> subset_2(n_2/3);
	subset_2.random(0, n_2-1);
	features_2->add_subset(subset_2);
	auto active_data_2=features_2->get_feature_matrix();

	SGMatrix<float64_t> expected_merged_mat(dim, active_data_1.num_cols+active_data_2.num_cols);
	std::copy(active_data_1.matrix, active_data_1.matrix+active_data_1.size(),
			expected_merged_mat.matrix);
	std::copy(active_data_2.matrix, active_data_2.matrix+active_data_2.size(),
			expected_merged_mat.matrix+active_data_1.size());

	auto merged=static_cast<CDenseFeatures<float64_t>*>(features_1->create_merged_copy(features_2));
	SGMatrix<float64_t> merged_mat=merged->steal_feature_matrix();

	ASSERT_EQ(expected_merged_mat.num_rows, merged_mat.num_rows);
	ASSERT_EQ(expected_merged_mat.num_cols, merged_mat.num_cols);
	for (index_t j=0; j<expected_merged_mat.num_cols; ++j)
	{
		for (index_t i=0; i<expected_merged_mat.num_rows; ++i)
			EXPECT_NEAR(expected_merged_mat(i, j), merged_mat(i, j), 1E-15);
	}

	SG_UNREF(merged);
}

TEST(DenseFeaturesTest, copy_dimension_subset)
{
	index_t dim=5;
	index_t n=10;

	SGMatrix<float64_t> data(dim, n);
	for (index_t i=0; i<dim*n; ++i)
		data.matrix[i]=i;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	SGVector<index_t> dims(dim/2);
	for (index_t i=0; i<dims.vlen; ++i)
		dims[i]=CMath::random(0, dim-1);

	CDenseFeatures<float64_t>* f_reduced=(CDenseFeatures<float64_t>*)
		features->copy_dimension_subset(dims);

	SGMatrix<float64_t> data_reduced=f_reduced->get_feature_matrix();

	for (index_t i=0; i<data_reduced.num_rows; ++i)
	{
		for (index_t j=0; j<data_reduced.num_cols; ++j)
			EXPECT_NEAR(data(dims[i], j), data_reduced(i, j), 1E-16);
	}

	SG_UNREF(features);
	SG_UNREF(f_reduced);
}

TEST(DenseFeaturesTest, copy_dimension_subset_with_subsets)
{
	index_t dim=5;
	index_t n=10;

	SGMatrix<float64_t> data(dim, n);
	for (index_t i=0; i<dim*n; ++i)
		data.matrix[i]=i;

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);

	SGVector<index_t> inds(n/2);
	for (index_t i=0; i<inds.vlen; ++i)
		inds[i]=CMath::random(0, n-1);

	features->add_subset(inds);

	SGVector<index_t> dims(dim/2);
	for (index_t i=0; i<dims.vlen; ++i)
		dims[i]=CMath::random(0, dim-1);

	CDenseFeatures<float64_t>* f_reduced=(CDenseFeatures<float64_t>*)
		features->copy_dimension_subset(dims);

	SGMatrix<float64_t> data_reduced=f_reduced->get_feature_matrix();
	for (index_t i=0; i<data_reduced.num_rows; ++i)
	{
		for (index_t j=0; j<data_reduced.num_cols; ++j)
			EXPECT_NEAR(data(dims[i], inds[j]), data_reduced(i, j), 1E-16);
	}

	SG_UNREF(features);
	SG_UNREF(f_reduced);
}

TEST(DenseFeaturesTest, shallow_copy_subset_data)
{
	index_t dim=5;
	index_t n=10;

	SGMatrix<float64_t> data(dim, n);
	std::iota(data.data(), data.data()+data.size(), 1);
	SGVector<index_t> inds(n/2);
	inds.random(0, n-1);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	features->add_subset(inds);
	CFeatures* features_copy = features->shallow_subset_copy();

	SGMatrix<float64_t> orig_matrix=features->get_feature_matrix();
	SGMatrix<float64_t> copy_matrix=static_cast<CDenseFeatures<float64_t>*>(features_copy)->get_feature_matrix();


	for (index_t i=0; i<dim; ++i)
	{
		for (index_t j=0; j<inds.size(); ++j)
			EXPECT_NEAR(orig_matrix(i,j), copy_matrix(i,j), 1E-15);
	}

	SG_UNREF(features_copy);
	SG_UNREF(features);
}

TEST(DenseFeaturesTest, copy_feature_matrix)
{
	index_t dim=5;
	index_t n=10;

	SGMatrix<float64_t> data(dim, n);
	std::iota(data.data(), data.data()+data.size(), 1);
	SGVector<index_t> inds(n/2);
	inds.random(0, n-1);

	auto features=some<CDenseFeaturesMock>(data);
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
