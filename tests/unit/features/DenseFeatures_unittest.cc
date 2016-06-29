/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>

#ifdef HAVE_CXX11
#include <numeric>
#endif

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
		EXPECT_EQ(data_2.matrix[i], concat_data.matrix[n_1*dim+i]);

	SG_UNREF(concatenation);
	SG_UNREF(features_1);
	SG_UNREF(features_2);
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
#ifdef HAVE_CXX11
	std::iota(data.data(), data.data()+data.size(), 1);
#else
	for (index_t i=0; i<dim*n; ++i)
		data.matrix[i]=i+1;
#endif
	SGVector<index_t> inds(n/2);
	inds.random(0, n-1);

	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(data);
	features->add_subset(inds);
	CFeatures* features_copy = features->shallow_subset_copy();

	SGMatrix<float64_t> orig_matrix=features->get_feature_matrix();
	SGMatrix<float64_t> copy_matrix=static_cast<CDenseFeatures<float64_t>*>(features_copy)->get_feature_matrix();


	for (index_t i=0; i<dim; ++i)
		for (index_t j=0; j<inds.size(); ++j)
			EXPECT_EQ(orig_matrix(i,j), copy_matrix(i,j));

	SG_UNREF(features_copy);
	SG_UNREF(features);
}
