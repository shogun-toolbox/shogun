/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Soumyajit De
 */
 
#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGSparseVector, dense_dot_complex64_float64)
{
	const int32_t size=10;
	const int32_t num_feat=size/2;

	SGSparseVector<complex64_t> vec(num_feat);
	for (index_t i=0; i<vec.num_feat_entries; ++i)
	{
		SGSparseVectorEntry<complex64_t> entry;
		entry.feat_index=(i+1)*2;
		entry.entry=complex64_t(0.5, 0.75);
		vec.features[i]=entry;
	}

	SGVector<float64_t> v(size);
	v.set_const(1.5);

	complex64_t dot=vec.dense_dot(v);
	EXPECT_NEAR(dot.real(), 3.0, 1E-19);
	EXPECT_NEAR(dot.imag(), 4.5, 1E-19);
}

TEST(SGSparseVector, dense_dot_complex64_int32)
{
	const int32_t size=10;
	const int32_t num_feat=size/2;

	SGSparseVector<complex64_t> vec(num_feat);
	for (index_t i=0; i<vec.num_feat_entries; ++i)
	{
		SGSparseVectorEntry<complex64_t> entry;
		entry.feat_index=(i+1)*2;
		entry.entry=complex64_t(0.5, 0.75);
		vec.features[i]=entry;
	}

	SGVector<int32_t> v(size);
	v.set_const(1);

	complex64_t dot=vec.dense_dot(v);
	EXPECT_NEAR(dot.real(), 2.0, 1E-19);
	EXPECT_NEAR(dot.imag(), 3.0, 1E-19);
}

TEST(SGSparseVector, dense_dot_float64_int32)
{
	const int32_t size=10;
	const int32_t num_feat=size/2;

	SGSparseVector<float64_t> vec(num_feat);
	for (index_t i=0; i<vec.num_feat_entries; ++i)
	{
		SGSparseVectorEntry<float64_t> entry;
		entry.feat_index=(i+1)*2;
		entry.entry=0.5;
		vec.features[i]=entry;
	}

	SGVector<int32_t> v(size);
	v.set_const(1);

	float64_t dot=vec.dense_dot(v);
	EXPECT_NEAR(dot, 2.0, 1E-19);
}

TEST(SGSparseVector, get_feature_unique)
{
	SGSparseVector<float64_t> vec(3);
	EXPECT_EQ(vec.num_feat_entries, 3);

	vec.features[0].feat_index = 1;
	vec.features[0].entry = -3.0;

	vec.features[1].feat_index = 2;
	vec.features[1].entry = -4.0;

	vec.features[2].feat_index = 3;
	vec.features[2].entry = -6.0;

	EXPECT_NEAR(vec.get_feature(1), -3.0, 1E-19);
	EXPECT_NEAR(vec.get_feature(2), -4.0, 1E-19);
	EXPECT_NEAR(vec.get_feature(3), -6.0, 1E-19);
}

TEST(SGSparseVector, get_feature_duplicate)
{
	SGSparseVector<float64_t> vec(3);
	EXPECT_EQ(vec.num_feat_entries, 3);

	vec.features[0].feat_index = 3;
	vec.features[0].entry = -3.0;

	vec.features[1].feat_index = 2;
	vec.features[1].entry = -4.0;

	vec.features[2].feat_index = 3;
	vec.features[2].entry = -6.0;

	EXPECT_NEAR(vec.get_feature(2), -4.0, 1E-19);
	EXPECT_NEAR(vec.get_feature(3), -9.0, 1E-19);
}

TEST(SGSparseVector, sort_features_empty)
{
	SGSparseVector<float64_t> vec(0);
	EXPECT_EQ(vec.num_feat_entries, 0);

	vec.sort_features();
	EXPECT_EQ(vec.num_feat_entries, 0);
}

TEST(SGSparseVector, sort_features_one)
{
	SGSparseVector<float64_t> vec(1);
	EXPECT_EQ(vec.num_feat_entries, 1);

	vec.features[0].feat_index = 1;
	vec.features[0].entry = +1.0;

	vec.sort_features();
	EXPECT_EQ(vec.num_feat_entries, 1);

	EXPECT_EQ(vec.features[0].feat_index, 1);
	EXPECT_NEAR(vec.features[0].entry, +1.0, 1E-19);
}

TEST(SGSparseVector, sort_features_duplicate)
{
	SGSparseVector<float64_t> vec(4);
	EXPECT_EQ(vec.num_feat_entries, 4);

	vec.features[0].feat_index = 3;
	vec.features[0].entry = -3.0;

	vec.features[1].feat_index = 2;
	vec.features[1].entry = -4.0;

	vec.features[2].feat_index = 3;
	vec.features[2].entry = -6.0;

	vec.features[3].feat_index = 4;
	vec.features[3].entry = -16.0;

	EXPECT_NEAR(vec.get_feature(2), -4.0, 1E-19);
	EXPECT_NEAR(vec.get_feature(3), -9.0, 1E-19);
	EXPECT_NEAR(vec.get_feature(4), -16.0, 1E-19);

	vec.sort_features();
	EXPECT_EQ(vec.num_feat_entries, 3);

	EXPECT_EQ(vec.features[0].feat_index, 2);
	EXPECT_NEAR(vec.features[0].entry, -4.0, 1E-19);

	EXPECT_EQ(vec.features[1].feat_index, 3);
	EXPECT_NEAR(vec.features[1].entry, -9.0, 1E-19);

	EXPECT_EQ(vec.features[2].feat_index, 4);
	EXPECT_NEAR(vec.features[2].entry, -16.0, 1E-19);

	EXPECT_NEAR(vec.get_feature(2), -4.0, 1E-19);
	EXPECT_NEAR(vec.get_feature(3), -9.0, 1E-19);
	EXPECT_NEAR(vec.get_feature(4), -16.0, 1E-19);
}
