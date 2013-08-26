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

#include <vector>

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

TEST(SGSparseVector, get_dense)
{
	SGSparseVector<float64_t> vec(3);
	EXPECT_EQ(vec.num_feat_entries, 3);

	vec.features[0].feat_index = 3;
	vec.features[0].entry = -3.0;

	vec.features[1].feat_index = 2;
	vec.features[1].entry = -4.0;

	vec.features[2].feat_index = 3;
	vec.features[2].entry = -6.0;

	SGVector<float64_t> dense = vec.get_dense(5);
	EXPECT_EQ(dense.size(), 5);

	EXPECT_NEAR(dense[0], 0.0, 1E-19);
	EXPECT_NEAR(dense[1], 0.0, 1E-19);
	EXPECT_NEAR(dense[2], -4.0, 1E-19);
	EXPECT_NEAR(dense[3], -9.0, 1E-19);
	EXPECT_NEAR(dense[4], 0.0, 1E-19);
}

TEST(SGSparseVector, get_dense_shortest)
{
	SGSparseVector<float64_t> vec(3);
	EXPECT_EQ(vec.num_feat_entries, 3);

	vec.features[0].feat_index = 3;
	vec.features[0].entry = -3.0;

	vec.features[1].feat_index = 2;
	vec.features[1].entry = -4.0;

	vec.features[2].feat_index = 3;
	vec.features[2].entry = -6.0;

	SGVector<float64_t> dense = vec.get_dense();
	EXPECT_EQ(dense.size(), 4);

	EXPECT_NEAR(dense[0], 0.0, 1E-19);
	EXPECT_NEAR(dense[1], 0.0, 1E-19);
	EXPECT_NEAR(dense[2], -4.0, 1E-19);
	EXPECT_NEAR(dense[3], -9.0, 1E-19);
}

TEST(SGSparseVector, sort_features_deduplicate_no_realloc)
{
        const int32_t vlen = 1024*1024;

	SGSparseVector<float64_t> vec(vlen);
	EXPECT_EQ(vlen, vec.num_feat_entries);

	for (int32_t i=0; i<vlen; i++) {
	        vec.features[i].feat_index = 0;
	        vec.features[i].entry = 0.0;
	}

	SGSparseVectorEntry<float64_t>* features_ptr = vec.features;

	vec.sort_features(true);

	EXPECT_EQ(vec.num_feat_entries, 0);
	EXPECT_EQ(vec.features, features_ptr);
}

TEST(SGSparseVector, sort_features_deduplicate_realloc)
{
        const int32_t vlen = 1024*1024;

	SGSparseVector<float64_t> vec(vlen);
	EXPECT_EQ(vlen, vec.num_feat_entries);

	for (int32_t i=0; i<vlen; i++) {
	        vec.features[i].feat_index = 0;
	        vec.features[i].entry = 0.0;
	}

	SGSparseVectorEntry<float64_t>* features_ptr = vec.features;

	vec.sort_features(false);

	EXPECT_EQ(vec.num_feat_entries, 0);
	EXPECT_NE(vec.features, features_ptr);
}

std::vector< std::pair< SGSparseVector<float64_t>,SGSparseVector<float64_t> > >
create_sort_features_mock_vectors()
{
	std::vector< std::pair< SGSparseVector<float64_t>,SGSparseVector<float64_t> > > test_cases;

	{
		SGSparseVector<float64_t> before(1);

		before.features[0].feat_index = 1;
		before.features[0].entry = +1.0;

		SGSparseVector<float64_t> after(1);

		after.features[0].feat_index = 1;
		after.features[0].entry = +1.0;

		test_cases.push_back(std::make_pair(before, after));
	}

	{
		SGSparseVector<float64_t> before(1);

		before.features[0].feat_index = 1;
		before.features[0].entry = 0.0;

		SGSparseVector<float64_t> after(0);

		test_cases.push_back(std::make_pair(before, after));
	}

	{
		SGSparseVector<float64_t> before(8);

		before.features[0].feat_index = 1;
		before.features[0].entry = +0.0;
		before.features[1].feat_index = 3;
		before.features[1].entry = +1.0;
		before.features[2].feat_index = 5;
		before.features[2].entry = +0.0;
		before.features[3].feat_index = 4;
		before.features[3].entry = +0.0;
		before.features[4].feat_index = 3;
		before.features[4].entry = -1.0;
		before.features[5].feat_index = 3;
		before.features[5].entry = +0.0;
		before.features[6].feat_index = 1;
		before.features[6].entry = +1.0;
		before.features[7].feat_index = 1;
		before.features[7].entry = +0.0;

		SGSparseVector<float64_t> after(1);

		after.features[0].feat_index = 1;
		after.features[0].entry = +1.0;

		test_cases.push_back(std::make_pair(before, after));
	}

	{
		SGSparseVector<float64_t> before(5);

		before.features[0].feat_index = 3;
		before.features[0].entry = -3.0;
		before.features[1].feat_index = 2;
		before.features[1].entry = -4.0;
		before.features[2].feat_index = 3;
		before.features[2].entry = -6.0;
		before.features[3].feat_index = 4;
		before.features[3].entry = -16.0;
		before.features[4].feat_index = 1;
		before.features[4].entry = 0.0;

		SGSparseVector<float64_t> after(3);

		after.features[0].feat_index = 2;
		after.features[0].entry      = -4.0;
		after.features[1].feat_index = 3;
		after.features[1].entry      = -9.0;
		after.features[2].feat_index = 4;
		after.features[2].entry      = -16.0;

		test_cases.push_back(std::make_pair(before, after));
	}

	{
		// before=[3:3 2:2]
		// after=[3:6]
		// expected=[2:2 3:3]

		SGSparseVector<float64_t> before(2);

		before.features[0].feat_index = 3;
		before.features[0].entry      = 3;
		before.features[1].feat_index = 2;
		before.features[1].entry      = 2;

		SGSparseVector<float64_t> after(2);

		after.features[0].feat_index = 2;
		after.features[0].entry      = 2;
		after.features[1].feat_index = 3;
		after.features[1].entry      = 3;

		test_cases.push_back(std::make_pair(before, after));
	}

	{
		// before=[3:3 2:2 1:1]
		// after=[3:3 2:2 3:3]
		// expected=[1:1 2:2 3:3]

		SGSparseVector<float64_t> before(3);

		before.features[0].feat_index = 3;
		before.features[0].entry      = 3;
		before.features[1].feat_index = 2;
		before.features[1].entry      = 2;
		before.features[2].feat_index = 1;
		before.features[2].entry      = 1;

		SGSparseVector<float64_t> after(3);

		after.features[0].feat_index = 1;
		after.features[0].entry      = 1;
		after.features[1].feat_index = 2;
		after.features[1].entry      = 2;
		after.features[2].feat_index = 3;
		after.features[2].entry      = 3;

		test_cases.push_back(std::make_pair(before, after));
	}

	return test_cases;
}

TEST(SGSparseVector, clone_loop)
{
	std::vector< std::pair< SGSparseVector<float64_t>,SGSparseVector<float64_t> > > test_cases = create_sort_features_mock_vectors();

	for (uint32_t i=0; i < test_cases.size(); i++) {
		std::pair< SGSparseVector<float64_t>,SGSparseVector<float64_t> > test_case = test_cases[i];

		SGSparseVector<float64_t> expected = test_case.first;
		SGSparseVector<float64_t> result   = expected.clone();

		EXPECT_EQ(expected.num_feat_entries, result.num_feat_entries);
		EXPECT_TRUE(NULL != result.features);
		EXPECT_NE(expected.features, result.features);

		ASSERT_EQ(expected.num_feat_entries, result.num_feat_entries);
		for (int32_t idx=0; idx<expected.num_feat_entries; idx++) {
			 SGSparseVectorEntry<float64_t> rfeat = result.features[idx];
			 SGSparseVectorEntry<float64_t> efeat = expected.features[idx];

			 EXPECT_EQ(efeat.feat_index, rfeat.feat_index);
			 EXPECT_NEAR(efeat.entry, rfeat.entry, 1E-19);
		}

		EXPECT_EQ(expected.get_num_dimensions(), result.get_num_dimensions());
		for (int32_t fidx=0; fidx<expected.get_num_dimensions()+1; fidx++) {
			 EXPECT_NEAR(
				 expected.get_feature(fidx),
				 result.get_feature(fidx),
				 1E-19);
		}
	}
}

TEST(SGSparseVector, sort_features_loop)
{
	// testing with and without realloc to be sure
	for (int32_t r=0; r<2; r++) {
		std::vector< std::pair< SGSparseVector<float64_t>,SGSparseVector<float64_t> > > test_cases = create_sort_features_mock_vectors();
		bool stable_pointer = (r==1);

		for (uint32_t i=0; i < test_cases.size(); i++) {
			std::pair< SGSparseVector<float64_t>,SGSparseVector<float64_t> > test_case = test_cases[i];

			SGSparseVector<float64_t> result = test_case.first.clone();
			SGSparseVector<float64_t> expected = test_case.second; // .clone();

			const SGSparseVectorEntry<float64_t>* fptr = result.features;
			result.sort_features(stable_pointer);

			// we really rely that the pointers don't change
			if (stable_pointer) {
				ASSERT_EQ(fptr, result.features);
			}

			ASSERT_EQ(expected.num_feat_entries, result.num_feat_entries);
			for (int32_t idx=0; idx<result.num_feat_entries; idx++) {
				SGSparseVectorEntry<float64_t> vfeat = result.features[idx];
				SGSparseVectorEntry<float64_t> efeat = expected.features[idx];

				EXPECT_EQ(efeat.feat_index, vfeat.feat_index);
				EXPECT_NEAR(efeat.entry, vfeat.entry, 1E-19);
			}

			EXPECT_EQ(expected.get_num_dimensions(), result.get_num_dimensions());
			for (int32_t fidx=0; fidx<result.get_num_dimensions()+1; fidx++) {
				EXPECT_NEAR(
					expected.get_feature(fidx),
					result.get_feature(fidx),
					1E-19);
			}
		}
	}
}
