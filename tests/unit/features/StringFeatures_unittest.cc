/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Heiko Strathmann, Soeren Sonnenburg,
 *          Leon Kuchenbecker
 */

#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/memory.h>
#include <random>

using namespace shogun;

TEST(StringFeaturesTest,copy_subset)
{
	std::mt19937_64 prng(25);
	std::vector<SGVector<char>> strings = generateRandomStringData(prng);

	/* create num_feautres 2-dimensional vectors */
	auto f=std::make_shared<StringFeatures<char>>(strings, ALPHANUM);

	index_t offset_subset=1;
	SGVector<index_t> feature_subset(8);
	feature_subset.range_fill(offset_subset);
	//feature_subset.display_vector("feature subset");

	f->add_subset(feature_subset);
	//SG_PRINT("feature vectors after setting subset on original data:\n");
	for (index_t i=0; i<f->get_num_vectors(); ++i)
	{
		SGVector<char> vec=f->get_feature_vector(i);

		//SG_PRINT("%i: ", i);
		//for (index_t j=0; j<vec.vlen; ++j)
		//	SG_PRINT("%c", vec.vector[j]);

		//SG_PRINT("\n");

		f->free_feature_vector(vec, i);
	}

	index_t offset_copy=2;
	SGVector<index_t> feature_copy_subset(4);
	feature_copy_subset.range_fill(offset_copy);
	//feature_copy_subset.display_vector("indices that are to be copied");

	auto subset_copy=f->copy_subset(
			feature_copy_subset)->as<StringFeatures<char>>();

	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGVector<char> vec=subset_copy->get_feature_vector(i);

		//SG_PRINT("%i: ", i);
		//for (index_t j=0; j<vec.vlen; ++j)
		//	SG_PRINT("%c", vec.vector[j]);

		//SG_PRINT("\n");

		subset_copy->free_feature_vector(vec, i);
	}

	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGVector<char> vec=subset_copy->get_feature_vector(i);

		for (index_t j=0; j<vec.vlen; ++j)
		{
			index_t offset_idx=i+(offset_copy+offset_subset);
			EXPECT_EQ(vec.vector[j], strings[offset_idx].vector[j]);
		}

		subset_copy->free_feature_vector(vec, i);
	}



}

TEST(StringFeaturesTest,equals)
{
	std::mt19937_64 prng(25);
	std::vector<SGVector<char>> strings = generateRandomStringData(prng);

	auto f=std::make_shared<StringFeatures<char>>(strings, ALPHANUM);
	auto f_clone = f->clone()->as<StringFeatures<char>>();
	EXPECT_EQ(f->equals(f_clone), true);



}
