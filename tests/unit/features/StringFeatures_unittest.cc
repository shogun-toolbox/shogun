/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Heiko Strathmann, Soeren Sonnenburg, 
 *          Leon Kuchenbecker
 */

#include "utils/Utils.h"
#include <gtest/gtest.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/memory.h>

using namespace shogun;

TEST(StringFeaturesTest,copy_subset)
{
	SGStringList<char> strings = generateRandomStringData();

	/* create num_feautres 2-dimensional vectors */
	CStringFeatures<char>* f=new CStringFeatures<char>(strings, ALPHANUM);

	index_t offset_subset=1;
	SGVector<index_t> feature_subset(8);
	feature_subset.range_fill(offset_subset);
	//feature_subset.display_vector("feature subset");

	f->add_subset(feature_subset);
	//SG_SPRINT("feature vectors after setting subset on original data:\n");
	for (index_t i=0; i<f->get_num_vectors(); ++i)
	{
		SGVector<char> vec=f->get_feature_vector(i);

		//SG_SPRINT("%i: ", i);
		//for (index_t j=0; j<vec.vlen; ++j)
		//	SG_SPRINT("%c", vec.vector[j]);

		//SG_SPRINT("\n");

		f->free_feature_vector(vec, i);
	}

	index_t offset_copy=2;
	SGVector<index_t> feature_copy_subset(4);
	feature_copy_subset.range_fill(offset_copy);
	//feature_copy_subset.display_vector("indices that are to be copied");

	CStringFeatures<char>* subset_copy=(CStringFeatures<char>*)f->copy_subset(
			feature_copy_subset);

	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGVector<char> vec=subset_copy->get_feature_vector(i);

		//SG_SPRINT("%i: ", i);
		//for (index_t j=0; j<vec.vlen; ++j)
		//	SG_SPRINT("%c", vec.vector[j]);

		//SG_SPRINT("\n");

		subset_copy->free_feature_vector(vec, i);
	}

	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGVector<char> vec=subset_copy->get_feature_vector(i);

		for (index_t j=0; j<vec.vlen; ++j)
		{
			index_t offset_idx=i+(offset_copy+offset_subset);
			EXPECT_EQ(vec.vector[j], strings.strings[offset_idx].string[j]);
		}

		subset_copy->free_feature_vector(vec, i);
	}

	SG_UNREF(f);
	SG_UNREF(subset_copy);
}

TEST(StringFeaturesTest,clone)
{
	SGStringList<char> strings = generateRandomStringData();

	CStringFeatures<char>* f=new CStringFeatures<char>(strings, ALPHANUM);
	CStringFeatures<char>* f_clone = (CStringFeatures<char> *) f->clone();

	for (index_t i=0; i<f->get_num_vectors(); ++i)
	{
		index_t a_len;
		index_t b_len;
		bool a_free;
		bool b_free;

		char * a_vec = f->get_feature_vector(i, a_len, a_free);
		char * b_vec = f_clone->get_feature_vector(i, b_len, b_free);

		EXPECT_EQ(a_len, b_len);
		EXPECT_EQ(a_free, b_free);
		for (index_t j = 0; j < a_len; ++j)
			EXPECT_EQ(a_vec[j], b_vec[j]);
	}

	SG_UNREF(f);
	SG_UNREF(f_clone);
}
