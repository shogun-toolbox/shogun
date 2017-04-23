/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2013 Heiko Strathmann
 */

#include <shogun/lib/memory.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/SGStringList.h>
#include <gtest/gtest.h>

using namespace shogun;

SGStringList<char> generateRandomData(index_t num_strings=10, index_t max_string_length=20, index_t min_string_length=10)
{
	SGStringList<char> strings(num_strings, max_string_length);

	//SG_SPRINT("original string data:\n");
	for (index_t i=0; i<num_strings; ++i)
	{
		index_t len=CMath::random(min_string_length, max_string_length);
		SGString<char> current(len);

		//SG_SPRINT("[%i]: \"", i);
		/* fill with random uppercase letters (ASCII) */
		for (index_t j=0; j<len; ++j)
		{
			current.string[j]=(char)CMath::random('A', 'Z');

			/* attach \0 to print letter */
			char* string=SG_MALLOC(char, 2);
			string[0]=current.string[j];
			string[1]='\0';
			//SG_SPRINT("%s", string);
			SG_FREE(string);
		}
		//SG_SPRINT("\"\n");

		strings.strings[i]=current;
	}
	return strings;
}

TEST(StringFeaturesTest,copy_subset)
{
	SGStringList<char> strings = generateRandomData();

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
	SGStringList<char> strings = generateRandomData();

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
