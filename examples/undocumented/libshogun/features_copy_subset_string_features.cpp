/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/features/Subset.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	index_t num_strings=10;
	index_t max_string_length=20;
	index_t min_string_length=max_string_length/2;

	SGStringList<char> strings(num_strings, max_string_length);

	SG_SPRINT("original string data:\n");
	for (index_t i=0; i<num_strings; ++i)
	{
		index_t len=CMath::random(min_string_length, max_string_length);
		SGString<char> current(len);

		SG_SPRINT("[%i]: \"", i);
		/* fill with random uppercase letters (ASCII) */
		for (index_t j=0; j<len; ++j)
		{
			current.string[j]=(char)CMath::random('A', 'Z');

			/* attach \0 to print letter */
			char* string=SG_MALLOC(char, 2);
			string[0]=current.string[j];
			string[1]='\0';
			SG_SPRINT("%s", string);
			SG_FREE(string);
		}
		SG_SPRINT("\"\n");

		strings.strings[i]=current;
	}

	/* create num_feautres 2-dimensional vectors */
	CStringFeatures<char>* f=new CStringFeatures<char>(strings, ALPHANUM);

	index_t offset_subset=1;
	SGVector<index_t> feature_subset(8);
	CMath::range_fill_vector(feature_subset.vector, feature_subset.vlen,
			offset_subset);
	CMath::display_vector(feature_subset.vector, feature_subset.vlen,
			"feature subset");

	f->add_subset(feature_subset);
	SG_SPRINT("feature vectors after setting subset on original data:\n");
	for (index_t i=0; i<f->get_num_vectors(); ++i)
	{
		SGVector<char> vec=f->get_feature_vector(i);

		SG_SPRINT("%i: ", i);
		for (index_t j=0; j<vec.vlen; ++j)
			SG_SPRINT("%c", vec.vector[j]);

		SG_SPRINT("\n");

		f->free_feature_vector(vec.vector, i, vec.do_free);
	}


	index_t offset_copy=2;
	SGVector<index_t> feature_copy_subset(4);
	CMath::range_fill_vector(feature_copy_subset.vector,
			feature_copy_subset.vlen, offset_copy);
	CMath::display_vector(feature_copy_subset.vector, feature_copy_subset.vlen,
			"indices that are to be copied");

	CStringFeatures<char>* subset_copy=(CStringFeatures<char>*)f->copy_subset(
			feature_copy_subset);

	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGVector<char> vec=subset_copy->get_feature_vector(i);

		SG_SPRINT("%i: ", i);
		for (index_t j=0; j<vec.vlen; ++j)
			SG_SPRINT("%c", vec.vector[j]);

		SG_SPRINT("\n");

		subset_copy->free_feature_vector(vec.vector, i, vec.do_free);
	}

	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGVector<char> vec=subset_copy->get_feature_vector(i);

		for (index_t j=0; j<vec.vlen; ++j)
		{
			index_t offset_idx=i+(offset_copy+offset_subset);
			ASSERT(vec.vector[j]==strings.strings[offset_idx].string[j]);
		}

		subset_copy->free_feature_vector(vec.vector, i, vec.do_free);
	}

	SG_UNREF(f);
	SG_UNREF(subset_copy);
	feature_copy_subset.destroy_vector();
	feature_subset.destroy_vector();

	exit_shogun();

	return 0;
}

