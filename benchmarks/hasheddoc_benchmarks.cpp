/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/base/init.h>
#include <shogun/classifier/svm/SVMOcas.h>
#include <shogun/features/HashedDocDotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/NGramTokenizer.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

int main(int argv, char** argc)
{
	init_shogun_with_defaults();

	int32_t bits[] = {8, 10, 12, 16, 20};
	int32_t bits_length = 5;

	int32_t num_strings = 5000;
	int32_t max_str_length = 10000;
	SGStringList<char> string_list(num_strings, max_str_length);

	SG_SPRINT("Creating features...\n");
	for (index_t i=0; i<num_strings; i++)
	{
		string_list.strings[i] = SGString<char>(max_str_length);
		for (index_t j=0; j<max_str_length; j++)
			string_list.strings[i].string[j] = (char) CMath::random('A', 'Z');
	}
	SG_SPRINT("Features were created.\n");

	CStringFeatures<char>* string_feats = new CStringFeatures<char>(string_list, RAWBYTE);
	CNGramTokenizer* tzer = new CNGramTokenizer(3);

	for (index_t i=0; i<bits_length; i++)
	{
		int32_t b = bits[i];
		SG_SPRINT("Starting training for num_bits = %d\n", b);

		SG_REF(string_feats);
		SG_REF(tzer);
		CHashedDocDotFeatures* feats = new CHashedDocDotFeatures(b, string_feats, tzer);
		feats->benchmark_dense_dot_range();
		feats->benchmark_add_to_dense_vector();
	}
	exit_shogun();
}
