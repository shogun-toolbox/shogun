/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/streaming/StreamingHashedDocDotFeatures.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/converter/HashedDocConverter.h>

#include <gtest/gtest.h>

using namespace shogun;

TEST(StreamingHashedDocFeaturesTest, example_reading)
{
	const char* doc_1 = "You're never too old to rock and roll, if you're too young to die";
	const char* doc_2 = "Give me some rope, tie me to dream, give me the hope to run out of steam";
	const char* doc_3 = "Thank you Jack Daniels, Old Number Seven, Tennessee Whiskey got me drinking in heaven";
	
	SGString<char> string_1(65);
	for (index_t i=0; i<65; i++)
		string_1.string[i] = doc_1[i];

	SGString<char> string_2(72);
	for (index_t i=0; i<72; i++)
		string_2.string[i] = doc_2[i];

	SGString<char> string_3(85);
	for (index_t i=0; i<85; i++)
		string_3.string[i] = doc_3[i];

	SGStringList<char> list(3,85);
	list.strings[0] = string_1;
	list.strings[1] = string_2;
	list.strings[2] = string_3;

	CDelimiterTokenizer* tokenizer = new CDelimiterTokenizer();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	CHashedDocConverter* converter = new CHashedDocConverter(tokenizer, 5, false);
	CStringFeatures<char>* doc_collection = new CStringFeatures<char>(list, RAWBYTE);
	CStreamingHashedDocDotFeatures* feats = new CStreamingHashedDocDotFeatures(doc_collection,
			tokenizer, 5);

	index_t i = 0;
	feats->start_parser();
	while (feats->get_next_example())
	{
		SGSparseVector<float64_t> example = feats->get_vector();

		SGVector<char> tmp(list.strings[i].string, list.strings[i].slen, false);
		tmp.vector = list.strings[i].string;
		SGSparseVector<float64_t> converted_doc = converter->apply(tmp);
		
		EXPECT_EQ(example.num_feat_entries, converted_doc.num_feat_entries);

		for (index_t j=0; j<example.num_feat_entries; j++)
		{
			EXPECT_EQ(example.features[j].feat_index, converted_doc.features[j].feat_index);
			EXPECT_EQ(example.features[j].entry, converted_doc.features[j].entry);
		}
		feats->release_example();
		i++;
	}
	feats->end_parser();
	SG_UNREF(feats);
	SG_UNREF(doc_collection);
	SG_UNREF(converter);
}

TEST(StreamingHashedDocFeaturesTest, dot_tests)
{
	const char* doc_1 = "You're never too old to rock and roll, if you're too young to die";
	const char* doc_2 = "Give me some rope, tie me to dream, give me the hope to run out of steam";
	const char* doc_3 = "Thank you Jack Daniels, Old Number Seven, Tennessee Whiskey got me drinking in heaven";
	
	SGString<char> string_1(65);
	for (index_t i=0; i<65; i++)
		string_1.string[i] = doc_1[i];

	SGString<char> string_2(72);
	for (index_t i=0; i<72; i++)
		string_2.string[i] = doc_2[i];

	SGString<char> string_3(85);
	for (index_t i=0; i<85; i++)
		string_3.string[i] = doc_3[i];

	SGStringList<char> list(3,85);
	list.strings[0] = string_1;
	list.strings[1] = string_2;
	list.strings[2] = string_3;

	CDelimiterTokenizer* tokenizer = new CDelimiterTokenizer();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	CHashedDocConverter* converter = new CHashedDocConverter(tokenizer, 5, false);
	CStringFeatures<char>* doc_collection = new CStringFeatures<char>(list, RAWBYTE);
	CStreamingHashedDocDotFeatures* feats = new CStreamingHashedDocDotFeatures(doc_collection,
			tokenizer, 5);
	feats->start_parser();

	SGVector<float32_t> dense_vec(32);
	for (index_t j=0; j<32; j++)
		dense_vec[j] = CMath::random();

	index_t i = 0;
	while (feats->get_next_example())
	{
		/** Dense dot test */
		SGVector<char> tmp(list.strings[i].string, list.strings[i].slen, false);
		tmp.vector = list.strings[i].string;
		SGSparseVector<float64_t> converted_doc = converter->apply(tmp);
		
		float32_t tmp_res = 0;
		for (index_t j=0; j<converted_doc.num_feat_entries; j++)
			tmp_res += dense_vec[converted_doc.features[j].feat_index] * converted_doc.features[j].entry;

		EXPECT_EQ(tmp_res, feats->dense_dot(dense_vec.vector, dense_vec.vlen));

		/** Add to dense test */
		SGSparseVector<float64_t> example = feats->get_vector();
		SGVector<float32_t> dense_vec2(32);
		for (index_t j=0; j<32; j++)
			dense_vec2[j] = dense_vec[j];
		
		feats->add_to_dense_vec(1, dense_vec2.vector, dense_vec2.vlen);
		index_t sparse_idx = 0;
		for (index_t j=0; j<32; j++)
		{
			if ( (sparse_idx < example.num_feat_entries) && 
					(example.features[sparse_idx].feat_index == j) )
			{
				EXPECT_EQ(dense_vec2[j], dense_vec[j] + example.features[sparse_idx].entry);
				sparse_idx++;
			}
			else
				EXPECT_EQ(dense_vec2[j], dense_vec[j]);
		}
		
		feats->release_example();
		i++;
	}
	
	feats->end_parser();
	SG_UNREF(feats);
	SG_UNREF(doc_collection);
	SG_UNREF(converter);
}
