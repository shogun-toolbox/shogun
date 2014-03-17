/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/HashedDocDotFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/NGramTokenizer.h>
#include <shogun/lib/Hash.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(HashedDocDotFeaturesTest, computed_features_test)
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

	int32_t hash_bits = 5; //log2(32).

	CDelimiterTokenizer* tokenizer = new CDelimiterTokenizer();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	CStringFeatures<char>* doc_collection = new CStringFeatures<char>(list, RAWBYTE);
	CHashedDocDotFeatures* hddf = new CHashedDocDotFeatures(hash_bits, doc_collection,
			tokenizer, false);

	CHashedDocConverter* converter = new CHashedDocConverter(tokenizer, hash_bits, false);

	CSparseFeatures<float64_t>* converted_docs = (CSparseFeatures<float64_t>* ) converter->apply(doc_collection);

	for (index_t i=0; i<3; i++)
	{
		SGVector<float64_t> c_feat_2 = converted_docs->get_full_feature_vector(i);
		SGVector<float64_t> feat_2 = hddf->get_computed_dot_feature_vector(i);
		EXPECT_EQ(c_feat_2.size(), feat_2.size());

		for (index_t j=0; j<feat_2.size(); j++)
			EXPECT_EQ(c_feat_2[j], feat_2[j]);
	}

	SG_UNREF(converter);
	SG_UNREF(converted_docs);
	SG_UNREF(hddf);
}

TEST(HashedDocDotFeaturesTest, dense_dot_test)
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

	int32_t dimension = 32;
	int32_t hash_bits = 5;

	CDelimiterTokenizer* tokenizer = new CDelimiterTokenizer();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	CStringFeatures<char>* doc_collection = new CStringFeatures<char>(list, RAWBYTE);
	CHashedDocDotFeatures* hddf = new CHashedDocDotFeatures(hash_bits, doc_collection,
			tokenizer, false);

	CHashedDocConverter* converter = new CHashedDocConverter(tokenizer, hash_bits, false);
	CSparseFeatures<float64_t>* converted_docs = (CSparseFeatures<float64_t>* ) converter->apply(doc_collection);

	SGVector<float64_t> vec(dimension);
	for (index_t i=0; i<dimension; i++)
		vec[i] = CMath::random(-dimension, dimension);

	for (index_t i=0; i<3; i++)
	{
		SGVector<float64_t> sv = converted_docs->get_full_feature_vector(i);
		float64_t converter_result = 0;
		for (index_t j=0; j<dimension; j++)
			converter_result += sv[j] * vec[j];

		float64_t features_result = hddf->dense_dot(i, vec.vector, dimension);
		EXPECT_EQ(features_result, converter_result);
	}

	SG_UNREF(converter);
	SG_UNREF(converted_docs);
	SG_UNREF(hddf);
}

TEST(HashedDocDotFeaturesTest, quadratic_dense_dot)
{
	const char* doc_1 = "Shogun";
	const char* grams[] = {"Sho", "hog", "ogu", "gun"};
	uint32_t *hashes = SG_MALLOC(uint32_t, 4);

	const int32_t seed = 0xdeadbeaf;
	for (index_t i=0; i<4; i++)
		hashes[i] = CHash::MurmurHash3((uint8_t* ) &grams[i][0], 3, seed);


	int32_t dimension = 32;
	int32_t hash_bits = 5;

	SGVector<float64_t> vec(dimension);
	SGVector<float64_t>::fill_vector(vec.vector, vec.vlen, 0);

	uint32_t val = hashes[0];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[1];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[2];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[1] ^ hashes[2];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[2] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[1];
	vec[val % dimension]++;

	val = hashes[1] ^ hashes[2];
	vec[val % dimension]++;

	val = hashes[1] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[1] ^ hashes[2] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[2];
	vec[val % dimension]++;

	val = hashes[2] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[3];
	vec[val % dimension]++;

	SGString<char> string_1(6);
	for (index_t i=0; i<6; i++)
		string_1.string[i] = doc_1[i];

	SGStringList<char> list(1,6);
	list.strings[0] = string_1;

	CNGramTokenizer* tokenizer = new CNGramTokenizer(3);
	SG_REF(tokenizer);
	CStringFeatures<char>* doc_collection = new CStringFeatures<char>(list, RAWBYTE);
	CHashedDocDotFeatures* hddf = new CHashedDocDotFeatures(hash_bits, doc_collection,
			tokenizer, false, 3, 2);
	CHashedDocConverter* conv = new CHashedDocConverter(tokenizer, hash_bits, false, 3, 2);
	CSparseFeatures<float64_t>* sf = (CSparseFeatures<float64_t>* ) conv->apply(doc_collection);
	SG_UNREF(conv);
	SGVector<float64_t> dense_vec(dimension);
	float64_t dot_product = 0;
	for (index_t i=0; i<dimension; i++)
	{
		dense_vec[i] = i;
		dot_product += i * vec[i];
	}

	float64_t hashed_dot_product = hddf->dense_dot(0, dense_vec.vector, dense_vec.vlen);
	EXPECT_EQ(hashed_dot_product, dot_product);
	float64_t sparse_dot_product = sf->dense_dot(0, dense_vec.vector, dense_vec.vlen);
	EXPECT_EQ(sparse_dot_product, dot_product);

	SG_UNREF(sf);
	SG_UNREF(hddf);
	SG_FREE(hashes);
	SG_UNREF(tokenizer);
}

TEST(HashedDocDotFeaturesTest, quadratic_add_to_dense)
{
	const char* doc_1 = "Shogun";
	const char* grams[] = {"Sho", "hog", "ogu", "gun"};
	uint32_t *hashes = SG_MALLOC(uint32_t, 4);

	const int32_t seed = 0xdeadbeaf;
	for (index_t i=0; i<4; i++)
		hashes[i] = CHash::MurmurHash3((uint8_t* ) &grams[i][0], 3, seed);


	int32_t dimension = 32;
	int32_t hash_bits = 5;

	SGVector<float64_t> vec(dimension);
	SGVector<float64_t>::fill_vector(vec.vector, vec.vlen, 0);

	uint32_t val = hashes[0];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[1];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[2];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[1] ^ hashes[2];
	vec[val % dimension]++;

	val = hashes[0] ^ hashes[2] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[1];
	vec[val % dimension]++;

	val = hashes[1] ^ hashes[2];
	vec[val % dimension]++;

	val = hashes[1] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[1] ^ hashes[2] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[2];
	vec[val % dimension]++;

	val = hashes[2] ^ hashes[3];
	vec[val % dimension]++;

	val = hashes[3];
	vec[val % dimension]++;

	SGString<char> string_1(6);
	for (index_t i=0; i<6; i++)
		string_1.string[i] = doc_1[i];

	SGStringList<char> list(1,6);
	list.strings[0] = string_1;

	CNGramTokenizer* tokenizer = new CNGramTokenizer(3);
	CStringFeatures<char>* doc_collection = new CStringFeatures<char>(list, RAWBYTE);
	CHashedDocDotFeatures* hddf = new CHashedDocDotFeatures(hash_bits, doc_collection,
			tokenizer, false, 3, 2);

	SGVector<float64_t> dense_vec(dimension);
	SGVector<float64_t> dense_vec2(dimension);
	for (index_t i=0; i<dimension; i++)
	{
		dense_vec[i] = i + vec[i];
		dense_vec2[i] = i;
	}

	hddf->add_to_dense_vec(1, 0, dense_vec2.vector, dense_vec2.vlen, 0);

	for (index_t i=0; i<dimension; i++)
		EXPECT_EQ(dense_vec[i], dense_vec2[i]);

	SG_UNREF(hddf);
	SG_FREE(hashes);
}
