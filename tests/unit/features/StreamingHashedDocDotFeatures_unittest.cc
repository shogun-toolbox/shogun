/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Soeren Sonnenburg, Bjoern Esser
 */

#include <gtest/gtest.h>
#include <shogun/features/streaming/StreamingHashedDocDotFeatures.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/converter/HashedDocConverter.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <random>

using namespace shogun;

TEST(StreamingHashedDocFeaturesTest, example_reading)
{
	const char* doc_1 = "You're never too old to rock and roll, if you're too young to die";
	const char* doc_2 = "Give me some rope, tie me to dream, give me the hope to run out of steam";
	const char* doc_3 = "Thank you Jack Daniels, Old Number Seven, Tennessee Whiskey got me drinking in heaven";

	SGVector<char> string_1(65);
	for (index_t i=0; i<65; i++)
		string_1[i] = doc_1[i];

	SGVector<char> string_2(72);
	for (index_t i=0; i<72; i++)
		string_2[i] = doc_2[i];

	SGVector<char> string_3(85);
	for (index_t i=0; i<85; i++)
		string_3[i] = doc_3[i];

	std::vector<SGVector<char>> list;
	list.reserve(3);
	list.push_back(string_1);
	list.push_back(string_2);
	list.push_back(string_3);

	auto tokenizer = std::make_shared<DelimiterTokenizer>();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	auto converter = std::make_shared<HashedDocConverter>(tokenizer, 5, true);
	auto doc_collection = std::make_shared<StringFeatures<char>>(list, RAWBYTE);
	auto feats = std::make_shared<StreamingHashedDocDotFeatures>(doc_collection,
			tokenizer, 5);

	index_t i = 0;
	feats->start_parser();
	while (feats->get_next_example())
	{
		SGSparseVector<float64_t> example = feats->get_vector();

		SGVector<char> tmp = list[i];
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



}

TEST(StreamingHashedDocFeaturesTest, dot_tests)
{
	int32_t seed = 12;
	const char* doc_1 = "You're never too old to rock and roll, if you're too young to die";
	const char* doc_2 = "Give me some rope, tie me to dream, give me the hope to run out of steam";
	const char* doc_3 = "Thank you Jack Daniels, Old Number Seven, Tennessee Whiskey got me drinking in heaven";

	SGVector<char> string_1(65);
	for (index_t i=0; i<65; i++)
		string_1[i] = doc_1[i];

	SGVector<char> string_2(72);
	for (index_t i=0; i<72; i++)
		string_2[i] = doc_2[i];

	SGVector<char> string_3(85);
	for (index_t i=0; i<85; i++)
		string_3[i] = doc_3[i];

	std::vector<SGVector<char>> list;
	list.reserve(3);
	list.push_back(string_1);
	list.push_back(string_2);
	list.push_back(string_3);

	auto tokenizer = std::make_shared<DelimiterTokenizer>();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	auto converter = std::make_shared<HashedDocConverter>(tokenizer, 5, true);
	auto doc_collection = std::make_shared<StringFeatures<char>>(list, RAWBYTE);
	auto feats = std::make_shared<StreamingHashedDocDotFeatures>(doc_collection,
			tokenizer, 5);
	feats->start_parser();

	std::mt19937_64 prng(seed);
	UniformRealDistribution<float64_t> uniform_real_dist;
	SGVector<float32_t> dense_vec(32);
	for (index_t j=0; j<32; j++)
		dense_vec[j] = uniform_real_dist(prng, {0.0, 1.0});

	index_t i = 0;
	while (feats->get_next_example())
	{
		/** Dense dot test */
		SGVector<char> tmp = list[i];
		SGSparseVector<float64_t> converted_doc = converter->apply(tmp);

		float32_t tmp_res = 0;
		for (index_t j=0; j<converted_doc.num_feat_entries; j++)
			tmp_res += dense_vec[converted_doc.features[j].feat_index] * converted_doc.features[j].entry;

		EXPECT_NEAR(tmp_res, feats->dense_dot(dense_vec.vector, dense_vec.vlen), 1e-7);

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
				EXPECT_NEAR(dense_vec2[j], dense_vec[j] + example.features[sparse_idx].entry, 1e-7);
				sparse_idx++;
			}
			else
				EXPECT_NEAR(dense_vec2[j], dense_vec[j], 1e-7);
		}

		feats->release_example();
		i++;
	}

	feats->end_parser();



}
