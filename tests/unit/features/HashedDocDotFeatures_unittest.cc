/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Thoralf Klein, Sergey Lisitsyn,
 *          Bjoern Esser
 */

#include <gtest/gtest.h>
#include <shogun/features/hashed/HashedDocDotFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/NGramTokenizer.h>
#include <shogun/lib/Hash.h>
#include <shogun/mathematics/UniformIntDistribution.h>

#include <random>

using namespace shogun;

TEST(HashedDocDotFeaturesTest, computed_features_test)
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

	int32_t hash_bits = 5; //log2(32).

	auto tokenizer = std::make_shared<DelimiterTokenizer>();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	auto doc_collection = std::make_shared<StringFeatures<char>>(list, RAWBYTE);
	auto hddf = std::make_shared<HashedDocDotFeatures>(hash_bits, doc_collection,
			tokenizer, false);

	auto converter = std::make_shared<HashedDocConverter>(tokenizer, hash_bits, false);

	auto converted_docs =
	    converter->transform(doc_collection)->as<SparseFeatures<float64_t>>();

	for (index_t i=0; i<3; i++)
	{
		SGVector<float64_t> c_feat_2 = converted_docs->get_full_feature_vector(i);
		SGVector<float64_t> feat_2 = hddf->get_computed_dot_feature_vector(i);
		EXPECT_EQ(c_feat_2.size(), feat_2.size());

		for (index_t j=0; j<feat_2.size(); j++)
			EXPECT_EQ(c_feat_2[j], feat_2[j]);
	}




}

TEST(HashedDocDotFeaturesTest, dense_dot_test)
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

	int32_t dimension = 32;
	int32_t hash_bits = 5;

	auto tokenizer = std::make_shared<DelimiterTokenizer>();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	auto doc_collection = std::make_shared<StringFeatures<char>>(list, RAWBYTE);
	auto hddf = std::make_shared<HashedDocDotFeatures>(hash_bits, doc_collection,
			tokenizer, false);

	auto converter = std::make_shared<HashedDocConverter>(tokenizer, hash_bits, false);
	auto converted_docs =
	    converter->transform(doc_collection)->as<SparseFeatures<float64_t>>();

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	SGVector<float64_t> vec(dimension);
	for (index_t i=0; i<dimension; i++)
		vec[i] = uniform_int_dist(prng, {-dimension, dimension});

	for (index_t i=0; i<3; i++)
	{
		SGVector<float64_t> sv = converted_docs->get_full_feature_vector(i);
		float64_t converter_result = 0;
		for (index_t j=0; j<dimension; j++)
			converter_result += sv[j] * vec[j];

		float64_t features_result = hddf->dot(i, vec);
		EXPECT_EQ(features_result, converter_result);
	}




}

TEST(HashedDocDotFeaturesTest, quadratic_dense_dot)
{
	const char* doc_1 = "Shogun";
	const char* grams[] = {"Sho", "hog", "ogu", "gun"};
	uint32_t *hashes = SG_MALLOC(uint32_t, 4);

	const int32_t seed = 0xdeadbeaf;
	for (index_t i=0; i<4; i++)
		hashes[i] = Hash::MurmurHash3((uint8_t* ) &grams[i][0], 3, seed);


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

	SGVector<char> string_1(6);
	for (index_t i=0; i<6; i++)
		string_1[i] = doc_1[i];

	std::vector<SGVector<char>> list;
	list.push_back(string_1);

	auto tokenizer = std::make_shared<NGramTokenizer>(3);

	auto doc_collection = std::make_shared<StringFeatures<char>>(list, RAWBYTE);
	auto hddf = std::make_shared<HashedDocDotFeatures>(hash_bits, doc_collection,
			tokenizer, false, 3, 2);
	auto conv = std::make_shared<HashedDocConverter>(tokenizer, hash_bits, false, 3, 2);
	auto sf =
	    conv->transform(doc_collection)->as<SparseFeatures<float64_t>>();

	SGVector<float64_t> dense_vec(dimension);
	float64_t dot_product = 0;
	for (index_t i=0; i<dimension; i++)
	{
		dense_vec[i] = i;
		dot_product += i * vec[i];
	}

	float64_t hashed_dot_product = hddf->dot(0, dense_vec);
	EXPECT_EQ(hashed_dot_product, dot_product);
	float64_t sparse_dot_product = sf->dot(0, dense_vec);
	EXPECT_EQ(sparse_dot_product, dot_product);



	SG_FREE(hashes);

}

TEST(HashedDocDotFeaturesTest, quadratic_add_to_dense)
{
	const char* doc_1 = "Shogun";
	const char* grams[] = {"Sho", "hog", "ogu", "gun"};
	uint32_t *hashes = SG_MALLOC(uint32_t, 4);

	const int32_t seed = 0xdeadbeaf;
	for (index_t i=0; i<4; i++)
		hashes[i] = Hash::MurmurHash3((uint8_t* ) &grams[i][0], 3, seed);


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

	SGVector<char> string_1(6);
	for (index_t i=0; i<6; i++)
		string_1[i] = doc_1[i];

	std::vector<SGVector<char>> list;
	list.push_back(string_1);

	auto tokenizer = std::make_shared<NGramTokenizer>(3);
	auto doc_collection = std::make_shared<StringFeatures<char>>(list, RAWBYTE);
	auto hddf = std::make_shared<HashedDocDotFeatures>(hash_bits, doc_collection,
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


	SG_FREE(hashes);
}
