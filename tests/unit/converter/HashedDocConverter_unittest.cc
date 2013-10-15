#include <shogun/converter/HashedDocConverter.h>
#include <shogun/features/HashedDocDotFeatures.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/NGramTokenizer.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(HashedDocConverterTest, apply_single_vector)
{
	const char* text = "When life gives you lemonade make lemons";
	const char* words[] = {"When", "life", "gives", "you", "lemonade", "make", "lemons"};
	int32_t sizes[] = {4, 4, 5, 3, 8, 4, 6};

	int32_t num_tokens = 7;
	int32_t new_dim_size = 16;
	int32_t hash_bits = 4;

	SGVector<float64_t> hashed_rep(new_dim_size);

	SGVector<float64_t>::fill_vector(hashed_rep, new_dim_size, 0);
	const int32_t seed = 0xdeadbeaf;
	uint32_t hash = 0;
	for (index_t i=0; i<num_tokens; i++)
	{
		hash = CHash::MurmurHash3((uint8_t*) words[i], sizes[i], seed);
		hash = hash & ((1 << hash_bits) - 1);
		hashed_rep[hash]++;
	}

	CHashedDocConverter* converter = new CHashedDocConverter(hash_bits, false);

	SGVector<char> doc(const_cast<char* >(text), 40, false);

	SGSparseVector<float64_t> c_doc = converter->apply(doc);

	for (index_t i=0; i<c_doc.num_feat_entries; i++)
	{
		ASSERT_EQ(c_doc.features[i].entry,
				hashed_rep[c_doc.features[i].feat_index]);
	}

	SG_UNREF(converter);
}

TEST(HashedDocConverterTest, compare_dot_features)
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

	int32_t hash_bits = 3;

	CDelimiterTokenizer* tokenizer = new CDelimiterTokenizer();
	tokenizer->delimiters[' '] = 1;
	tokenizer->delimiters['\''] = 1;
	tokenizer->delimiters[','] = 1;

	CStringFeatures<char>* s_feats = new CStringFeatures<char>(list, RAWBYTE);
	CHashedDocDotFeatures* d_feats = new CHashedDocDotFeatures(hash_bits, s_feats, tokenizer);
	CHashedDocConverter* converter = new CHashedDocConverter(tokenizer, hash_bits, true);

	float64_t e = 0.1e-14;
	for (index_t i=0; i<3; i++)
	{
		SGSparseVector<float64_t> c_vec = converter->apply(SGVector<char>(list.strings[i].string, list.strings[i].slen, false));
		SGVector<float64_t> d_vec = d_feats->get_computed_dot_feature_vector(i);
		for (index_t j=0; j<c_vec.num_feat_entries; j++)
			EXPECT_TRUE(e > c_vec.features[j].entry - d_vec[c_vec.features[j].feat_index]);
	}

	SG_UNREF(converter);
	SG_UNREF(d_feats);
}

TEST(HashedDocConverterTest, apply_quadratic_test)
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

	CNGramTokenizer* tzer = new CNGramTokenizer(3);
	CHashedDocConverter* converter = new CHashedDocConverter(tzer, hash_bits, false, 3, 2);

	SGVector<char> doc(const_cast<char* >(doc_1), 6, false);

	SGSparseVector<float64_t> c_doc = converter->apply(doc);

	for (index_t i=0; i<c_doc.num_feat_entries; i++)
		EXPECT_EQ(c_doc.features[i].entry,
					vec[c_doc.features[i].feat_index]);

	SG_FREE(hashes);
	SG_UNREF(converter);
}
