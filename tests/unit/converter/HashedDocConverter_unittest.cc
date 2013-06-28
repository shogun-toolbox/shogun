#include <shogun/converter/HashedDocConverter.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/Hash.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(HashedDocConverterTest, apply_single_vector)
{
	const char* text = "When life gives you lemonade make lemons";
	const char* words[] = {"When", "life", "gives", "you", "lemonade", "make", "lemons"};
	int32_t sizes[] = {4, 4, 5, 3, 8, 4, 6};

	int32_t num_tokens = 7;
	int32_t new_dim_size = 16;

	SGVector<uint32_t> hashed_rep(new_dim_size);

	SGVector<uint32_t>::fill_vector(hashed_rep, new_dim_size, 0);
	uint32_t hash = 0;
	for (index_t i=0; i<num_tokens; i++)
	{
		hash = CHash::MurmurHash3((uint8_t*) words[i], sizes[i], hash);
		hash = hash & ((1 << 4) - 1);
		hashed_rep[hash]++;
	}

	CHashedDocConverter* converter = new CHashedDocConverter(new_dim_size);

	SGVector<char> doc(const_cast<char* >(text), 40, false);

	SGSparseVector<uint32_t> c_doc = converter->apply(doc);

	for (index_t i=0; i<c_doc.num_feat_entries; i++)
	{
		ASSERT_EQ(c_doc.features[i].entry,
				hashed_rep[c_doc.features[i].feat_index]);
	}

	SG_UNREF(converter);
}
