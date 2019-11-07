/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Sergey Lisitsyn, Bjoern Esser
 */

#include <shogun/converter/HashedDocConverter.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/Hash.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/hashed/HashedDocDotFeatures.h>
#include <shogun/mathematics/Math.h>

#include <utility>

using namespace shogun;

namespace shogun
{
HashedDocConverter::HashedDocConverter() : Converter()
{
	init(NULL, 16, false, 1, 0);
}

HashedDocConverter::HashedDocConverter(int32_t hash_bits, bool normalize,
	int32_t n_grams, int32_t skips) : Converter()
{
	init(NULL, hash_bits, normalize, n_grams, skips);
}

HashedDocConverter::HashedDocConverter(std::shared_ptr<Tokenizer> tzer,
	int32_t hash_bits, bool normalize, int32_t n_grams, int32_t skips) : Converter()
{
	init(std::move(tzer), hash_bits, normalize, n_grams, skips);
}

HashedDocConverter::~HashedDocConverter()
{

}

void HashedDocConverter::init(const std::shared_ptr<Tokenizer>& tzer, int32_t hash_bits, bool normalize,
	int32_t n_grams, int32_t skips)
{
	num_bits = hash_bits;
	should_normalize = normalize;
	ngrams = n_grams;
	tokens_to_skip = skips;

	if (tzer==NULL)
	{
		auto tk = std::make_shared<DelimiterTokenizer>();
		tk->delimiters[(uint8_t) ' '] = 1;
		tk->delimiters[(uint8_t) '\t'] = 1;
		tokenizer = tk;
	}
	else
		tokenizer = tzer;

	SG_ADD(&num_bits, "num_bits", "Number of bits of the hash");
	SG_ADD(&ngrams, "ngrams", "Number of consecutive tokens");
	SG_ADD(&tokens_to_skip, "tokens_to_skip", "Number of tokens to skip");
	SG_ADD(&should_normalize, "should_normalize", "Whether to normalize vectors or not");
	SG_ADD(&tokenizer, "tokenizer", "Tokenizer");
}

const char* HashedDocConverter::get_name() const
{
	return "HashedDocConverter";
}

std::shared_ptr<Features> HashedDocConverter::transform(std::shared_ptr<Features> features, bool inplace)
{
	ASSERT(features);
	if (strcmp(features->get_name(), "StringFeatures")!=0)
		error(
			"HashedConverter::transform() : Features object passed is "
			"not of type StringFeatures.");

	auto s_features = std::static_pointer_cast<StringFeatures<char>>(features);

	int32_t dim = Math::pow(2, num_bits);
	SGSparseMatrix<float64_t> matrix(dim,features->get_num_vectors());
	for (index_t vec_idx=0; vec_idx<s_features->get_num_vectors(); vec_idx++)
	{
		SGVector<char> doc = s_features->get_feature_vector(vec_idx);
		matrix[vec_idx] = apply(doc);
		s_features->free_feature_vector(doc, vec_idx);
	}

	return std::make_shared<SparseFeatures<float64_t>>(matrix);
}

SGSparseVector<float64_t> HashedDocConverter::apply(SGVector<char> document)
{
	ASSERT(document.size()>0)
	const int32_t array_size = 1024*1024;
	/** the array will contain all the hashes generated from the tokens */
	std::vector<uint32_t> hashed_indices;
	hashed_indices.reserve(array_size);

	/** this vector will maintain the current n+k active tokens
	 * in a circular manner */
	SGVector<uint32_t> cached_hashes(ngrams+tokens_to_skip);
	index_t hashes_start = 0;
	index_t hashes_end = 0;
	int32_t len = cached_hashes.vlen - 1;

	/** the combinations generated from the current active tokens will be
	 * stored here to avoid creating new objects */
	SGVector<index_t> ngram_indices((ngrams-1)*(tokens_to_skip+1) + 1);

	/** Reading n+s-1 tokens */
	const int32_t seed = 0xdeadbeaf;
	tokenizer->set_text(document);
	index_t token_start = 0;
	while (hashes_end<ngrams-1+tokens_to_skip && tokenizer->has_next())
	{
		index_t end = tokenizer->next_token_idx(token_start);
		uint32_t token_hash = Hash::MurmurHash3((uint8_t* ) &document.vector[token_start],
				end-token_start, seed);
		cached_hashes[hashes_end++] = token_hash;
	}

	/** Reading token and storing index to hashed_indices */
	while (tokenizer->has_next())
	{
		index_t end = tokenizer->next_token_idx(token_start);
		uint32_t token_hash = Hash::MurmurHash3((uint8_t* ) &document.vector[token_start],
				end-token_start, seed);
		cached_hashes[hashes_end] = token_hash;

		HashedDocConverter::generate_ngram_hashes(cached_hashes, hashes_start, len,
				ngram_indices, num_bits, ngrams, tokens_to_skip);

		for (index_t i=0; i<ngram_indices.vlen; i++)
			hashed_indices.push_back(ngram_indices[i]);

		hashes_start++;
		hashes_end++;
		if (hashes_end==cached_hashes.vlen)
			hashes_end = 0;
		if (hashes_start==cached_hashes.vlen)
			hashes_start = 0;
	}

	/** For remaining combinations */
	if (ngrams>1)
	{
		while (hashes_start!=hashes_end)
		{
			len--;
			index_t max_idx = HashedDocConverter::generate_ngram_hashes(cached_hashes, hashes_start,
					len, ngram_indices, num_bits, ngrams, tokens_to_skip);

			for (index_t i=0; i<max_idx; i++)
				hashed_indices.push_back(ngram_indices[i]);

			hashes_start++;
			if (hashes_start==cached_hashes.vlen)
				hashes_start = 0;
		}
	}

	SGSparseVector<float64_t> sparse_doc_rep = create_hashed_representation(hashed_indices);

	/** Normalizing vector */
	if (should_normalize)
	{
		float64_t norm_const = std::sqrt((float64_t)document.size());
		for (index_t i=0; i<sparse_doc_rep.num_feat_entries; i++)
			sparse_doc_rep.features[i].entry /= norm_const;
	}

	return sparse_doc_rep;
}

SGSparseVector<float64_t> HashedDocConverter::create_hashed_representation(std::vector<uint32_t>& hashed_indices)
{
	int32_t num_nnz_features = count_distinct_indices(hashed_indices);

	SGSparseVector<float64_t> sparse_doc_rep(num_nnz_features);
	index_t sparse_idx = 0;
	for (size_t i=0; i<hashed_indices.size(); i++)
	{
		sparse_doc_rep.features[sparse_idx].feat_index = hashed_indices[i];
		sparse_doc_rep.features[sparse_idx].entry = 1;
		while ( (i+1<hashed_indices.size()) &&
				(hashed_indices[i+1]==hashed_indices[i]) )
		{
			sparse_doc_rep.features[sparse_idx].entry++;
			i++;
		}
		sparse_idx++;
	}
	return sparse_doc_rep;
}

index_t HashedDocConverter::generate_ngram_hashes(SGVector<uint32_t>& hashes, index_t hashes_start,
	index_t len, SGVector<index_t>& ngram_hashes, int32_t num_bits, int32_t ngrams, int32_t tokens_to_skip)
{
	index_t h_idx = 0;
	ngram_hashes[h_idx++] = hashes[hashes_start] & ((1 << num_bits) -1);

	for (index_t n=1; n<ngrams; n++)
	{
		for (index_t s=0; s<=tokens_to_skip; s++)
		{
			if ( n+s > len)
				break;

			uint32_t ngram_hash = hashes[hashes_start];
			for (index_t i=hashes_start+1+s; i<=hashes_start+n+s; i++)
				ngram_hash = ngram_hash ^ hashes[i % hashes.vlen];
			ngram_hash = ngram_hash & ((1 << num_bits) - 1);
			ngram_hashes[h_idx++] = ngram_hash;
		}
	}
	return h_idx;
}

int32_t HashedDocConverter::count_distinct_indices(std::vector<uint32_t>& hashed_indices)
{
	std::sort(hashed_indices.begin(), hashed_indices.end());

	/** Counting nnz features */
	int32_t num_nnz_features = 0;
	for (size_t i=0; i<hashed_indices.size(); i++)
	{
		num_nnz_features++;
		while ( (i+1<hashed_indices.size()) &&
				(hashed_indices[i+1]==hashed_indices[i]) )
		{
			i++;
		}
	}
	return num_nnz_features;
}

void HashedDocConverter::set_normalization(bool normalize)
{
	should_normalize = normalize;
}

void HashedDocConverter::set_k_skip_n_grams(int32_t k, int32_t n)
{
	tokens_to_skip = k;
	ngrams = n;
}
}
