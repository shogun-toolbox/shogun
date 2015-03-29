/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/hashed/HashedDocDotFeatures.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/Hash.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
CHashedDocDotFeatures::CHashedDocDotFeatures(int32_t hash_bits, CStringFeatures<char>* docs,
	CTokenizer* tzer, bool normalize, int32_t n_grams, int32_t skips, int32_t size) : CDotFeatures(size)
{
	if (n_grams < 1)
		n_grams = 1;

	if ( (n_grams==1 && skips!=0) || (skips<0))
		skips = 0;

	init(hash_bits, docs, tzer, normalize, n_grams, skips);
}

CHashedDocDotFeatures::CHashedDocDotFeatures(const CHashedDocDotFeatures& orig)
: CDotFeatures(orig)
{
	init(orig.num_bits, orig.doc_collection, orig.tokenizer, orig.should_normalize,
			orig.ngrams, orig.tokens_to_skip);
}

CHashedDocDotFeatures::CHashedDocDotFeatures(CFile* loader)
{
	SG_NOTIMPLEMENTED;
}

void CHashedDocDotFeatures::init(int32_t hash_bits, CStringFeatures<char>* docs,
	CTokenizer* tzer, bool normalize, int32_t n_grams, int32_t skips)
{
	num_bits = hash_bits;
	ngrams = n_grams;
	tokens_to_skip = skips;
	doc_collection = docs;
	tokenizer = tzer;
	should_normalize = normalize;

	if (!tokenizer)
	{
		tokenizer = new CDelimiterTokenizer();
		((CDelimiterTokenizer* )tokenizer)->init_for_whitespace();
	}

	SG_ADD(&num_bits, "num_bits", "Number of bits of hash", MS_NOT_AVAILABLE);
	SG_ADD(&ngrams, "ngrams", "Number of tokens to combine for quadratic feature support",
			MS_NOT_AVAILABLE);
	SG_ADD(&tokens_to_skip, "tokens_to_skip", "Number of tokens to skip when combining features",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &doc_collection, "doc_collection", "Document collection",
			MS_NOT_AVAILABLE);
	SG_ADD((CSGObject**) &tokenizer, "tokenizer", "Document tokenizer",
			MS_NOT_AVAILABLE);
	SG_ADD(&should_normalize, "should_normalize", "Normalize or not the dot products",
			MS_NOT_AVAILABLE);

	SG_REF(doc_collection);
	SG_REF(tokenizer);
}

CHashedDocDotFeatures::~CHashedDocDotFeatures()
{
	SG_UNREF(doc_collection);
	SG_UNREF(tokenizer);
}

int32_t CHashedDocDotFeatures::get_dim_feature_space() const
{
	return CMath::pow(2, num_bits);
}

float64_t CHashedDocDotFeatures::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_name() == get_name())

	CHashedDocDotFeatures* hddf = (CHashedDocDotFeatures*) df;

	SGVector<char> sv1 = doc_collection->get_feature_vector(vec_idx1);
	SGVector<char> sv2 = hddf->doc_collection->get_feature_vector(vec_idx2);

	CHashedDocConverter* converter = new CHashedDocConverter(tokenizer, num_bits,
			should_normalize, ngrams, tokens_to_skip);
	SGSparseVector<float64_t> cv1 = converter->apply(sv1);
	SGSparseVector<float64_t> cv2 = converter->apply(sv2);
	float64_t result = SGSparseVector<float64_t>::sparse_dot(cv1,cv2);

	doc_collection->free_feature_vector(sv1, vec_idx1);
	hddf->doc_collection->free_feature_vector(sv2, vec_idx2);
	SG_UNREF(converter);

	return result;
}

float64_t CHashedDocDotFeatures::dense_dot_sgvec(int32_t vec_idx1, const SGVector<float64_t> vec2)
{
	return dense_dot(vec_idx1, vec2.vector, vec2.vlen);
}

float64_t CHashedDocDotFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == CMath::pow(2,num_bits))

	SGVector<char> sv = doc_collection->get_feature_vector(vec_idx1);

	/** this vector will maintain the current n+k active tokens
	 * in a circular manner */
	SGVector<uint32_t> hashes(ngrams+tokens_to_skip);
	index_t hashes_start = 0;
	index_t hashes_end = 0;
	int32_t len = hashes.vlen - 1;

	/** the combinations generated from the current active tokens will be
	 * stored here to avoid creating new objects */
	SGVector<index_t> hashed_indices((ngrams-1)*(tokens_to_skip+1) + 1);

	float64_t result = 0;
	CTokenizer* local_tzer = tokenizer->get_copy();

	/** Reading n+k-1 tokens */
	const int32_t seed = 0xdeadbeaf;
	local_tzer->set_text(sv);
	index_t start = 0;
	while (hashes_end<ngrams-1+tokens_to_skip && local_tzer->has_next())
	{
		index_t end = local_tzer->next_token_idx(start);
		uint32_t token_hash = CHash::MurmurHash3((uint8_t* ) &sv.vector[start], end-start, seed);
		hashes[hashes_end++] = token_hash;
	}

	/** Reading token and storing indices to hashed_indices */
	while (local_tzer->has_next())
	{
		index_t end = local_tzer->next_token_idx(start);
		uint32_t token_hash = CHash::MurmurHash3((uint8_t* ) &sv.vector[start], end-start, seed);
		hashes[hashes_end] = token_hash;

		CHashedDocConverter::generate_ngram_hashes(hashes, hashes_start, len, hashed_indices,
				num_bits, ngrams, tokens_to_skip);

		for (index_t i=0; i<hashed_indices.vlen; i++)
			result += vec2[hashed_indices[i]];

		hashes_start++;
		hashes_end++;
		if (hashes_end==hashes.vlen)
			hashes_end = 0;
		if (hashes_start==hashes.vlen)
			hashes_start = 0;
	}

	if (ngrams>1)
	{
		while (hashes_start!=hashes_end)
		{
			len--;
			index_t max_idx = CHashedDocConverter::generate_ngram_hashes(hashes, hashes_start,
					len, hashed_indices, num_bits, ngrams, tokens_to_skip);

			for (index_t i=0; i<max_idx; i++)
				result += vec2[hashed_indices[i]];

			hashes_start++;
			if (hashes_start==hashes.vlen)
				hashes_start = 0;
		}
	}
	doc_collection->free_feature_vector(sv, vec_idx1);
	SG_UNREF(local_tzer);
	return should_normalize ? result / CMath::sqrt((float64_t) sv.size()) : result;
}

void CHashedDocDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
	float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == CMath::pow(2,num_bits))

	if (abs_val)
		alpha = CMath::abs(alpha);

	SGVector<char> sv = doc_collection->get_feature_vector(vec_idx1);
	const float64_t value = should_normalize ? alpha / CMath::sqrt((float64_t) sv.size()) : alpha;

	/** this vector will maintain the current n+k active tokens
	 * in a circular manner */
	SGVector<uint32_t> hashes(ngrams+tokens_to_skip);
	index_t hashes_start = 0;
	index_t hashes_end = 0;
	index_t len = hashes.vlen - 1;

	/** the combinations generated from the current active tokens will be
	 * stored here to avoid creating new objects */
	SGVector<index_t> hashed_indices((ngrams-1)*(tokens_to_skip+1) + 1);

	CTokenizer* local_tzer = tokenizer->get_copy();

	/** Reading n+k-1 tokens */
	const int32_t seed = 0xdeadbeaf;
	local_tzer->set_text(sv);
	index_t start = 0;
	while (hashes_end<ngrams-1+tokens_to_skip && local_tzer->has_next())
	{
		index_t end = local_tzer->next_token_idx(start);
		uint32_t token_hash = CHash::MurmurHash3((uint8_t* ) &sv.vector[start], end-start, seed);
		hashes[hashes_end++] = token_hash;
	}

	while (local_tzer->has_next())
	{
		index_t end = local_tzer->next_token_idx(start);
		uint32_t token_hash = CHash::MurmurHash3((uint8_t* ) &sv.vector[start], end-start, seed);
		hashes[hashes_end] = token_hash;

		CHashedDocConverter::generate_ngram_hashes(hashes, hashes_start, len, hashed_indices,
				num_bits, ngrams, tokens_to_skip);

		for (index_t i=0; i<hashed_indices.vlen; i++)
			vec2[hashed_indices[i]] += value;

		hashes_start++;
		hashes_end++;
		if (hashes_end==hashes.vlen)
			hashes_end = 0;
		if (hashes_start==hashes.vlen)
			hashes_start = 0;
	}

	if (ngrams>1)
	{
		while (hashes_start!=hashes_end)
		{
			len--;
			index_t max_idx = CHashedDocConverter::generate_ngram_hashes(hashes,
					hashes_start, len, hashed_indices, num_bits, ngrams, tokens_to_skip);

			for (index_t i=0; i<max_idx; i++)
				vec2[hashed_indices[i]] += value;

			hashes_start++;
			if (hashes_start==hashes.vlen)
				hashes_start = 0;
		}
	}

	doc_collection->free_feature_vector(sv, vec_idx1);
	SG_UNREF(local_tzer);
}

uint32_t CHashedDocDotFeatures::calculate_token_hash(char* token,
		int32_t length, int32_t num_bits, uint32_t seed)
{
	int32_t hash = CHash::MurmurHash3((uint8_t* ) token, length, seed);
	return hash & ((1 << num_bits) - 1);
}

void CHashedDocDotFeatures::set_doc_collection(CStringFeatures<char>* docs)
{
	SG_UNREF(doc_collection);
	doc_collection = docs;
}

int32_t CHashedDocDotFeatures::get_nnz_features_for_vector(int32_t num)
{
	SGVector<char> sv = doc_collection->get_feature_vector(num);
	int32_t num_nnz_features = sv.size();
	doc_collection->free_feature_vector(sv, num);
	return num_nnz_features;
}

void* CHashedDocDotFeatures::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}

bool CHashedDocDotFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED;
	return false;
}

void CHashedDocDotFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED;
}

const char* CHashedDocDotFeatures::get_name() const
{
	return "HashedDocDotFeatures";
}

CFeatures* CHashedDocDotFeatures::duplicate() const
{
	return new CHashedDocDotFeatures(*this);
}

EFeatureType CHashedDocDotFeatures::get_feature_type() const
{
	return F_UINT;
}

EFeatureClass CHashedDocDotFeatures::get_feature_class() const
{
	return C_SPARSE;
}

int32_t CHashedDocDotFeatures::get_num_vectors() const
{
	return doc_collection->get_num_vectors();
}
}
