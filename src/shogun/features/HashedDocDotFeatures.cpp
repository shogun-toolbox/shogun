/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/HashedDocDotFeatures.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/Hash.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
CHashedDocDotFeatures::CHashedDocDotFeatures(int32_t hash_bits, CStringFeatures<char>* docs, 
	CTokenizer* tzer, bool normalize, int32_t size) : CDotFeatures(size)
{
	init(hash_bits, docs, tzer, normalize);
}

CHashedDocDotFeatures::CHashedDocDotFeatures(const CHashedDocDotFeatures& orig)
: CDotFeatures(orig)
{
	init(orig.num_bits, orig.doc_collection, orig.tokenizer, orig.should_normalize);
}

CHashedDocDotFeatures::CHashedDocDotFeatures(CFile* loader)
{
	SG_NOTIMPLEMENTED;
}

void CHashedDocDotFeatures::init(int32_t hash_bits, CStringFeatures<char>* docs, 
	CTokenizer* tzer, bool normalize)
{
	num_bits = hash_bits;

	doc_collection = docs;
	tokenizer = tzer;
	should_normalize = normalize;

	if (!tokenizer)
	{
		tokenizer = new CDelimiterTokenizer();
		((CDelimiterTokenizer* )tokenizer)->init_for_whitespace();
	}

	SG_ADD(&num_bits, "num_bits", "Number of bits of hash", MS_NOT_AVAILABLE);
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
			should_normalize);
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

	float64_t result = 0;
	CTokenizer* local_tzer = tokenizer->get_copy();

	const int32_t seed = 0xdeadbeaf;
	local_tzer->set_text(sv);
	index_t start = 0;
	while (local_tzer->has_next())
	{
		index_t end = local_tzer->next_token_idx(start);
		uint32_t hashed_idx = calculate_token_hash(&sv.vector[start], end-start, num_bits, seed);
		result += vec2[hashed_idx];
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
	CTokenizer* local_tzer = tokenizer->get_copy();

	const float64_t value = should_normalize ? alpha / CMath::sqrt((float64_t) sv.size()) : alpha;
	const int32_t seed = 0xdeadbeaf;
	index_t start = 0;
	local_tzer->set_text(sv);
	while (local_tzer->has_next())
	{
		index_t end = local_tzer->next_token_idx(start);
		uint32_t hashed_idx = calculate_token_hash(&sv.vector[start], end-start, num_bits, seed);
		vec2[hashed_idx] += value;
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
