/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/converter/HashedDocConverter.h>
#include <shogun/lib/DelimiterTokenizer.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/HashedDocDotFeatures.h>
#include <shogun/mathematics/Math.h>


using namespace shogun;

namespace shogun
{
CHashedDocConverter::CHashedDocConverter() : CConverter()
{
	init(NULL, 16, false, 1, 0);
}

CHashedDocConverter::CHashedDocConverter(int32_t hash_bits, bool normalize,
	int32_t n_grams, int32_t skips) : CConverter()
{
	init(NULL, hash_bits, normalize, n_grams, skips);
}

CHashedDocConverter::CHashedDocConverter(CTokenizer* tzer,
	int32_t hash_bits, bool normalize, int32_t n_grams, int32_t skips) : CConverter()
{
	init(tzer, hash_bits, normalize, n_grams, skips);
}

CHashedDocConverter::~CHashedDocConverter()
{
	SG_UNREF(tokenizer);
}
	
void CHashedDocConverter::init(CTokenizer* tzer, int32_t hash_bits, bool normalize,
	int32_t n_grams, int32_t skips)
{
	num_bits = hash_bits;
	should_normalize = normalize;
	ngrams = n_grams;
	tokens_to_skip = skips;

	if (tzer==NULL)
	{
		CDelimiterTokenizer* tk = new CDelimiterTokenizer();
		tk->delimiters[(uint8_t) ' '] = 1;
		tk->delimiters[(uint8_t) '\t'] = 1;
		tokenizer = tk;
	}
	else
		tokenizer = tzer;

	SG_REF(tokenizer);
	SG_ADD(&num_bits, "num_bits", "Number of bits of the hash",
		MS_NOT_AVAILABLE);
	SG_ADD(&ngrams, "ngrams", "Number of consecutive tokens",
		MS_NOT_AVAILABLE);
	SG_ADD(&tokens_to_skip, "tokens_to_skip", "Number of tokens to skip",
		MS_NOT_AVAILABLE);
	SG_ADD(&should_normalize, "should_normalize", "Whether to normalize vectors or not",
		MS_NOT_AVAILABLE);
	m_parameters->add((CSGObject**) &tokenizer, "tokenizer",
		"Tokenizer");
}

const char* CHashedDocConverter::get_name() const
{
	return "HashedDocConverter";
}
	
CFeatures* CHashedDocConverter::apply(CFeatures* features)
{
	ASSERT(features);
	if (strcmp(features->get_name(), "StringFeatures")!=0)
		SG_ERROR("CHashedConverter::apply() : CFeatures object passed is not of type CStringFeatures.");

	CStringFeatures<char>* s_features = (CStringFeatures<char>*) features;

	int32_t dim = CMath::pow(2, num_bits);	
	SGSparseMatrix<float64_t> matrix(dim,features->get_num_vectors());
	for (index_t vec_idx=0; vec_idx<s_features->get_num_vectors(); vec_idx++)
	{
		SGVector<char> doc = s_features->get_feature_vector(vec_idx);
		matrix[vec_idx] = apply(doc);
		s_features->free_feature_vector(doc, vec_idx);
	}

	return (CFeatures*) new CSparseFeatures<float64_t>(matrix);
}

SGSparseVector<float64_t> CHashedDocConverter::apply(SGVector<char> document)
{
	ASSERT(document.size()>0)
	const int32_t array_size = 1024*1024;
	CDynamicArray<uint32_t> hashed_indices(array_size);
	CDynamicArray<uint32_t>* cached_hashes = new CDynamicArray<uint32_t>(ngrams);
	CDynamicArray<index_t>* ngram_indices = new CDynamicArray<index_t>(ngrams*(ngrams+1)/2*(1+tokens_to_skip));

	const int32_t seed = 0xdeadbeaf;
	tokenizer->set_text(document);
	index_t token_start = 0;
	int32_t n = 0;

	/** Reading n+s-1 tokens */
	while (n<ngrams-1+tokens_to_skip && tokenizer->has_next())
	{
		index_t end = tokenizer->next_token_idx(token_start);
		uint32_t token_hash = CHash::MurmurHash3((uint8_t* ) &document.vector[token_start],
				end-token_start, seed);
		cached_hashes->append_element(token_hash);
		n++;
	}

	/** Reading token and storing index to hashed_indices */
	while (tokenizer->has_next())
	{
		index_t end = tokenizer->next_token_idx(token_start);
		uint32_t token_hash = CHash::MurmurHash3((uint8_t* ) &document.vector[token_start],
				end-token_start, seed);
		cached_hashes->append_element(token_hash);
		CHashedDocConverter::generate_ngram_hashes(cached_hashes, ngram_indices, num_bits,
				ngrams, tokens_to_skip);

		for (index_t i=0; i<ngram_indices->get_num_elements(); i++)
			hashed_indices.append_element(ngram_indices->get_element(i));

		cached_hashes->delete_element(0);
	}

	/** For remaining combinations */
	if (ngrams>1)
	{
		while (cached_hashes->get_num_elements()>0)
		{
			CHashedDocConverter::generate_ngram_hashes(cached_hashes, ngram_indices, num_bits,
					ngrams, tokens_to_skip);

			for (index_t i=0; i<ngram_indices->get_num_elements(); i++)
				hashed_indices.append_element(ngram_indices->get_element(i));

			cached_hashes->delete_element(0);
		}
	}
	SG_UNREF(cached_hashes);
	SG_UNREF(ngram_indices);

	SGSparseVector<float64_t> sparse_doc_rep = create_hashed_representation(hashed_indices);	

	/** Normalizing vector */
	if (should_normalize)
	{
		float64_t norm_const = CMath::sqrt((float64_t) document.size());
		for (index_t i=0; i<sparse_doc_rep.num_feat_entries; i++)
			sparse_doc_rep.features[i].entry /= norm_const; 
	}
	
	return sparse_doc_rep;
}

SGSparseVector<float64_t> CHashedDocConverter::create_hashed_representation(CDynamicArray<uint32_t>& hashed_indices)
{
	int32_t num_nnz_features = count_distinct_indices(hashed_indices);

	SGSparseVector<float64_t> sparse_doc_rep(num_nnz_features);
	index_t sparse_idx = 0;
	for (index_t i=0; i<hashed_indices.get_num_elements(); i++)
	{
		sparse_doc_rep.features[sparse_idx].feat_index = hashed_indices[i];
		sparse_doc_rep.features[sparse_idx].entry = 1;
		while ( (i+1<hashed_indices.get_num_elements()) && 
				(hashed_indices[i+1]==hashed_indices[i]) )
		{
			sparse_doc_rep.features[sparse_idx].entry++;
			i++;
		}
		sparse_idx++;
	}
	return sparse_doc_rep;
}

void CHashedDocConverter::generate_ngram_hashes(CDynamicArray<uint32_t>* hashes,
	CDynamicArray<index_t>* ngram_hashes, int32_t num_bits, int32_t ngrams, int32_t tokens_to_skip)
{
	while (ngram_hashes->get_num_elements()>0)
		ngram_hashes->delete_element(0);

	ngram_hashes->append_element(hashes->get_element(0) & ((1 << num_bits) -1));
	for (index_t n=1; n<ngrams; n++)
	{
		for (index_t s=0; s<=tokens_to_skip; s++)
		{
			if ( n+s >= hashes->get_num_elements())
				break;

			uint32_t ngram_hash = hashes->get_element(0);
			for (index_t i=1+s; i<=n+s; i++)
				ngram_hash = ngram_hash ^ hashes->get_element(i);
			ngram_hash = ngram_hash & ((1 << num_bits) - 1);
			ngram_hashes->append_element(ngram_hash);
		}
	}
}

int32_t CHashedDocConverter::count_distinct_indices(CDynamicArray<uint32_t>& hashed_indices)
{
	CMath::qsort(hashed_indices.get_array(), hashed_indices.get_num_elements());

	/** Counting nnz features */
	int32_t num_nnz_features = 0;
	for (index_t i=0; i<hashed_indices.get_num_elements(); i++)
	{
		num_nnz_features++;
		while ( (i+1<hashed_indices.get_num_elements()) && 
				(hashed_indices[i+1]==hashed_indices[i]) )
		{
			i++;
		}
	}
	return num_nnz_features;	
}
}
