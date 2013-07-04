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
	init(NULL, 50);
}

CHashedDocConverter::CHashedDocConverter(int32_t d)
: CConverter()
{
	init(NULL, d);
}

CHashedDocConverter::CHashedDocConverter(CTokenizer* tzer,
	int32_t d) : CConverter()
{
	init(tzer, d);
}

CHashedDocConverter::~CHashedDocConverter()
{
	SG_UNREF(tokenizer);
}
	
void CHashedDocConverter::init(CTokenizer* tzer, int32_t d)
{
	dim = d;
	num_bits = (int32_t) CMath::ceil(CMath::log2(d));

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
	SG_ADD(&dim, "dim", "Dimension of target feature space",
		MS_NOT_AVAILABLE);
	SG_ADD(&num_bits, "num_bits", "Number of bits of the hash",
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
	
	SGSparseMatrix<uint32_t> matrix(dim,features->get_num_vectors());
	for (index_t vec_idx=0; vec_idx<s_features->get_num_vectors(); vec_idx++)
	{
		SGVector<char> doc = s_features->get_feature_vector(vec_idx);
		matrix[vec_idx] = apply(doc);
		s_features->free_feature_vector(doc, vec_idx);
	}

	return (CFeatures*) new CSparseFeatures<uint32_t>(matrix);
}

SGSparseVector<uint32_t> CHashedDocConverter::apply(SGVector<char> document)
{
	ASSERT(document.size()>0)
	const int32_t array_size = 1024*1024;
	CDynamicArray<uint32_t> hashed_indices(array_size);

	const int32_t seed = 0xdeadbeaf;
	tokenizer->set_text(document);
	index_t token_start = 0;
	while (tokenizer->has_next())
	{
		index_t next_token_idx = tokenizer->next_token_idx(token_start);		
		uint32_t hashed_idx = CHashedDocDotFeatures::calculate_token_hash(
				&document.vector[token_start], next_token_idx-token_start, num_bits, seed);
		hashed_indices.push_back(hashed_idx);
	}

	CMath::qsort(hashed_indices.get_array(), hashed_indices.get_num_elements());
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

	SGSparseVector<uint32_t> sparse_doc_rep(num_nnz_features);
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
}
