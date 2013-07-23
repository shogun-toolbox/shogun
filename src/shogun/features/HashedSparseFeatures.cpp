/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/HashedSparseFeatures.h>
#include <shogun/features/HashedDenseFeatures.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/Hash.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/mathematics/Math.h>
#include <string.h>
#include <iostream>

namespace shogun {

template <class ST>
CHashedSparseFeatures<ST>::CHashedSparseFeatures(int32_t size)
: CDotFeatures(size)
{
	init(NULL, 0);
}

template <class ST>
CHashedSparseFeatures<ST>::CHashedSparseFeatures(CSparseFeatures<ST>* feats, int32_t d)
 : CDotFeatures()
{
	init(feats, d);
}

template <class ST>
CHashedSparseFeatures<ST>::CHashedSparseFeatures(SGSparseMatrix<ST> matrix, int32_t d)
: CDotFeatures()
{
	CSparseFeatures<ST>* feats = new CSparseFeatures<ST>(matrix);
	init(feats, d);
}

template <class ST>
CHashedSparseFeatures<ST>::CHashedSparseFeatures(CFile* loader, int32_t d)
: CDotFeatures(loader)
{
	CSparseFeatures<ST>* feats = new CSparseFeatures<ST>();
	feats->load(loader);
	init(feats, d);
}

template <class ST>
void CHashedSparseFeatures<ST>::init(CSparseFeatures<ST>* feats, int32_t d)
{
	dim = d;
	sparse_feats = feats;
	SG_REF(sparse_feats);
	SG_ADD(&dim, "dim", "Dimension of new feature space", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject** ) &sparse_feats, "sparse_feats ", "Sparse features to work on",
		MS_NOT_AVAILABLE);

	set_generic<ST>();
}

template <class ST>
CHashedSparseFeatures<ST>::CHashedSparseFeatures(const CHashedSparseFeatures& orig)
: CDotFeatures(orig)
{
	init(orig.sparse_feats, orig.dim);
}

template <class ST>
CHashedSparseFeatures<ST>::~CHashedSparseFeatures()
{
	SG_UNREF(sparse_feats);
}

template <class ST>
CFeatures* CHashedSparseFeatures<ST>::duplicate() const
{
	return new CHashedSparseFeatures<ST>(*this);
}

template <class ST>
int32_t CHashedSparseFeatures<ST>::get_dim_feature_space() const
{
	return dim;
}

template <class ST>
SGSparseVector<ST> CHashedSparseFeatures<ST>::get_hashed_feature_vector(
	int32_t vec_idx) const
{
	return CHashedSparseFeatures<ST>::hash_vector(sparse_feats->get_sparse_feature_vector(vec_idx), dim);
}

template <class ST>
SGSparseVector<ST> CHashedSparseFeatures<ST>::hash_vector(SGVector<ST> vec, int32_t dim)
{
	return CHashedDenseFeatures<ST>::hash_vector(vec, dim);
}

template <class ST>
SGSparseVector<ST> CHashedSparseFeatures<ST>::hash_vector(SGSparseVector<ST> vec, int32_t dim)
{
	SGVector<ST> h_vec(dim);
	SGVector<ST>::fill_vector(h_vec, dim, 0);
	for (index_t i=0; i<vec.num_feat_entries; i++)
	{
		uint32_t h = CHash::MurmurHash3((uint8_t* ) &vec.features[i].feat_index, sizeof (index_t),
						vec.features[i].feat_index);
		h_vec[h % dim] += vec.features[i].entry;
	}

	int32_t num_nnz_features = 0; 
	for (index_t i=0; i<dim; i++)
	{
		if (h_vec[i]!=0)
			num_nnz_features++;
	}

	SGSparseVector<ST> sv(num_nnz_features);

	int32_t sparse_index = 0;	
	for (index_t i=0; i<dim; i++)
	{
		if (h_vec[i]!=0)
		{
			sv.features[sparse_index].entry = h_vec[i];
			sv.features[sparse_index++].feat_index = i;
		}
	}

	return sv;
}

template <class ST>
float64_t CHashedSparseFeatures<ST>::dot(int32_t vec_idx1, CDotFeatures* df,
	int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	ASSERT(strcmp(df->get_name(), get_name())==0)
	
	CHashedSparseFeatures<ST>* feats = (CHashedSparseFeatures<ST>* ) df;
	SGSparseVector<ST> vec_1 = get_hashed_feature_vector(vec_idx1);
	SGSparseVector<ST> vec_2 = feats->get_hashed_feature_vector(vec_idx2);

	float64_t result = vec_1.sparse_dot(vec_2); 
	return result;	
}

template <class ST>
float64_t CHashedSparseFeatures<ST>::dense_dot(int32_t vec_idx1, const float64_t* vec2,
	int32_t vec2_len)
{
	ASSERT(vec2_len == dim)

	SGSparseVector<ST> vec = sparse_feats->get_sparse_feature_vector(vec_idx1);

	float64_t result = 0;
	for (index_t i=0; i<vec.num_feat_entries; i++)
	{
		uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &vec.features[i].feat_index, sizeof (index_t),
					   vec.features[i].feat_index);
		h_idx = h_idx % dim;
		result += vec2[h_idx] * vec.features[i].entry;
	}

	sparse_feats ->free_feature_vector(vec_idx1);
	return result;
}

template <class ST>
void CHashedSparseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
	float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	float64_t val = abs_val ? CMath::abs(alpha) : alpha;
	ASSERT(vec2_len == dim)
	
	SGSparseVector<ST> vec = sparse_feats->get_sparse_feature_vector(vec_idx1);

	for (index_t i=0; i<vec.num_feat_entries; i++)
	{
		uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &vec.features[i].feat_index, sizeof (index_t),
					   vec.features[i].feat_index);
		vec2[h_idx%dim] += val * vec.features[i].entry;
	}
	sparse_feats ->free_feature_vector(vec_idx1);	
}

template <class ST>
int32_t CHashedSparseFeatures<ST>::get_nnz_features_for_vector(int32_t num)
{
	return dim;
}

template <class ST>
void* CHashedSparseFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}
template <class ST>
bool CHashedSparseFeatures<ST>::get_next_feature(int32_t& index, float64_t& value,
	void* iterator)
{
	SG_NOTIMPLEMENTED;
	return false;
}
template <class ST>
void CHashedSparseFeatures<ST>::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED;
}

template <class ST>
const char* CHashedSparseFeatures<ST>::get_name() const
{
	return "HashedSparseFeatures";
}

template <class ST>
EFeatureType CHashedSparseFeatures<ST>::get_feature_type() const
{
	return F_UINT;
}

template <class ST>
EFeatureClass CHashedSparseFeatures<ST>::get_feature_class() const
{
	return C_SPARSE;
}

template <class ST>
int32_t CHashedSparseFeatures<ST>::get_num_vectors() const
{
	return sparse_feats ->get_num_vectors();
}

template class CHashedSparseFeatures <bool>;
template class CHashedSparseFeatures <char>;
template class CHashedSparseFeatures <int8_t>;
template class CHashedSparseFeatures <uint8_t>;
template class CHashedSparseFeatures <int16_t>;
template class CHashedSparseFeatures <uint16_t>;
template class CHashedSparseFeatures <int32_t>;
template class CHashedSparseFeatures <uint32_t>;
template class CHashedSparseFeatures <int64_t>;
template class CHashedSparseFeatures <uint64_t>;
template class CHashedSparseFeatures <float32_t>;
template class CHashedSparseFeatures <float64_t>;
template class CHashedSparseFeatures <floatmax_t>;
}
