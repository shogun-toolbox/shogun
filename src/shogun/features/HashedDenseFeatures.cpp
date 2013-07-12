/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/HashedDenseFeatures.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/Hash.h>
#include <shogun/io/SGIO.h>

#include <string.h>
#include <iostream>

using namespace shogun;

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(int32_t size)
: CDotFeatures(size)
{
	init(NULL, 0);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(CDenseFeatures<ST>* feats, int32_t d)
 : CDotFeatures()
{
	init(feats, d);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(SGMatrix<ST> matrix, int32_t d)
: CDotFeatures()
{
	CDenseFeatures<ST>* feats = new CDenseFeatures<ST>(matrix);
	init(feats, d);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(ST* src, int32_t num_feat, int32_t num_vec,
	int32_t d) : CDotFeatures()
{
	CDenseFeatures<ST>* feats = new CDenseFeatures<ST>(src, num_feat, num_vec);
	init(feats, d);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(CFile* loader, int32_t d)
: CDotFeatures(loader)
{
	init(NULL, d);
}

template <class ST>
void CHashedDenseFeatures<ST>::init(CDenseFeatures<ST>* feats, int32_t d)
{
	dim = d;
	dense_feats = feats;
	SG_REF(dense_feats);
	SG_ADD(&dim, "dim", "Dimension of new feature space", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject** ) &dense_feats, "dense_feats", "Dense features to work on",
		MS_NOT_AVAILABLE);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(const CHashedDenseFeatures& orig)
: CDotFeatures(orig)
{
	init(orig.dense_feats, orig.dim);
}

template <class ST>
CHashedDenseFeatures<ST>::~CHashedDenseFeatures()
{
	SG_UNREF(dense_feats);
}

template <class ST>
CFeatures* CHashedDenseFeatures<ST>::duplicate() const
{
	return new CHashedDenseFeatures<ST>(*this);
}

template <class ST>
int32_t CHashedDenseFeatures<ST>::get_dim_feature_space() const
{
	return dim;
}

template <class ST>
float64_t CHashedDenseFeatures<ST>::dot(int32_t vec_idx1, CDotFeatures* df,
	int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	ASSERT(strcmp(df->get_name(), get_name())==0)

	CHashedDenseFeatures<ST>* feats = (CHashedDenseFeatures<ST>* ) df;
	ASSERT(feats->get_dim_feature_space() == get_dim_feature_space())
	SGVector<ST> vec_1 = dense_feats->get_feature_vector(vec_idx1);
	SGVector<ST> vec_2 = feats->dense_feats->get_feature_vector(vec_idx2);

	SGVector<ST> h_vec_1(dim);
	SGVector<ST> h_vec_2(dim);
	
	SGVector<ST>::fill_vector(h_vec_1, dim, 0);
	SGVector<ST>::fill_vector(h_vec_2, dim, 0);

	for (index_t i=0; i<vec_1.vlen; i++)
	{
		uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &vec_1[i], sizeof (ST), i);
		h_vec_1[h_idx%dim]++;
		h_idx = CHash::MurmurHash3((uint8_t* ) &vec_2[i], sizeof (ST), i);
		h_vec_2[h_idx%dim]++;
		
	}

	float64_t result = SGVector<ST>::dot(h_vec_1.vector, h_vec_2.vector, dim);
	
	dense_feats->free_feature_vector(vec_1, vec_idx1);
	feats->dense_feats->free_feature_vector(vec_2, vec_idx2);
	
	return result;	
}

template <class ST>
float64_t CHashedDenseFeatures<ST>::dense_dot(int32_t vec_idx1, const float64_t* vec2,
	int32_t vec2_len)
{
	ASSERT(vec2_len == dim)

	SGVector<ST> vec = dense_feats->get_feature_vector(vec_idx1);

	float64_t result = 0;
	for (index_t i=0; i<vec.vlen; i++)
	{
		uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &vec[i], sizeof (ST), i);
		result += vec2[h_idx%dim];
	}

	dense_feats->free_feature_vector(vec, vec_idx1);
	return result;
}

template <class ST>
void CHashedDenseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
	float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	float64_t val = abs_val ? CMath::abs(alpha) : alpha;
	ASSERT(vec2_len == dim)
	
	SGVector<ST> vec = dense_feats->get_feature_vector(vec_idx1);

	for (index_t i=0; i<vec.vlen; i++)
	{
		uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &vec[i], sizeof (ST), i);
		vec2[h_idx%dim] += val;
	}
	dense_feats->free_feature_vector(vec, vec_idx1);	
}

template <class ST>
int32_t CHashedDenseFeatures<ST>::get_nnz_features_for_vector(int32_t num)
{
	return dim;
}

template <class ST>
void* CHashedDenseFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}
template <class ST>
bool CHashedDenseFeatures<ST>::get_next_feature(int32_t& index, float64_t& value,
	void* iterator)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}
template <class ST>
void CHashedDenseFeatures<ST>::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED;
}

template <class ST>
const char* CHashedDenseFeatures<ST>::get_name() const
{
	return "HashedDenseFeatures";
}

template <class ST>
EFeatureType CHashedDenseFeatures<ST>::get_feature_type() const
{
	return F_UINT;
}

template <class ST>
EFeatureClass CHashedDenseFeatures<ST>::get_feature_class() const
{
	return C_SPARSE;
}

template <class ST>
int32_t CHashedDenseFeatures<ST>::get_num_vectors() const
{
	return dense_feats->get_num_vectors();
}

template class CHashedDenseFeatures<bool>;
template class CHashedDenseFeatures<char>;
template class CHashedDenseFeatures<int8_t>;
template class CHashedDenseFeatures<uint8_t>;
template class CHashedDenseFeatures<int16_t>;
template class CHashedDenseFeatures<uint16_t>;
template class CHashedDenseFeatures<int32_t>;
template class CHashedDenseFeatures<uint32_t>;
template class CHashedDenseFeatures<int64_t>;
template class CHashedDenseFeatures<uint64_t>;
template class CHashedDenseFeatures<float32_t>;
template class CHashedDenseFeatures<float64_t>;
template class CHashedDenseFeatures<floatmax_t>;
