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
#include <shogun/lib/DynamicArray.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

#include <string.h>

namespace shogun {
template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(int32_t size, bool use_quadr, bool keep_lin_terms)
: CDotFeatures(size)
{
	init(NULL, 0, use_quadr, keep_lin_terms);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(CDenseFeatures<ST>* feats, int32_t d,
	bool use_quadr, bool keep_lin_terms) : CDotFeatures()
{
	init(feats, d, use_quadr, keep_lin_terms);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(SGMatrix<ST> matrix, int32_t d, bool use_quadr,
	bool keep_lin_terms) : CDotFeatures()
{
	CDenseFeatures<ST>* feats = new CDenseFeatures<ST>(matrix);
	init(feats, d, use_quadr, keep_lin_terms);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(ST* src, int32_t num_feat, int32_t num_vec,
	int32_t d, bool use_quadr, bool keep_lin_terms) : CDotFeatures()
{
	CDenseFeatures<ST>* feats = new CDenseFeatures<ST>(src, num_feat, num_vec);
	init(feats, d, use_quadr, keep_lin_terms);
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(CFile* loader, int32_t d, bool use_quadr,
	bool keep_lin_terms) : CDotFeatures(loader)
{
	CDenseFeatures<ST>* feats = new CDenseFeatures<ST>();
	feats->load(loader);
	init(feats, d, use_quadr, keep_lin_terms);
}

template <class ST>
void CHashedDenseFeatures<ST>::init(CDenseFeatures<ST>* feats, int32_t d, bool use_quadr,
	bool keep_lin_terms)
{
	dim = d;
	dense_feats = feats;
	SG_REF(dense_feats);
	use_quadratic = use_quadr;
	keep_linear_terms = keep_lin_terms;

	SG_ADD(&use_quadratic, "use_quadratic", "Whether to use quadratic features",
		MS_NOT_AVAILABLE);
	SG_ADD(&keep_linear_terms, "keep_linear_terms", "Whether to keep the linear terms or not",
		MS_NOT_AVAILABLE);
	SG_ADD(&dim, "dim", "Dimension of new feature space", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject** ) &dense_feats, "dense_feats", "Dense features to work on",
		MS_NOT_AVAILABLE);

	set_generic<ST>();
}

template <class ST>
CHashedDenseFeatures<ST>::CHashedDenseFeatures(const CHashedDenseFeatures& orig)
: CDotFeatures(orig)
{
	init(orig.dense_feats, orig.dim, orig.use_quadratic, orig.keep_linear_terms);
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

	SGSparseVector<uint32_t> vec_1 = get_hashed_feature_vector(vec_idx1);
	SGSparseVector<uint32_t> vec_2 = feats->get_hashed_feature_vector(vec_idx2);

	float64_t result = vec_1.sparse_dot(vec_2);
//	result /= (float64_t) vec_1.num_feat_entries * vec_2.num_feat_entries;	
	return result;	
}

template <class ST>
float64_t CHashedDenseFeatures<ST>::dense_dot(int32_t vec_idx1, const float64_t* vec2,
	int32_t vec2_len)
{
	ASSERT(vec2_len == dim)

	SGVector<ST> vec = dense_feats->get_feature_vector(vec_idx1);

	float64_t result = 0;
	if ( (!use_quadratic) || (use_quadratic && keep_linear_terms))
		for (index_t i=0; i<vec.vlen; i++)
		{
			uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &vec[i], sizeof (ST), i);
			result += vec2[h_idx % dim];
		}

	if (use_quadratic)
		for (index_t i=0; i<vec.size(); i++)
			for (index_t j=i; j<vec.size(); j++)
			{
				ST value = (i != j) ? 2 * vec[i] * vec[j] : vec[i] * vec[j];
				uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &value, sizeof (ST), i+j);
				result += vec2[h_idx % dim];
			}
	
	dense_feats->free_feature_vector(vec, vec_idx1);
//	result /= (float64_t) vec.size() * (vec.size()+1) / 2;
	return result;
}

template <class ST>
void CHashedDenseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
	float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	float64_t val = abs_val ? CMath::abs(alpha) : alpha;
//	val /=  (float64_t) vec2_len * (vec2_len+1) / 2;
	ASSERT(vec2_len == dim)
	
	SGVector<ST> vec = dense_feats->get_feature_vector(vec_idx1);
	if ( (!use_quadratic) || (use_quadratic && keep_linear_terms))
		for (index_t i=0; i<vec.vlen; i++)
		{
			uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &vec[i], sizeof (ST), i);
			vec2[h_idx % dim] += val;
		}

	if (use_quadratic)
		for (index_t i=0; i<vec.size(); i++)
			for (index_t j=i; j<vec.size(); j++)
			{
				ST value = (i != j) ? 2 * vec[i] * vec[j] : vec[i] * vec[j];
				uint32_t h_idx = CHash::MurmurHash3((uint8_t* ) &value, sizeof (ST), i+j);
				vec2[h_idx % dim] += val;
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
	return false;
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

template <class ST>
SGSparseVector<uint32_t> CHashedDenseFeatures<ST>::get_hashed_feature_vector(int32_t vec_idx)
{
	SGVector<ST> vec = dense_feats->get_feature_vector(vec_idx);
	SGSparseVector<uint32_t> hashed_vec = CHashedDenseFeatures<ST>::hash_vector(
			vec, dim, use_quadratic);
	dense_feats->free_feature_vector(vec, vec_idx);
	return hashed_vec;
}

template <class ST>
SGSparseVector<uint32_t> CHashedDenseFeatures<ST>::hash_vector(SGVector<ST> vec, int32_t dim,
	bool use_quadr, bool keep_lin_terms)
{
	CDynamicArray<ST> indices(vec.size());
	if ( (!use_quadr) || (use_quadr && keep_lin_terms) ) 
		for (index_t i=0; i<vec.size(); i++)
		{
			uint32_t hash = CHash::MurmurHash3((uint8_t* ) &vec[i], sizeof (ST), i);
			indices.append_element(hash % dim);
		}

	if (use_quadr)
		for (index_t i=0; i<vec.size(); i++)
			for (index_t j=i; j<vec.size(); j++)
			{
				ST value = (i != j) ? 2 * vec[i] * vec[j] : vec[i] * vec[j];
				uint32_t hash = CHash::MurmurHash3((uint8_t* ) &value, sizeof (ST), i+j);
				indices.append_element(hash % dim);	
			}

	CMath::qsort(indices.get_array(), indices.get_num_elements());

	int32_t num_different_indices = 0;
	for(index_t i=0; i<indices.get_num_elements(); i++)
	{
		num_different_indices++;
		while ( (i+1 < indices.get_num_elements()) &&
				(indices[i+1] == indices[i]) )
			i++;
	}

	SGSparseVector<uint32_t> hashed_vector(num_different_indices);

	int32_t sparse_feat_index = 0;
	for (index_t i=0; i<indices.get_num_elements(); i++)
	{
		int32_t count = 1;
		while ( (i+1 < indices.get_num_elements()) &&
				(indices[i+1] == indices[i]) )
		{
			count++;
			i++;
		}
		hashed_vector.features[sparse_feat_index].feat_index = indices[i];
		hashed_vector.features[sparse_feat_index++].entry = count;
	}

	return hashed_vector;
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
}
