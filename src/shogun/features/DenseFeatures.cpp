/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011-2013 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/preprocessor/DensePreprocessor.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>

#include <string.h>

namespace shogun {

template<class ST> CDenseFeatures<ST>::CDenseFeatures(int32_t size) : CDotFeatures(size)
{
	init();
}

template<class ST> CDenseFeatures<ST>::CDenseFeatures(const CDenseFeatures & orig) :
		CDotFeatures(orig)
{
	init();
	set_feature_matrix(orig.feature_matrix);
	initialize_cache();

	if (orig.m_subset_stack != NULL)
	{
		SG_UNREF(m_subset_stack);
		m_subset_stack=new CSubsetStack(*orig.m_subset_stack);
		SG_REF(m_subset_stack);
	}
}

template<class ST> CDenseFeatures<ST>::CDenseFeatures(SGMatrix<ST> matrix) :
		CDotFeatures()
{
	init();
	set_feature_matrix(matrix);
}

template<class ST> CDenseFeatures<ST>::CDenseFeatures(ST* src, int32_t num_feat, int32_t num_vec) :
		CDotFeatures()
{
	init();
	set_feature_matrix(SGMatrix<ST>(src, num_feat, num_vec));
}
template<class ST> CDenseFeatures<ST>::CDenseFeatures(CFile* loader) :
		CDotFeatures()
{
	init();
	load(loader);
}

template<class ST> CFeatures* CDenseFeatures<ST>::duplicate() const
{
	return new CDenseFeatures<ST>(*this);
}

template<class ST> CDenseFeatures<ST>::~CDenseFeatures()
{
	free_features();
}

template<class ST> void CDenseFeatures<ST>::free_features()
{
	m_subset_stack->remove_all_subsets();
	free_feature_matrix();
	SG_UNREF(feature_cache);
}

template<class ST> void CDenseFeatures<ST>::free_feature_matrix()
{
	m_subset_stack->remove_all_subsets();
	feature_matrix=SGMatrix<ST>();
	num_vectors = 0;
	num_features = 0;
}

template<class ST> ST* CDenseFeatures<ST>::get_feature_vector(int32_t num, int32_t& len, bool& dofree)
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	len = num_features;

	if (feature_matrix.matrix)
	{
		dofree = false;
		return &feature_matrix.matrix[real_num * int64_t(num_features)];
	}

	ST* feat = NULL;
	dofree = false;

	if (feature_cache)
	{
		feat = feature_cache->lock_entry(real_num);

		if (feat)
			return feat;
		else
			feat = feature_cache->set_entry(real_num);
	}

	if (!feat)
		dofree = true;
	feat = compute_feature_vector(num, len, feat);

	if (get_num_preprocessors())
	{
		int32_t tmp_len = len;
		ST* tmp_feat_before = feat;
		ST* tmp_feat_after = NULL;

		for (int32_t i = 0; i < get_num_preprocessors(); i++)
		{
			CDensePreprocessor<ST>* p =
					(CDensePreprocessor<ST>*) get_preprocessor(i);
			// temporary hack
			SGVector<ST> applied = p->apply_to_feature_vector(
					SGVector<ST>(tmp_feat_before, tmp_len));
			tmp_feat_after = applied.vector;
			SG_UNREF(p);

			if (i != 0) // delete feature vector, except for the the first one, i.e., feat
				SG_FREE(tmp_feat_before);
			tmp_feat_before = tmp_feat_after;
		}

		// note: tmp_feat_after should be checked as it is used by memcpy
		if (tmp_feat_after)
		{
			memcpy(feat, tmp_feat_after, sizeof(ST) * tmp_len);
			SG_FREE(tmp_feat_after);

			len = tmp_len;
		}
	}
	return feat;
}

template<class ST> void CDenseFeatures<ST>::set_feature_vector(SGVector<ST> vector, int32_t num)
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (num>=get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
		"requested %d)\n", get_num_vectors(), num);
	}

	if (!feature_matrix.matrix)
		SG_ERROR("Requires a in-memory feature matrix\n")

	if (vector.vlen != num_features)
		SG_ERROR(
				"Vector not of length %d (has %d)\n", num_features, vector.vlen);

	memcpy(&feature_matrix.matrix[real_num * int64_t(num_features)], vector.vector,
			int64_t(num_features) * sizeof(ST));
}

template<class ST> SGVector<ST> CDenseFeatures<ST>::get_feature_vector(int32_t num)
{
	/* index conversion for subset, only for array access */
	int32_t real_num=m_subset_stack->subset_idx_conversion(num);

	if (num >= get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
		"requested %d)\n", get_num_vectors(), real_num);
	}

	int32_t vlen;
	bool do_free;
	ST* vector= get_feature_vector(num, vlen, do_free);
	return SGVector<ST>(vector, vlen, do_free);
}

template<class ST> void CDenseFeatures<ST>::free_feature_vector(ST* feat_vec, int32_t num, bool dofree)
{
	if (feature_cache)
		feature_cache->unlock_entry(m_subset_stack->subset_idx_conversion(num));

	if (dofree)
		SG_FREE(feat_vec);
}

template<class ST> void CDenseFeatures<ST>::free_feature_vector(SGVector<ST> vec, int32_t num)
{
	free_feature_vector(vec.vector, num, false);
	vec=SGVector<ST>();
}

template<class ST> void CDenseFeatures<ST>::vector_subset(int32_t* idx, int32_t idx_len)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call vector_subset\n")

	ASSERT(feature_matrix.matrix)
	ASSERT(idx_len<=num_vectors)

	int32_t num_vec = num_vectors;
	num_vectors = idx_len;

	int32_t old_ii = -1;

	for (int32_t i = 0; i < idx_len; i++)
	{
		int32_t ii = idx[i];
		ASSERT(old_ii<ii)

		if (ii < 0 || ii >= num_vec)
			SG_ERROR("Index out of range: should be 0<%d<%d\n", ii, num_vec)

		if (i == ii)
			continue;

		memcpy(&feature_matrix.matrix[int64_t(num_features) * i],
				&feature_matrix.matrix[int64_t(num_features) * ii],
				num_features * sizeof(ST));
		old_ii = ii;
	}
}

template<class ST> void CDenseFeatures<ST>::feature_subset(int32_t* idx, int32_t idx_len)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call feature_subset\n")

	ASSERT(feature_matrix.matrix)
	ASSERT(idx_len<=num_features)
	int32_t num_feat = num_features;
	num_features = idx_len;

	for (int32_t i = 0; i < num_vectors; i++)
	{
		ST* src = &feature_matrix.matrix[int64_t(num_feat) * i];
		ST* dst = &feature_matrix.matrix[int64_t(num_features) * i];

		int32_t old_jj = -1;
		for (int32_t j = 0; j < idx_len; j++)
		{
			int32_t jj = idx[j];
			ASSERT(old_jj<jj)
			if (jj < 0 || jj >= num_feat)
				SG_ERROR(
						"Index out of range: should be 0<%d<%d\n", jj, num_feat);

			dst[j] = src[jj];
			old_jj = jj;
		}
	}
}

template<class ST> SGMatrix<ST> CDenseFeatures<ST>::get_feature_matrix()
{
	if (!m_subset_stack->has_subsets())
		return feature_matrix;

	SGMatrix<ST> submatrix(num_features, get_num_vectors());

	/* copy a subset vector wise */
	for (int32_t i=0; i<submatrix.num_cols; ++i)
	{
		int32_t real_i = m_subset_stack->subset_idx_conversion(i);
		memcpy(&submatrix.matrix[i*int64_t(num_features)],
				&feature_matrix.matrix[real_i * int64_t(num_features)],
				num_features * sizeof(ST));
	}

	return submatrix;
}

template<class ST> SGMatrix<ST> CDenseFeatures<ST>::steal_feature_matrix()
{
	SGMatrix<ST> st_feature_matrix=feature_matrix;
	m_subset_stack->remove_all_subsets();
	SG_UNREF(feature_cache);
	clean_preprocessors();
	free_feature_matrix();
	return st_feature_matrix;
}

template<class ST> void CDenseFeatures<ST>::set_feature_matrix(SGMatrix<ST> matrix)
{
	m_subset_stack->remove_all_subsets();
	free_feature_matrix();
	feature_matrix = matrix;
	num_features = matrix.num_rows;
	num_vectors = matrix.num_cols;
}

template<class ST> ST* CDenseFeatures<ST>::get_feature_matrix(int32_t &num_feat, int32_t &num_vec)
{
	num_feat = num_features;
	num_vec = num_vectors;
	return feature_matrix.matrix;
}

template<class ST> CDenseFeatures<ST>* CDenseFeatures<ST>::get_transposed()
{
	int32_t num_feat;
	int32_t num_vec;
	ST* fm = get_transposed(num_feat, num_vec);

	return new CDenseFeatures<ST>(fm, num_feat, num_vec);
}

template<class ST> ST* CDenseFeatures<ST>::get_transposed(int32_t &num_feat, int32_t &num_vec)
{
	num_feat = get_num_vectors();
	num_vec = num_features;

	int32_t old_num_vec=get_num_vectors();

	ST* fm = SG_MALLOC(ST, int64_t(num_feat) * num_vec);

	for (int32_t i=0; i<old_num_vec; i++)
	{
		SGVector<ST> vec=get_feature_vector(i);

		for (int32_t j=0; j<vec.vlen; j++)
			fm[j*int64_t(old_num_vec)+i]=vec.vector[j];

		free_feature_vector(vec, i);
	}

	return fm;
}

template<class ST> void CDenseFeatures<ST>::copy_feature_matrix(SGMatrix<ST> src)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call copy_feature_matrix\n")

	free_feature_matrix();
	feature_matrix = src.clone();
	num_features = src.num_rows;
	num_vectors = src.num_cols;
	initialize_cache();
}

template<class ST> void CDenseFeatures<ST>::obtain_from_dot(CDotFeatures* df)
{
	m_subset_stack->remove_all_subsets();

	int32_t num_feat = df->get_dim_feature_space();
	int32_t num_vec = df->get_num_vectors();

	ASSERT(num_feat>0 && num_vec>0)

	free_feature_matrix();
	feature_matrix = SGMatrix<ST>(num_feat, num_vec);

	for (int32_t i = 0; i < num_vec; i++)
	{
		SGVector<float64_t> v = df->get_computed_dot_feature_vector(i);
		ASSERT(num_feat==v.vlen)

		for (int32_t j = 0; j < num_feat; j++)
			feature_matrix.matrix[i * int64_t(num_feat) + j] = (ST) v.vector[j];
	}
	num_features = num_feat;
	num_vectors = num_vec;
}

template<class ST> bool CDenseFeatures<ST>::apply_preprocessor(bool force_preprocessing)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call apply_preproc\n")

	SG_DEBUG("force: %d\n", force_preprocessing)

	if (feature_matrix.matrix && get_num_preprocessors())
	{
		for (int32_t i = 0; i < get_num_preprocessors(); i++)
		{
			if ((!is_preprocessed(i) || force_preprocessing))
			{
				set_preprocessed(i);
				CDensePreprocessor<ST>* p =
						(CDensePreprocessor<ST>*) get_preprocessor(i);
				SG_INFO("preprocessing using preproc %s\n", p->get_name())

				if (p->apply_to_feature_matrix(this).matrix == NULL)
				{
					SG_UNREF(p);
					return false;
				}
				SG_UNREF(p);

			}
		}

		return true;
	}
	else
	{
		if (!feature_matrix.matrix)
			SG_ERROR("no feature matrix\n")

		if (!get_num_preprocessors())
			SG_ERROR("no preprocessors available\n")

		return false;
	}
}

template<class ST> int32_t CDenseFeatures<ST>::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : num_vectors;
}

template<class ST> int32_t CDenseFeatures<ST>::get_num_features() const { return num_features; }

template<class ST> void CDenseFeatures<ST>::set_num_features(int32_t num)
{
	num_features = num;
	initialize_cache();
}

template<class ST> void CDenseFeatures<ST>::set_num_vectors(int32_t num)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call set_num_vectors\n")

	num_vectors = num;
	initialize_cache();
}

template<class ST> void CDenseFeatures<ST>::initialize_cache()
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call initialize_cache\n")

	if (num_features && num_vectors)
	{
		SG_UNREF(feature_cache);
		feature_cache = new CCache<ST>(get_cache_size(), num_features,
				num_vectors);
		SG_REF(feature_cache);
	}
}

template<class ST> EFeatureClass CDenseFeatures<ST>::get_feature_class() const  { return C_DENSE; }

template<class ST> bool CDenseFeatures<ST>::reshape(int32_t p_num_features, int32_t p_num_vectors)
{
	if (m_subset_stack->has_subsets())
		SG_ERROR("A subset is set, cannot call reshape\n")

	if (p_num_features * p_num_vectors
			== this->num_features * this->num_vectors)
	{
		num_features = p_num_features;
		num_vectors = p_num_vectors;
		return true;
	} else
		return false;
}

template<class ST> int32_t CDenseFeatures<ST>::get_dim_feature_space() const { return num_features; }

template<class ST> float64_t CDenseFeatures<ST>::dot(int32_t vec_idx1, CDotFeatures* df,
		int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CDenseFeatures<ST>* sf = (CDenseFeatures<ST>*) df;

	int32_t len1, len2;
	bool free1, free2;

	ST* vec1 = get_feature_vector(vec_idx1, len1, free1);
	ST* vec2 = sf->get_feature_vector(vec_idx2, len2, free2);

	float64_t result = SGVector<ST>::dot(vec1, vec2, len1);

	free_feature_vector(vec1, vec_idx1, free1);
	sf->free_feature_vector(vec2, vec_idx2, free2);

	return result;
}

template<class ST> void CDenseFeatures<ST>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
		float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	ST* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)

	if (abs_val)
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * CMath::abs(vec1[i]);
	}
	else
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * vec1[i];
	}

	free_feature_vector(vec1, vec_idx1, vfree);
}

template<>
void CDenseFeatures<float64_t>::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
		float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	float64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)

	if (abs_val)
	{
		for (int32_t i = 0; i < num_features; i++)
			vec2[i] += alpha * CMath::abs(vec1[i]);
	}
	else
	{
		SGVector<float64_t>::vec1_plus_scalar_times_vec2(vec2, alpha, vec1, num_features);
	}

	free_feature_vector(vec1, vec_idx1, vfree);
}

template<class ST> int32_t CDenseFeatures<ST>::get_nnz_features_for_vector(int32_t num)
{
	return num_features;
}

template<class ST> void* CDenseFeatures<ST>::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=get_num_vectors())
	{
		SG_ERROR("Index out of bounds (number of vectors %d, you "
		"requested %d)\n", get_num_vectors(), vector_index);
	}

	dense_feature_iterator* iterator = SG_MALLOC(dense_feature_iterator, 1);
	iterator->vec = get_feature_vector(vector_index, iterator->vlen,
			iterator->vfree);
	iterator->vidx = vector_index;
	iterator->index = 0;
	return iterator;
}

template<class ST> bool CDenseFeatures<ST>::get_next_feature(int32_t& index, float64_t& value,
		void* iterator)
{
	dense_feature_iterator* it = (dense_feature_iterator*) iterator;
	if (!it || it->index >= it->vlen)
		return false;

	index = it->index++;
	value = (float64_t) it->vec[index];

	return true;
}

template<class ST> void CDenseFeatures<ST>::free_feature_iterator(void* iterator)
{
	if (!iterator)
		return;

	dense_feature_iterator* it = (dense_feature_iterator*) iterator;
	free_feature_vector(it->vec, it->vidx, it->vfree);
	SG_FREE(it);
}

template<class ST> CFeatures* CDenseFeatures<ST>::copy_subset(SGVector<index_t> indices)
{
	SGMatrix<ST> feature_matrix_copy(num_features, indices.vlen);

	for (index_t i=0; i<indices.vlen; ++i)
	{
		index_t real_idx=m_subset_stack->subset_idx_conversion(indices.vector[i]);
		memcpy(&feature_matrix_copy.matrix[i*num_features],
				&feature_matrix.matrix[real_idx*num_features],
				num_features*sizeof(ST));
	}

	CFeatures* result=new CDenseFeatures(feature_matrix_copy);
	SG_REF(result);
	return result;
}

template<class ST> ST* CDenseFeatures<ST>::compute_feature_vector(int32_t num, int32_t& len,
		ST* target)
{
	SG_NOTIMPLEMENTED
	len = 0;
	return NULL;
}

template<class ST> void CDenseFeatures<ST>::init()
{
	num_vectors = 0;
	num_features = 0;

	feature_matrix = SGMatrix<ST>();
	feature_cache = NULL;

	set_generic<ST>();

	/* not store number of vectors in subset */
	SG_ADD(&num_vectors, "num_vectors", "Number of vectors.", MS_NOT_AVAILABLE);
	SG_ADD(&num_features, "num_features", "Number of features.", MS_NOT_AVAILABLE);
	SG_ADD(&feature_matrix, "feature_matrix",
			"Matrix of feature vectors / 1 vector per column.", MS_NOT_AVAILABLE);
}

#define GET_FEATURE_TYPE(f_type, sg_type)	\
template<> EFeatureType CDenseFeatures<sg_type>::get_feature_type() const \
{																			\
	return f_type;															\
}

GET_FEATURE_TYPE(F_BOOL, bool)
GET_FEATURE_TYPE(F_CHAR, char)
GET_FEATURE_TYPE(F_BYTE, uint8_t)
GET_FEATURE_TYPE(F_BYTE, int8_t)
GET_FEATURE_TYPE(F_SHORT, int16_t)
GET_FEATURE_TYPE(F_WORD, uint16_t)
GET_FEATURE_TYPE(F_INT, int32_t)
GET_FEATURE_TYPE(F_UINT, uint32_t)
GET_FEATURE_TYPE(F_LONG, int64_t)
GET_FEATURE_TYPE(F_ULONG, uint64_t)
GET_FEATURE_TYPE(F_SHORTREAL, float32_t)
GET_FEATURE_TYPE(F_DREAL, float64_t)
GET_FEATURE_TYPE(F_LONGREAL, floatmax_t)
#undef GET_FEATURE_TYPE

template<> float64_t CDenseFeatures<bool>::dense_dot(int32_t vec_idx1,
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	bool* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] ? vec2[i] : 0;

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<char>::dense_dot(int32_t vec_idx1,
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	char* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<int8_t>::dense_dot(int32_t vec_idx1,
		const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	int8_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<uint8_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	uint8_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<int16_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	int16_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<uint16_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	uint16_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<int32_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	int32_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<uint32_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	uint32_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<int64_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	int64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<uint64_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	uint64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<float32_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	float32_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<float64_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	float64_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = SGVector<float64_t>::dot(vec1, vec2, num_features);

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<> float64_t CDenseFeatures<floatmax_t>::dense_dot(
		int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == num_features)

	int32_t vlen;
	bool vfree;
	floatmax_t* vec1 = get_feature_vector(vec_idx1, vlen, vfree);

	ASSERT(vlen == num_features)
	float64_t result = 0;

	for (int32_t i = 0; i < num_features; i++)
		result += vec1[i] * vec2[i];

	free_feature_vector(vec1, vec_idx1, vfree);

	return result;
}

template<class ST> bool CDenseFeatures<ST>::is_equal(CDenseFeatures* rhs)
{
	if ( num_features != rhs->num_features || num_vectors != rhs->num_vectors )
		return false;

	ST* vec1;
	ST* vec2;
	int32_t v1len, v2len;
	bool v1free, v2free, stop = false;

	for (int32_t i = 0; i < num_vectors; i++)
	{
		vec1 = get_feature_vector(i, v1len, v1free);
		vec2 = rhs->get_feature_vector(i, v2len, v2free);

		if (v1len!=v2len)
			stop = true;

		for (int32_t j=0; j<v1len; j++)
		{
			if (vec1[j]!=vec2[j])
				stop = true;
		}

		free_feature_vector(vec1, i, v1free);
		free_feature_vector(vec2, i, v2free);

		if (stop)
			return false;
	}

	return true;
}

template<class ST> CFeatures* CDenseFeatures<ST>::create_merged_copy(
		CList* others)
{
	SG_DEBUG("entering %s::create_merged_copy()\n", get_name());

	if (!others)
		return NULL;

	/* first, check other features and count number of elements */
	CSGObject* other=others->get_first_element();
	index_t num_vectors_merged=num_vectors;
	while (other)
	{
		CDenseFeatures<ST>* casted=dynamic_cast<CDenseFeatures<ST>* >(other);

		if (!casted)
		{
			SG_ERROR("%s::create_merged_copy(): Could not cast object of %s to "
					"same type as %s\n",get_name(), other->get_name(), get_name());
		}

		if (get_feature_type()!=casted->get_feature_type() ||
				get_feature_class()!=casted->get_feature_class() ||
				strcmp(get_name(), casted->get_name()))
		{
			SG_ERROR("%s::create_merged_copy(): Features are of different type!\n",
					get_name());
		}

		if (num_features!=casted->num_features)
		{
			SG_ERROR("%s::create_merged_copy(): Provided feature object has "
					"different dimension than this one\n");
		}

		num_vectors_merged+=casted->get_num_vectors();

		/* check if reference counting is used */
		if (others->get_delete_data())
			SG_UNREF(other);
		other=others->get_next_element();
	}

	/* create new feature matrix and copy both instances data into it */
	SGMatrix<ST> data(num_features, num_vectors_merged);

	/* copy data of this instance */
	SG_DEBUG("copying matrix of this instance\n")
	memcpy(data.matrix, feature_matrix.matrix,
			num_features*num_vectors*sizeof(ST));

	/* count number of vectors (not elements) processed so far */
	index_t num_processed=num_vectors;

	/* now copy data of other features block wise */
	other=others->get_first_element();
	while (other)
	{
		/* cast is safe due to above check */
		CDenseFeatures<ST>* casted=(CDenseFeatures<ST>*)other;

		SG_DEBUG("copying matrix of provided instance\n")
		memcpy(&(data.matrix[num_processed*num_features]),
				casted->get_feature_matrix().matrix,
				num_features*casted->get_num_vectors()*sizeof(ST));

		/* update counting */
		num_processed+=casted->get_num_vectors();

		/* check if reference counting is used */
		if (others->get_delete_data())
			SG_UNREF(other);
		other=others->get_next_element();
	}

	/* create new instance and return */
	CDenseFeatures<ST>* result=new CDenseFeatures<ST>(data);

	SG_DEBUG("leaving %s::create_merged_copy()\n", get_name());
	return result;
}

template<class ST> CFeatures* CDenseFeatures<ST>::create_merged_copy(
		CFeatures* other)
{
	SG_DEBUG("entering %s::create_merged_copy()\n", get_name());

	/* create list with one element and call general method */
	CList* list=new CList();
	list->append_element(other);
	CFeatures* result=create_merged_copy(list);
	SG_UNREF(list);

	SG_DEBUG("leaving %s::create_merged_copy()\n", get_name());
	return result;
}

template<class ST>
void CDenseFeatures<ST>::load(CFile* loader)
{
	SGMatrix<ST> matrix;
	matrix.load(loader);
	set_feature_matrix(matrix);
}

template<class ST>
void CDenseFeatures<ST>::save(CFile* writer)
{
	feature_matrix.save(writer);
}

template< class ST > CDenseFeatures< ST >* CDenseFeatures< ST >::obtain_from_generic(CFeatures* const base_features)
{
	REQUIRE(base_features->get_feature_class() == C_DENSE,
			"base_features must be of dynamic type CDenseFeatures\n")

	return (CDenseFeatures< ST >*) base_features;
}

template class CDenseFeatures<bool>;
template class CDenseFeatures<char>;
template class CDenseFeatures<int8_t>;
template class CDenseFeatures<uint8_t>;
template class CDenseFeatures<int16_t>;
template class CDenseFeatures<uint16_t>;
template class CDenseFeatures<int32_t>;
template class CDenseFeatures<uint32_t>;
template class CDenseFeatures<int64_t>;
template class CDenseFeatures<uint64_t>;
template class CDenseFeatures<float32_t>;
template class CDenseFeatures<float64_t>;
template class CDenseFeatures<floatmax_t>;
}
