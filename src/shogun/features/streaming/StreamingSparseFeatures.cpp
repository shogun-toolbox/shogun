/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Modifications (W) 2013 Thoralf Klein
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include <features/streaming/StreamingSparseFeatures.h>
namespace shogun
{

template <class T>
CStreamingSparseFeatures<T>::CStreamingSparseFeatures() : CStreamingDotFeatures()
{
	set_read_functions();
	init();
}

template <class T>
CStreamingSparseFeatures<T>::CStreamingSparseFeatures(CStreamingFile* file,
			 bool is_labelled,
			 int32_t size)
	: CStreamingDotFeatures()
{
	set_read_functions();
	init(file, is_labelled, size);
}

template <class T>
CStreamingSparseFeatures<T>::~CStreamingSparseFeatures()
{
	if (parser.is_running())
		parser.end_parser();
}

template <class T>
T CStreamingSparseFeatures<T>::get_feature(int32_t index)
{
	ASSERT(index>=0 && index<current_num_features)
	return current_sgvector.get_feature(index);
}

template <class T>
void CStreamingSparseFeatures<T>::reset_stream()
{
	SG_NOTIMPLEMENTED
}

template <class T>
int32_t CStreamingSparseFeatures<T>::set_num_features(int32_t num)
{
	int32_t n=current_num_features;
	ASSERT(n<=num)
	current_num_features=num;
	return n;
}

template <class T>
T CStreamingSparseFeatures<T>::sparse_dot(T alpha, SGSparseVectorEntry<T>* avec, int32_t alen, SGSparseVectorEntry<T>* bvec, int32_t blen)
{
	T result=0;

	//result remains zero when one of the vectors is non existent
	if (avec && bvec)
	{
		SGSparseVector<T> asv(avec, alen, false);
		SGSparseVector<T> bsv(bvec, blen, false);

		result=alpha*SGSparseVector<T>::sparse_dot(asv, bsv);
	}

	return result;
}

template <class T>
T CStreamingSparseFeatures<T>::dense_dot(T alpha, T* vec, int32_t dim, T b)
{
	ASSERT(vec)
	ASSERT(dim>=current_num_features)

	return current_sgvector.dense_dot(alpha, vec, dim, b);
}

template <class T>
float64_t CStreamingSparseFeatures<T>::dense_dot(const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2)

	int32_t current_length = current_sgvector.num_feat_entries;
	SGSparseVectorEntry<T>* current_vector = current_sgvector.features;

	float64_t result=0;
	if (current_vector)
	{
		for (int32_t i=0; i<current_length; i++) {
			if (current_vector[i].feat_index < vec2_len) {
				result+=vec2[current_vector[i].feat_index]*current_vector[i].entry;
			}
		}
	}

	return result;
}

template <class T>
float32_t CStreamingSparseFeatures<T>::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2)

	int32_t current_length = current_sgvector.num_feat_entries;
	SGSparseVectorEntry<T>* current_vector = current_sgvector.features;

	float32_t result=0;
	if (current_vector)
	{
		for (int32_t i=0; i<current_length; i++) {
			if (current_vector[i].feat_index < vec2_len) {
				result+=vec2[current_vector[i].feat_index]*current_vector[i].entry;
			}
		}
	}

	return result;
}

template <class T>
void CStreamingSparseFeatures<T>::add_to_dense_vec(float64_t alpha, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2)
	if (vec2_len < current_num_features)
	{
		SG_ERROR("dimension of vec (=%d) does not match number of features (=%d)\n",
			 vec2_len, current_num_features);
	}

	SGSparseVectorEntry<T>* sv=current_sgvector.features;
	int32_t num_feat=current_sgvector.num_feat_entries;

	if (sv)
	{
		if (abs_val)
		{
			for (int32_t i=0; i<num_feat; i++)
				vec2[sv[i].feat_index]+= alpha*CMath::abs(sv[i].entry);
		}
		else
		{
			for (int32_t i=0; i<num_feat; i++)
				vec2[sv[i].feat_index]+= alpha*sv[i].entry;
		}
	}
}

template <class T>
void CStreamingSparseFeatures<T>::add_to_dense_vec(float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2)
	if (vec2_len < current_num_features)
	{
		SG_ERROR("dimension of vec (=%d) does not match number of features (=%d)\n",
			 vec2_len, current_num_features);
	}

	SGSparseVectorEntry<T>* sv=current_sgvector.features;
	int32_t num_feat=current_sgvector.num_feat_entries;

	if (sv)
	{
		if (abs_val)
		{
			for (int32_t i=0; i<num_feat; i++)
				vec2[sv[i].feat_index]+= alpha*CMath::abs(sv[i].entry);
		}
		else
		{
			for (int32_t i=0; i<num_feat; i++)
				vec2[sv[i].feat_index]+= alpha*sv[i].entry;
		}
	}
}

template <class T>
int64_t CStreamingSparseFeatures<T>::get_num_nonzero_entries()
{
	return current_sgvector.num_feat_entries;
}

template <class T>
float32_t CStreamingSparseFeatures<T>::compute_squared()
{
	int32_t current_length = current_sgvector.num_feat_entries;
	SGSparseVectorEntry<T>* current_vector = current_sgvector.features;

	ASSERT(current_vector)

	float32_t sq=0;

	for (int32_t i=0; i<current_length; i++)
		sq += current_vector[i].entry * current_vector[i].entry;

	return sq;
}

template <class T>
void CStreamingSparseFeatures<T>::sort_features()
{
	SGSparseVectorEntry<T>* old_ptr = current_sgvector.features;

	// setting false to disallow reallocation
	// and guarantee stable get_vector().features pointer
	get_vector().sort_features(true);

	ASSERT(old_ptr == current_sgvector.features);
}

template <class T>
CFeatures* CStreamingSparseFeatures<T>::duplicate() const
{
	return new CStreamingSparseFeatures<T>(*this);
}

template <class T>
int32_t CStreamingSparseFeatures<T>::get_num_vectors() const
{
	if (current_sgvector.features)
		return 1;
	return 0;
}

template <class T> void CStreamingSparseFeatures<T>::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_sparse_vector);
}

template <class T> void CStreamingSparseFeatures<T>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label
		(&CStreamingFile::get_sparse_vector_and_label);
}

#define GET_FEATURE_TYPE(f_type, sg_type)				\
template<> EFeatureType CStreamingSparseFeatures<sg_type>::get_feature_type() const \
{									\
	return f_type;							\
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


template <class T>
void CStreamingSparseFeatures<T>::init()
{
	working_file=NULL;
	current_vec_index=0;
	current_num_features=-1;

	set_generic<T>();
}

template <class T>
void CStreamingSparseFeatures<T>::init(CStreamingFile* file,
				    bool is_labelled,
				    int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	SG_REF(working_file);
	parser.init(file, is_labelled, size);
	parser.set_free_vector_after_release(false);
}

template <class T>
void CStreamingSparseFeatures<T>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template <class T>
void CStreamingSparseFeatures<T>::end_parser()
{
	parser.end_parser();
}

template <class T>
bool CStreamingSparseFeatures<T>::get_next_example()
{
	int32_t current_length = 0;
	SGSparseVectorEntry<T>* current_vector = NULL;

	bool ret_value;
	ret_value = (bool) parser.get_next_example(current_vector,
						   current_length,
						   current_label);

	if (!ret_value)
		return false;

	// ref_count disabled, because parser still owns the memory
	current_sgvector = SGSparseVector<T>(current_vector, current_length, false);

	// Update number of features based on highest index
	int32_t current_dimension = get_vector().get_num_dimensions();
	current_num_features = CMath::max(current_num_features, current_dimension);

	current_vec_index++;
	return true;
}

template <class T>
SGSparseVector<T> CStreamingSparseFeatures<T>::get_vector()
{
	return current_sgvector;
}

template <class T>
float64_t CStreamingSparseFeatures<T>::get_label()
{
	ASSERT(has_labels)

	return current_label;
}

template <class T>
void CStreamingSparseFeatures<T>::release_example()
{
	parser.finalize_example();
}

template <class T>
int32_t CStreamingSparseFeatures<T>::get_dim_feature_space() const
{
	return current_num_features;
}

template <class T>
	float32_t CStreamingSparseFeatures<T>::dot(CStreamingDotFeatures* df)
{
	SG_NOTIMPLEMENTED
	return -1;
}

template <class T>
int32_t CStreamingSparseFeatures<T>::get_num_features()
{
	return current_num_features;
}

template <class T>
int32_t CStreamingSparseFeatures<T>::get_nnz_features_for_vector()
{
	return current_sgvector.num_feat_entries;
}

template <class T>
EFeatureClass CStreamingSparseFeatures<T>::get_feature_class() const
{
	return C_STREAMING_SPARSE;
}

template class CStreamingSparseFeatures<bool>;
template class CStreamingSparseFeatures<char>;
template class CStreamingSparseFeatures<int8_t>;
template class CStreamingSparseFeatures<uint8_t>;
template class CStreamingSparseFeatures<int16_t>;
template class CStreamingSparseFeatures<uint16_t>;
template class CStreamingSparseFeatures<int32_t>;
template class CStreamingSparseFeatures<uint32_t>;
template class CStreamingSparseFeatures<int64_t>;
template class CStreamingSparseFeatures<uint64_t>;
template class CStreamingSparseFeatures<float32_t>;
template class CStreamingSparseFeatures<float64_t>;
template class CStreamingSparseFeatures<floatmax_t>;
}
