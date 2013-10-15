/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/streaming/StreamingHashedSparseFeatures.h>
#include <shogun/features/HashedSparseFeatures.h>
#include <shogun/io/streaming/StreamingFileFromSparseFeatures.h>

namespace shogun
{

template <class ST>
CStreamingHashedSparseFeatures<ST>::CStreamingHashedSparseFeatures()
{
	init(NULL, false, 0, 0, false, true);
}

template <class ST>
CStreamingHashedSparseFeatures<ST>::CStreamingHashedSparseFeatures(CStreamingFile* file,
	bool is_labelled, int32_t size, int32_t d, bool use_quadr, bool keep_lin_terms)
{
	init(file, is_labelled, size, d, use_quadr, keep_lin_terms);
}

template <class ST>
CStreamingHashedSparseFeatures<ST>::CStreamingHashedSparseFeatures(CSparseFeatures<ST>* dot_features,
	int32_t d, bool use_quadr, bool keep_lin_terms, float64_t* lab)
{
	ASSERT(dot_features);

	CStreamingFileFromSparseFeatures<ST>* file =
			new CStreamingFileFromSparseFeatures<ST>(dot_features, lab);
	bool is_labelled = (lab != NULL);
	int32_t size = 1024;

	init(file, is_labelled, size, d, use_quadr, keep_lin_terms);

	parser.set_free_vectors_on_destruct(false);
	seekable=true;
}

template <class ST>
CStreamingHashedSparseFeatures<ST>::~CStreamingHashedSparseFeatures()
{
}

template <class ST>
void CStreamingHashedSparseFeatures<ST>::init(CStreamingFile* file, bool is_labelled,
	int32_t size, int32_t d, bool use_quadr, bool keep_lin_terms)
{
	dim = d;
	SG_ADD(&dim, "dim", "Size of target dimension", MS_NOT_AVAILABLE);

	use_quadratic = use_quadr;
	keep_linear_terms = keep_lin_terms;

	SG_ADD(&use_quadratic, "use_quadratic", "Whether to use quadratic features",
		MS_NOT_AVAILABLE);
	SG_ADD(&keep_linear_terms, "keep_linear_terms", "Whether to keep the linear terms or not",
		MS_NOT_AVAILABLE);

	has_labels = is_labelled;
	if (file)
	{
		working_file = file;
		SG_REF(working_file);
		parser.init(file, is_labelled, size);
		seekable = false;
	}
	else
		file = NULL;

	set_read_functions();
	parser.set_free_vector_after_release(false);

	set_generic<ST>();
}

template <class ST>
float32_t CStreamingHashedSparseFeatures<ST>::dot(CStreamingDotFeatures* df)
{
	ASSERT(df);
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(strcmp(df->get_name(),get_name())==0)

	CStreamingHashedSparseFeatures<ST>* hdf = (CStreamingHashedSparseFeatures<ST>* ) df;
	return current_vector.sparse_dot(hdf->current_vector);
}

template <class ST>
float32_t CStreamingHashedSparseFeatures<ST>::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == dim);

	float32_t result = 0;
	for (index_t i=0; i<current_vector.num_feat_entries; i++)
		result += vec2[current_vector.features[i].feat_index] * current_vector.features[i].entry;

	return result;
}

template <class ST>
void CStreamingHashedSparseFeatures<ST>::add_to_dense_vec(float32_t alpha, float32_t* vec2,
	int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == dim);

	if (abs_val)
		alpha = CMath::abs(alpha);

	for (index_t i=0; i<current_vector.num_feat_entries; i++)
		vec2[current_vector.features[i].feat_index] += alpha * current_vector.features[i].entry;
}

template <class ST>
int32_t CStreamingHashedSparseFeatures<ST>::get_dim_feature_space() const
{
	return dim;
}

template <class ST>
const char* CStreamingHashedSparseFeatures<ST>::get_name() const
{
	return "StreamingHashedSparseFeatures";
}

template <class ST>
int32_t CStreamingHashedSparseFeatures<ST>::get_num_vectors() const
{
	return 1;
}

template <class ST>
CFeatures* CStreamingHashedSparseFeatures<ST>::duplicate() const
{
	return new CStreamingHashedSparseFeatures<ST>(*this);
}

template <class ST>
void CStreamingHashedSparseFeatures<ST>::set_vector_reader()
{
	SG_DEBUG("called inside set_vector_reader\n");
	parser.set_read_vector(&CStreamingFile::get_sparse_vector);
}

template <class ST>
void CStreamingHashedSparseFeatures<ST>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&CStreamingFile::get_sparse_vector_and_label);
}

template <class ST>
EFeatureType CStreamingHashedSparseFeatures<ST>::get_feature_type() const
{
	return F_UINT;
}

template <class ST>
EFeatureClass CStreamingHashedSparseFeatures<ST>::get_feature_class() const
{
	return C_STREAMING_SPARSE;
}

template <class ST>
void CStreamingHashedSparseFeatures<ST>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template <class ST>
void CStreamingHashedSparseFeatures<ST>::end_parser()
{
	parser.end_parser();
}

template <class ST>
float64_t CStreamingHashedSparseFeatures<ST>::get_label()
{
	return current_label;
}

template <class ST>
bool CStreamingHashedSparseFeatures<ST>::get_next_example()
{
	SGSparseVector<ST> tmp;
	if (parser.get_next_example(tmp.features,
		tmp.num_feat_entries, current_label))
	{
		current_vector = CHashedSparseFeatures<ST>::hash_vector(tmp, dim,
				use_quadratic, keep_linear_terms);
		tmp.features = NULL;
		tmp.num_feat_entries = -1;
		return true;
	}
	return false;
}

template <class ST>
void CStreamingHashedSparseFeatures<ST>::release_example()
{
	parser.finalize_example();
}

template <class ST>
int32_t CStreamingHashedSparseFeatures<ST>::get_num_features()
{
	return dim;
}

template <class ST>
SGSparseVector<ST> CStreamingHashedSparseFeatures<ST>::get_vector()
{
	return current_vector;
}

template class CStreamingHashedSparseFeatures<bool>;
template class CStreamingHashedSparseFeatures<char>;
template class CStreamingHashedSparseFeatures<int8_t>;
template class CStreamingHashedSparseFeatures<uint8_t>;
template class CStreamingHashedSparseFeatures<int16_t>;
template class CStreamingHashedSparseFeatures<uint16_t>;
template class CStreamingHashedSparseFeatures<int32_t>;
template class CStreamingHashedSparseFeatures<uint32_t>;
template class CStreamingHashedSparseFeatures<int64_t>;
template class CStreamingHashedSparseFeatures<uint64_t>;
template class CStreamingHashedSparseFeatures<float32_t>;
template class CStreamingHashedSparseFeatures<float64_t>;
template class CStreamingHashedSparseFeatures<floatmax_t>;
}
