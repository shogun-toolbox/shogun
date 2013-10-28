/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/streaming/StreamingHashedDenseFeatures.h>
#include <shogun/io/streaming/StreamingFileFromDenseFeatures.h>
#include <shogun/features/HashedDenseFeatures.h>

namespace shogun
{
template <class ST>
CStreamingHashedDenseFeatures<ST>::CStreamingHashedDenseFeatures()
{
	init(NULL, false, 0, 0, false, true);
}

template <class ST>
CStreamingHashedDenseFeatures<ST>::CStreamingHashedDenseFeatures(CStreamingFile* file,
	bool is_labelled, int32_t size, int32_t d, bool use_quadr, bool keep_lin_terms)
{
	init(file, is_labelled, size, d, use_quadr, keep_lin_terms);
}

template <class ST>
CStreamingHashedDenseFeatures<ST>::CStreamingHashedDenseFeatures(CDenseFeatures<ST>* dot_features,
	int32_t d, bool use_quadr, bool keep_lin_terms, float64_t* lab)
{
	ASSERT(dot_features);

	CStreamingFileFromDenseFeatures<ST>* file =
			new CStreamingFileFromDenseFeatures<ST>(dot_features, lab);
	bool is_labelled = (lab != NULL);
	int32_t size = 1024;

	init(file, is_labelled, size, d, use_quadr, keep_lin_terms);

	parser.set_free_vectors_on_destruct(false);
	seekable=true;
}

template <class ST>
CStreamingHashedDenseFeatures<ST>::~CStreamingHashedDenseFeatures()
{
}

template <class ST>
void CStreamingHashedDenseFeatures<ST>::init(CStreamingFile* file, bool is_labelled,
	int32_t size, int32_t d, bool use_quadr, bool keep_lin_terms)
{
	dim = d;
	use_quadratic = use_quadr;
	keep_linear_terms = keep_lin_terms;

	SG_ADD(&use_quadratic, "use_quadratic", "Whether to use quadratic features",
		MS_NOT_AVAILABLE);
	SG_ADD(&keep_linear_terms, "keep_linear_terms", "Whether to keep the linear terms or not",
		MS_NOT_AVAILABLE);
	SG_ADD(&dim, "dim", "Size of target dimension", MS_NOT_AVAILABLE);

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
float32_t CStreamingHashedDenseFeatures<ST>::dot(CStreamingDotFeatures* df)
{
	ASSERT(df);
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(strcmp(df->get_name(),get_name())==0)

	CStreamingHashedDenseFeatures<ST>* hdf = (CStreamingHashedDenseFeatures<ST>* ) df;
	return current_vector.sparse_dot(hdf->current_vector);
}

template <class ST>
float32_t CStreamingHashedDenseFeatures<ST>::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == dim);

	float32_t result = 0;
	for (index_t i=0; i<current_vector.num_feat_entries; i++)
		result += vec2[current_vector.features[i].feat_index] * current_vector.features[i].entry;

	return result;
}

template <class ST>
void CStreamingHashedDenseFeatures<ST>::add_to_dense_vec(float32_t alpha, float32_t* vec2,
	int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == dim);

	if (abs_val)
		alpha = CMath::abs(alpha);

	for (index_t i=0; i<current_vector.num_feat_entries; i++)
		vec2[current_vector.features[i].feat_index] += alpha * current_vector.features[i].entry;
}

template <class ST>
int32_t CStreamingHashedDenseFeatures<ST>::get_dim_feature_space() const
{
	return dim;
}

template <class ST>
const char* CStreamingHashedDenseFeatures<ST>::get_name() const
{
	return "StreamingHashedDenseFeatures";
}

template <class ST>
int32_t CStreamingHashedDenseFeatures<ST>::get_num_vectors() const
{
	return 1;
}

template <class ST>
CFeatures* CStreamingHashedDenseFeatures<ST>::duplicate() const
{
	return new CStreamingHashedDenseFeatures<ST>(*this);
}

template <class ST>
void CStreamingHashedDenseFeatures<ST>::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_vector);
}

template <class ST>
void CStreamingHashedDenseFeatures<ST>::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&CStreamingFile::get_vector_and_label);
}

template <class ST>
EFeatureType CStreamingHashedDenseFeatures<ST>::get_feature_type() const
{
	return F_UINT;
}

template <class ST>
EFeatureClass CStreamingHashedDenseFeatures<ST>::get_feature_class() const
{
	return C_STREAMING_SPARSE;
}

template <class ST>
void CStreamingHashedDenseFeatures<ST>::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

template <class ST>
void CStreamingHashedDenseFeatures<ST>::end_parser()
{
	parser.end_parser();
}

template <class ST>
float64_t CStreamingHashedDenseFeatures<ST>::get_label()
{
	return current_label;
}

template <class ST>
bool CStreamingHashedDenseFeatures<ST>::get_next_example()
{
	SGVector<ST> tmp;
	if (parser.get_next_example(tmp.vector,
		tmp.vlen, current_label))
	{
		current_vector = CHashedDenseFeatures<ST>::hash_vector(tmp, dim, use_quadratic,
				keep_linear_terms);
		tmp.vector = NULL;
		tmp.vlen = -1;
		return true;
	}
	return false;
}

template <class ST>
void CStreamingHashedDenseFeatures<ST>::release_example()
{
	parser.finalize_example();
}

template <class ST>
int32_t CStreamingHashedDenseFeatures<ST>::get_num_features()
{
	return dim;
}

template <class ST>
SGSparseVector<ST> CStreamingHashedDenseFeatures<ST>::get_vector()
{
	return current_vector;
}

template class CStreamingHashedDenseFeatures<bool>;
template class CStreamingHashedDenseFeatures<char>;
template class CStreamingHashedDenseFeatures<int8_t>;
template class CStreamingHashedDenseFeatures<uint8_t>;
template class CStreamingHashedDenseFeatures<int16_t>;
template class CStreamingHashedDenseFeatures<uint16_t>;
template class CStreamingHashedDenseFeatures<int32_t>;
template class CStreamingHashedDenseFeatures<uint32_t>;
template class CStreamingHashedDenseFeatures<int64_t>;
template class CStreamingHashedDenseFeatures<uint64_t>;
template class CStreamingHashedDenseFeatures<float32_t>;
template class CStreamingHashedDenseFeatures<float64_t>;
template class CStreamingHashedDenseFeatures<floatmax_t>;
}
