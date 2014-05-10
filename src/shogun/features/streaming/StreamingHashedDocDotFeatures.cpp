/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/streaming/StreamingHashedDocDotFeatures.h>
#include <shogun/features/HashedDocDotFeatures.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CStreamingHashedDocDotFeatures::CStreamingHashedDocDotFeatures(CStreamingFile* file,
	bool is_labelled, int32_t size,	CTokenizer* tzer, int32_t bits)
: CStreamingDotFeatures()
{
	init(file, is_labelled, size, tzer, bits, true, 1, 0);
}

CStreamingHashedDocDotFeatures::CStreamingHashedDocDotFeatures() : CStreamingDotFeatures()
{
	init(NULL, false, 0, NULL, 0, false, 1, 0);
}

CStreamingHashedDocDotFeatures::CStreamingHashedDocDotFeatures(
	CStringFeatures<char>* dot_features, CTokenizer* tzer, int32_t bits, float64_t* lab)
: CStreamingDotFeatures()
{
	CStreamingFileFromStringFeatures<char>* file =
		new CStreamingFileFromStringFeatures<char>(dot_features, lab);
	bool is_labelled = (lab != NULL);
	int32_t size=1024;

	init(file, is_labelled, size, tzer, bits, true, 1, 0);

	parser.set_free_vectors_on_destruct(false);
	seekable= true;
}
void CStreamingHashedDocDotFeatures::init(CStreamingFile* file, bool is_labelled,
	int32_t size, CTokenizer* tzer, int32_t bits, bool normalize, int32_t n_grams, int32_t skips)
{
	num_bits = bits;
	tokenizer = tzer;
	if (tokenizer)
	{
		SG_REF(tokenizer);
		converter = new CHashedDocConverter(tzer, bits, normalize, n_grams, skips);
	}
	else
		converter=NULL;

	SG_ADD(&num_bits, "num_bits", "Number of bits for hash", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject** ) &tokenizer, "tokenizer", "The tokenizer used on the documents",
		MS_NOT_AVAILABLE);
	SG_ADD((CSGObject** ) &converter, "converter", "Converter", MS_NOT_AVAILABLE);

	has_labels = is_labelled;
	if (file)
	{
		working_file = file;
		SG_REF(working_file);
		parser.init(file, is_labelled, size);
		seekable = false;
	}
	else
		working_file = NULL;

	set_read_functions();
	parser.set_free_vector_after_release(false);
}

CStreamingHashedDocDotFeatures::~CStreamingHashedDocDotFeatures()
{
	if (parser.is_running())
		parser.end_parser();
	SG_UNREF(working_file);
	SG_UNREF(tokenizer);
	SG_UNREF(converter);
}

float32_t CStreamingHashedDocDotFeatures::dot(CStreamingDotFeatures* df)
{
	ASSERT(df)
	ASSERT(df->get_name() == get_name())

	CStreamingHashedDocDotFeatures* cdf = (CStreamingHashedDocDotFeatures* ) df;
	float32_t result = current_vector.sparse_dot(cdf->current_vector);
	return result;
}

float32_t CStreamingHashedDocDotFeatures::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == CMath::pow(2, num_bits))

	float32_t result = 0;
	for (index_t i=0; i<current_vector.num_feat_entries; i++)
	{
		result += vec2[current_vector.features[i].feat_index] *
					current_vector.features[i].entry;
	}
	return result;
}

void CStreamingHashedDocDotFeatures::add_to_dense_vec(float32_t alpha, float32_t* vec2,
			int32_t vec2_len, bool abs_val)
{
	float32_t value = abs_val ? CMath::abs(alpha) : alpha;

	for (index_t i=0; i<current_vector.num_feat_entries; i++)
		vec2[current_vector.features[i].feat_index] += value * current_vector.features[i].entry;
}

int32_t CStreamingHashedDocDotFeatures::get_dim_feature_space() const
{
	return CMath::pow(2, num_bits);
}

const char* CStreamingHashedDocDotFeatures::get_name() const
{
	return "StreamingHashedDocDotFeatures";
}

CFeatures* CStreamingHashedDocDotFeatures::duplicate() const
{
	SG_NOTIMPLEMENTED
	// return new CStreamingHashedDocDotFeatures(*this);
	return NULL;
}

EFeatureType CStreamingHashedDocDotFeatures::get_feature_type() const
{
	return F_UINT;
}

EFeatureClass CStreamingHashedDocDotFeatures::get_feature_class() const
{
	return C_STREAMING_SPARSE;
}

void CStreamingHashedDocDotFeatures::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

void CStreamingHashedDocDotFeatures::end_parser()
{
	parser.end_parser();
}

bool CStreamingHashedDocDotFeatures::get_next_example()
{
	SGVector<char> tmp;
	if (parser.get_next_example(tmp.vector,
		tmp.vlen, current_label))
	{
		ASSERT(tmp.vector)
		ASSERT(tmp.vlen > 0)
		current_vector = converter->apply(tmp);
		return true;
	}
	return false;
}

void CStreamingHashedDocDotFeatures::release_example()
{
	parser.finalize_example();
}

int32_t CStreamingHashedDocDotFeatures::get_num_features()
{
	return (int32_t) CMath::pow(2, num_bits);
}

float64_t CStreamingHashedDocDotFeatures::get_label()
{
	return current_label;
}

int32_t CStreamingHashedDocDotFeatures::get_num_vectors() const
{
	return 1;
}

void CStreamingHashedDocDotFeatures::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_string);
}

void CStreamingHashedDocDotFeatures::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&CStreamingFile::get_string_and_label);
}

SGSparseVector<float64_t> CStreamingHashedDocDotFeatures::get_vector()
{
	return current_vector;
}

void CStreamingHashedDocDotFeatures::set_normalization(bool normalize)
{
	converter->set_normalization(normalize);
}

void CStreamingHashedDocDotFeatures::set_k_skip_n_grams(int32_t k, int32_t n)
{
	converter->set_k_skip_n_grams(k, n);
}
