/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <shogun/features/streaming/StreamingVwFeatures.h>

using namespace shogun;

CStreamingVwFeatures::CStreamingVwFeatures() : CStreamingDotFeatures()
{
	init();
	set_read_functions();
}

CStreamingVwFeatures::CStreamingVwFeatures(CStreamingVwFile* file,
		bool is_labelled, int32_t size)
: CStreamingDotFeatures()
{
	init(file, is_labelled, size);
	set_read_functions();
}

CStreamingVwFeatures::CStreamingVwFeatures(CStreamingVwCacheFile* file,
		bool is_labelled, int32_t size)
: CStreamingDotFeatures()
{
	init(file, is_labelled, size);
	set_read_functions();
}

CStreamingVwFeatures::~CStreamingVwFeatures()
{
	if (parser.is_running())
		parser.end_parser();
	SG_UNREF(env);
}

CFeatures* CStreamingVwFeatures::duplicate() const
{
	return new CStreamingVwFeatures(*this);
}

void CStreamingVwFeatures::set_vector_reader()
{
	parser.set_read_vector(&CStreamingFile::get_vector);
}

void CStreamingVwFeatures::set_vector_and_label_reader()
{
	parser.set_read_vector_and_label(&CStreamingFile::get_vector_and_label);
}

void CStreamingVwFeatures::reset_stream()
{
	if (working_file->is_seekable())
	{
		working_file->reset_stream();
		parser.exit_parser();
		parser.init(working_file, has_labels, parser.get_ring_size());
		parser.set_free_vector_after_release(false);
		parser.start_parser();
	}
	else
		SG_ERROR("The input cannot be reset! Please use 1 pass.\n")
}

CVwEnvironment* CStreamingVwFeatures::get_env()
{
	SG_REF(env);
	return env;
}

void CStreamingVwFeatures::set_env(CVwEnvironment* vw_env)
{
	env = vw_env;
	SG_REF(env);
}

void CStreamingVwFeatures::expand_if_required(float32_t*& vec, int32_t& len)
{
	int32_t dim = 1 << env->num_bits;
	if (dim > len)
	{
		vec = SG_REALLOC(float32_t, vec, len, dim);
		memset(&vec[len], 0, (dim-len) * sizeof(float32_t));
		len = dim;
	}
}

void CStreamingVwFeatures::expand_if_required(float64_t*& vec, int32_t& len)
{
	int32_t dim = 1 << env->num_bits;
	if (dim > len)
	{
		vec = SG_REALLOC(float64_t, vec, len, dim);
		memset(&vec[len], 0, (dim-len) * sizeof(float64_t));
		len = dim;
	}
}

float32_t CStreamingVwFeatures::real_weight(float32_t w, float32_t gravity)
{
	float32_t wprime = 0;
	if (gravity < fabsf(w))
		wprime = CMath::sign(w)*(fabsf(w) - gravity);
	return wprime;
}

int32_t CStreamingVwFeatures::get_nnz_features_for_vector()
{
	return current_length;
}

int32_t CStreamingVwFeatures::get_num_vectors() const
{
	if (current_example)
		return 1;
	else
		return 0;
}

EFeatureType CStreamingVwFeatures::get_feature_type() const
{
	return F_DREAL;
}

void CStreamingVwFeatures::init()
{
	working_file=NULL;
	seekable=false;
	current_length=-1;
	current_example=NULL;
	env=NULL;

	example_count = 0;
}

void CStreamingVwFeatures::init(CStreamingVwFile* file, bool is_labelled, int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled, size);
	parser.set_free_vector_after_release(false);
	seekable=false;

	// Get environment from the StreamingVwFile
	env = ((CStreamingVwFile*) file)->get_env();
	SG_REF(env);
}

void CStreamingVwFeatures::init(CStreamingVwCacheFile* file, bool is_labelled, int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled, size);
	parser.set_free_vector_after_release(false);
	seekable=true;

	// Get environment from the StreamingVwFile
	env = ((CStreamingVwCacheFile*) file)->get_env();
	SG_REF(env);
}

void CStreamingVwFeatures::setup_example(VwExample* ae)
{
	ae->pass = env->passes_complete;
	ae->num_features = 0;
	ae->total_sum_feat_sq = 1;
	ae->example_counter = ++example_count;
	ae->global_weight = ae->ld->weight;
	env->t += ae->global_weight;
	ae->example_t = env->t;

	// If some namespaces should be ignored, remove them
	if (env->ignore_some)
	{
		for (vw_size_t* i = ae->indices.begin; i != ae->indices.end; i++)
			if (env->ignore[*i])
			{
				ae->atomics[*i].erase();
				memmove(i,i+1,(ae->indices.end - (i+1))*sizeof(vw_size_t));
				ae->indices.end--;
				i--;
			}
	}

	// Add constant feature
	vw_size_t constant_namespace = 128;
	VwFeature temp = {1,constant_hash & env->mask};
	ae->indices.push(constant_namespace);
	ae->atomics[constant_namespace].push(temp);
	ae->sum_feat_sq[constant_namespace] = 0;

	if(env->stride != 1)
	{
		// Make room for per-feature information.
		vw_size_t stride = env->stride;
		for (vw_size_t* i = ae->indices.begin; i != ae->indices.end; i++)
			for(VwFeature* j = ae->atomics[*i].begin; j != ae->atomics[*i].end; j++)
				j->weight_index = j->weight_index*stride;
	}

	for (vw_size_t* i = ae->indices.begin; i != ae->indices.end; i++)
	{
		ae->num_features += ae->atomics[*i].end - ae->atomics[*i].begin;
		ae->total_sum_feat_sq += ae->sum_feat_sq[*i];
	}

	// For quadratic features
	for (int32_t k = 0; k < env->pairs.get_num_elements(); k++)
	{
		char* i = env->pairs.get_element(k);

		ae->num_features
			+= (ae->atomics[(int32_t)(i[0])].end - ae->atomics[(int32_t)(i[0])].begin)
			*(ae->atomics[(int32_t)(i[1])].end - ae->atomics[(int32_t)(i[1])].begin);

		ae->total_sum_feat_sq += ae->sum_feat_sq[(int32_t)(i[0])]*ae->sum_feat_sq[(int32_t)(i[1])];
	}
}

void CStreamingVwFeatures::start_parser()
{
	if (!parser.is_running())
		parser.start_parser();
}

void CStreamingVwFeatures::end_parser()
{
	parser.end_parser();
}

bool CStreamingVwFeatures::get_next_example()
{
	bool ret_value;
	ret_value = (bool) parser.get_next_example(current_example,
						   current_length,
						   current_label);
	if (current_length < 1)
		return false;

	if (ret_value)
		setup_example(current_example);
	else
		return false;

	current_label = current_example->ld->label;
	current_length = current_example->num_features;

	return ret_value;
}

VwExample* CStreamingVwFeatures::get_example()
{
	return current_example;
}

float64_t CStreamingVwFeatures::get_label()
{
	ASSERT(has_labels)

	return current_label;
}

void CStreamingVwFeatures::release_example()
{
	env->example_number++;
	env->weighted_examples += current_example->ld->weight;

	if (current_example->ld->label == FLT_MAX)
		env->weighted_labels += 0;
	else
		env->weighted_labels += current_example->ld->label * current_example->ld->weight;

	env->total_features += current_example->num_features;
	env->sum_loss += current_example->loss;

	current_example->reset_members();
	parser.finalize_example();
}

int32_t CStreamingVwFeatures::get_dim_feature_space() const
{
	return current_length;
}

float32_t CStreamingVwFeatures::dot(CStreamingDotFeatures* df)
{
	SG_NOTIMPLEMENTED
	return CMath::INFTY;
}

float32_t CStreamingVwFeatures::dense_dot(VwExample* &ex, const float32_t* vec2)
{
	float32_t ret = 0.;
	for (vw_size_t* i = ex->indices.begin; i!= ex->indices.end; i++)
	{
		for (VwFeature* f = ex->atomics[*i].begin; f != ex->atomics[*i].end; f++)
			ret += vec2[f->weight_index & env->thread_mask] * f->x;
	}
	return ret;
}

float32_t CStreamingVwFeatures::dense_dot(const float32_t* vec2, int32_t vec2_len)
{
	return dense_dot(current_example, vec2);
}

float32_t CStreamingVwFeatures::dense_dot(SGSparseVector<float32_t>* vec1, const float32_t* vec2)
{
	float32_t ret = 0.;
	for (int32_t i = 0; i < vec1->num_feat_entries; i++)
		ret += vec1->features[i].entry * vec2[vec1->features[i].feat_index & env->mask];

	return ret;
}

float32_t CStreamingVwFeatures::dense_dot_truncated(const float32_t* vec2, VwExample* &ex, float32_t gravity)
{
	float32_t ret = 0.;
	for (vw_size_t* i = ex->indices.begin; i != ex->indices.end; i++)
	{
		for (VwFeature* f = ex->atomics[*i].begin; f!= ex->atomics[*i].end; f++)
		{
			float32_t w = vec2[f->weight_index & env->thread_mask];
			float32_t wprime = real_weight(w,gravity);
			ret += wprime*f->x;
		}
	}

	return ret;
}

void CStreamingVwFeatures::add_to_dense_vec(float32_t alpha, VwExample* &ex, float32_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (abs_val)
	{
		for (vw_size_t* i = ex->indices.begin; i != ex->indices.end; i++)
		{
			for (VwFeature* f = ex->atomics[*i].begin; f != ex->atomics[*i].end; f++)
				vec2[f->weight_index & env->thread_mask] += alpha * abs(f->x);
		}
	}
	else
	{
		for (vw_size_t* i = ex->indices.begin; i != ex->indices.end; i++)
		{
			for (VwFeature* f = ex->atomics[*i].begin; f != ex->atomics[*i].end; f++)
				vec2[f->weight_index & env->thread_mask] += alpha * f->x;
		}
	}
}

void CStreamingVwFeatures::add_to_dense_vec(float32_t alpha, float32_t* vec2, int32_t vec2_len, bool abs_val)
{
	add_to_dense_vec(alpha, current_example, vec2, vec2_len, abs_val);
}

int32_t CStreamingVwFeatures::get_num_features()
{
	return current_length;
}

EFeatureClass CStreamingVwFeatures::get_feature_class() const
{
	return C_STREAMING_VW;
}
