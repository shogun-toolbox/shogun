/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/StreamingFileFromSparseFeatures.h"

using namespace shogun;

CStreamingFileFromSparseFeatures::CStreamingFileFromSparseFeatures()
	: CStreamingFileFromFeatures()
{
	init();
}

CStreamingFileFromSparseFeatures::CStreamingFileFromSparseFeatures(CFeatures* feat)
	: CStreamingFileFromFeatures(feat)
{
	init();
}

CStreamingFileFromSparseFeatures::CStreamingFileFromSparseFeatures(CFeatures* feat, float64_t* lab)
	: CStreamingFileFromFeatures(feat,lab)
{
	init();
}

CStreamingFileFromSparseFeatures::~CStreamingFileFromSparseFeatures()
{
}

void CStreamingFileFromSparseFeatures::init()
{
	vector_num=0;
}

/* Functions to return the vector from the SparseFeatures object */
#define GET_SPARSE_VECTOR(fname, sg_type)				\
	void CStreamingFileFromSparseFeatures::fname			\
	(SGSparseVectorEntry<sg_type>*& vector, int32_t& len)		\
	{								\
		CSparseFeatures<sg_type>* feat				\
			=(CSparseFeatures<sg_type>*) features;		\
									\
		if (vector_num >= feat->get_num_vectors())		\
		{							\
			vector=NULL;					\
			len=-1;						\
			return;						\
		}							\
									\
		bool vfree;						\
		vector=feat->get_sparse_feature_vector			\
			(vector_num, len, vfree);			\
									\
		vector_num++;						\
	}								\

GET_SPARSE_VECTOR(get_bool_sparse_vector, bool)
GET_SPARSE_VECTOR(get_byte_sparse_vector, uint8_t)
GET_SPARSE_VECTOR(get_char_sparse_vector, char)
GET_SPARSE_VECTOR(get_int_sparse_vector, int32_t)
GET_SPARSE_VECTOR(get_shortreal_sparse_vector, float32_t)
GET_SPARSE_VECTOR(get_real_sparse_vector, float64_t)
GET_SPARSE_VECTOR(get_short_sparse_vector, int16_t)
GET_SPARSE_VECTOR(get_word_sparse_vector, uint16_t)
GET_SPARSE_VECTOR(get_int8_sparse_vector, int8_t)
GET_SPARSE_VECTOR(get_uint_sparse_vector, uint32_t)
GET_SPARSE_VECTOR(get_long_sparse_vector, int64_t)
GET_SPARSE_VECTOR(get_ulong_sparse_vector, uint64_t)
GET_SPARSE_VECTOR(get_longreal_sparse_vector, floatmax_t)
#undef GET_SPARSE_VECTOR

/* Functions to return the vector from the SparseFeatures object */
#define GET_SPARSE_VECTOR_AND_LABEL(fname, sg_type)			\
	void CStreamingFileFromSparseFeatures::fname			\
	(SGSparseVectorEntry<sg_type>*& vector, int32_t& len, float64_t& label)	\
	{								\
		CSparseFeatures<sg_type>* feat				\
			=(CSparseFeatures<sg_type>*) features;		\
									\
		if (vector_num >= feat->get_num_vectors())		\
		{							\
			vector=NULL;					\
			len=-1;						\
			return;						\
		}							\
									\
		bool vfree;						\
		vector=feat->get_sparse_feature_vector			\
			(vector_num, len, vfree);			\
		label=labels[vector_num];				\
									\
		vector_num++;						\
	}								\

GET_SPARSE_VECTOR_AND_LABEL(get_bool_sparse_vector_and_label, bool)
GET_SPARSE_VECTOR_AND_LABEL(get_byte_sparse_vector_and_label, uint8_t)
GET_SPARSE_VECTOR_AND_LABEL(get_char_sparse_vector_and_label, char)
GET_SPARSE_VECTOR_AND_LABEL(get_int_sparse_vector_and_label, int32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_shortreal_sparse_vector_and_label, float32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_real_sparse_vector_and_label, float64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_short_sparse_vector_and_label, int16_t)
GET_SPARSE_VECTOR_AND_LABEL(get_word_sparse_vector_and_label, uint16_t)
GET_SPARSE_VECTOR_AND_LABEL(get_int8_sparse_vector_and_label, int8_t)
GET_SPARSE_VECTOR_AND_LABEL(get_uint_sparse_vector_and_label, uint32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_long_sparse_vector_and_label, int64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_ulong_sparse_vector_and_label, uint64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_longreal_sparse_vector_and_label, floatmax_t)
#undef GET_SPARSE_VECTOR_AND_LABEL
