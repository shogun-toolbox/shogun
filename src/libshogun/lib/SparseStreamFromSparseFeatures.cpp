/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/SparseStreamFromSparseFeatures.h"

using namespace shogun;

CSparseStreamFromSparseFeatures::CSparseStreamFromSparseFeatures()
	: CSparseStream()
{
	init();
}

CSparseStreamFromSparseFeatures::CSparseStreamFromSparseFeatures(CFeatures* feat)
	: CSparseStream()
{
	init();
	features=feat;
}

CSparseStreamFromSparseFeatures::CSparseStreamFromSparseFeatures(CFeatures* feat, float64_t* lab)
	: CSparseStream()
{
	init();
	features=feat;
	labels=lab;
}

CSparseStreamFromSparseFeatures::~CSparseStreamFromSparseFeatures()
{
}

void CSparseStreamFromSparseFeatures::init()
{
	features=NULL;
	labels=NULL;
	vector_num=0;
}

/* Functions to return the vector from the SparseFeatures object */
#define GET_VECTOR(fname, conv, sg_type)				\
	void CSparseStreamFromSparseFeatures::fname			\
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

GET_VECTOR(get_byte_vector, atoi, uint8_t)
GET_VECTOR(get_char_vector, atoi, char)
GET_VECTOR(get_int_vector, atoi, int32_t)
GET_VECTOR(get_shortreal_vector, atof, float32_t)
GET_VECTOR(get_real_vector, atof, float64_t)
GET_VECTOR(get_short_vector, atoi, int16_t)
GET_VECTOR(get_word_vector, atoi, uint16_t)
GET_VECTOR(get_int8_vector, atoi, int8_t)
GET_VECTOR(get_uint_vector, atoi, uint32_t)
GET_VECTOR(get_long_vector, atoi, int64_t)
GET_VECTOR(get_ulong_vector, atoi, uint64_t)
GET_VECTOR(get_longreal_vector, atoi, floatmax_t)
#undef GET_VECTOR

/* Functions to return the vector from the SparseFeatures object */
#define GET_VECTOR_AND_LABEL(fname, conv, sg_type)				\
	void CSparseStreamFromSparseFeatures::fname			\
	(SGSparseVectorEntry<sg_type>*& vector, int32_t& len, float64_t& label)	\
	{								\
		ASSERT(labels);						\
									\
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

GET_VECTOR_AND_LABEL(get_byte_vector_and_label, atoi, uint8_t)
GET_VECTOR_AND_LABEL(get_char_vector_and_label, atoi, char)
GET_VECTOR_AND_LABEL(get_int_vector_and_label, atoi, int32_t)
GET_VECTOR_AND_LABEL(get_shortreal_vector_and_label, atof, float32_t)
GET_VECTOR_AND_LABEL(get_real_vector_and_label, atof, float64_t)
GET_VECTOR_AND_LABEL(get_short_vector_and_label, atoi, int16_t)
GET_VECTOR_AND_LABEL(get_word_vector_and_label, atoi, uint16_t)
GET_VECTOR_AND_LABEL(get_int8_vector_and_label, atoi, int8_t)
GET_VECTOR_AND_LABEL(get_uint_vector_and_label, atoi, uint32_t)
GET_VECTOR_AND_LABEL(get_long_vector_and_label, atoi, int64_t)
GET_VECTOR_AND_LABEL(get_ulong_vector_and_label, atoi, uint64_t)
GET_VECTOR_AND_LABEL(get_longreal_vector_and_label, atoi, floatmax_t)
#undef GET_VECTOR_AND_LABEL
