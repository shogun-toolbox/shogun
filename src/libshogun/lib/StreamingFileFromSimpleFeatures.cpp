/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/StreamingFileFromSimpleFeatures.h"

using namespace shogun;

CStreamingFileFromSimpleFeatures::CStreamingFileFromSimpleFeatures()
	: CStreamingFileFromFeatures()
{
	init();
}

CStreamingFileFromSimpleFeatures::CStreamingFileFromSimpleFeatures(CFeatures* feat)
	: CStreamingFileFromFeatures(feat)
{
	init();
}

CStreamingFileFromSimpleFeatures::CStreamingFileFromSimpleFeatures(CFeatures* feat, float64_t* lab)
	: CStreamingFileFromFeatures(feat,lab)
{
	init();
}

CStreamingFileFromSimpleFeatures::~CStreamingFileFromSimpleFeatures()
{
}

void CStreamingFileFromSimpleFeatures::init()
{
	vector_num=0;
}

/* Functions to return the vector from the SimpleFeatures object */
#define GET_VECTOR(fname, sg_type)					\
	void CStreamingFileFromSimpleFeatures::fname(sg_type*& vector, int32_t& num_feat) \
	{								\
		CSimpleFeatures<sg_type>* simple_features=		\
			(CSimpleFeatures<sg_type>*) features;		\
									\
		if (vector_num >= simple_features->get_num_vectors())	\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
		SGVector<sg_type> sg_vector=				\
			simple_features->get_feature_vector(vector_num); \
									\
		vector = sg_vector.vector;				\
		num_feat = sg_vector.vlen;;				\
		vector_num++;						\
									\
	}								\
	
GET_VECTOR(get_vector, bool)
GET_VECTOR(get_vector, uint8_t)
GET_VECTOR(get_vector, char)
GET_VECTOR(get_vector, int32_t)
GET_VECTOR(get_vector, float32_t)
GET_VECTOR(get_vector, float64_t)
GET_VECTOR(get_vector, int16_t)
GET_VECTOR(get_vector, uint16_t)
GET_VECTOR(get_int8_vector, int8_t)
GET_VECTOR(get_uint_vector, uint32_t)
GET_VECTOR(get_long_vector, int64_t)
GET_VECTOR(get_ulong_vector, uint64_t)
GET_VECTOR(get_longreal_vector, floatmax_t)
#undef GET_VECTOR

/* Functions to return the vector from the SimpleFeatures object with label */
#define GET_VECTOR_AND_LABEL(fname, sg_type)				\
	void CStreamingFileFromSimpleFeatures::fname			\
	(sg_type*& vector, int32_t& num_feat, float64_t& label)		\
	{								\
		CSimpleFeatures<sg_type>* feat				\
			=(CSimpleFeatures<sg_type>*) features;		\
									\
		if (vector_num >= feat->get_num_vectors())		\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
		SGVector<sg_type> sg_vector				\
			=feat->get_feature_vector(vector_num);		\
									\
		vector = sg_vector.vector;				\
		num_feat = sg_vector.vlen;				\
		label = labels[vector_num];				\
									\
		vector_num++;						\
	}								\

GET_VECTOR_AND_LABEL(get_bool_vector_and_label, bool)
GET_VECTOR_AND_LABEL(get_byte_vector_and_label, uint8_t)
GET_VECTOR_AND_LABEL(get_char_vector_and_label, char)
GET_VECTOR_AND_LABEL(get_int_vector_and_label, int32_t)
GET_VECTOR_AND_LABEL(get_shortreal_vector_and_label, float32_t)
GET_VECTOR_AND_LABEL(get_real_vector_and_label, float64_t)
GET_VECTOR_AND_LABEL(get_short_vector_and_label, int16_t)
GET_VECTOR_AND_LABEL(get_word_vector_and_label, uint16_t)
GET_VECTOR_AND_LABEL(get_int8_vector_and_label, int8_t)
GET_VECTOR_AND_LABEL(get_uint_vector_and_label, uint32_t)
GET_VECTOR_AND_LABEL(get_long_vector_and_label, int64_t)
GET_VECTOR_AND_LABEL(get_ulong_vector_and_label, uint64_t)
GET_VECTOR_AND_LABEL(get_longreal_vector_and_label, floatmax_t)
#undef GET_VECTOR_AND_LABEL
