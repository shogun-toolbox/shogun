/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/StreamingFileFromStringFeatures.h"

using namespace shogun;

CStreamingFileFromStringFeatures::CStreamingFileFromStringFeatures()
	: CStreamingFileFromFeatures()
{
	init();
}

CStreamingFileFromStringFeatures::CStreamingFileFromStringFeatures(CFeatures* feat)
	: CStreamingFileFromFeatures(feat)
{
	init();
}

CStreamingFileFromStringFeatures::CStreamingFileFromStringFeatures(CFeatures* feat, float64_t* lab)
	: CStreamingFileFromFeatures(feat,lab)
{
	init();
}

CStreamingFileFromStringFeatures::~CStreamingFileFromStringFeatures()
{
}

void CStreamingFileFromStringFeatures::init()
{
	vector_num=0;
}

/* Functions to return the vector from the StringFeatures object */
#define GET_STRING(fname, sg_type)					\
	void CStreamingFileFromStringFeatures::fname(sg_type*& vector, int32_t& num_feat) \
	{								\
		CStringFeatures<sg_type>* string_features=		\
			(CStringFeatures<sg_type>*) features;		\
									\
		if (vector_num >= string_features->get_num_vectors())	\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
		SGVector<sg_type> sg_vector=				\
			string_features->get_feature_vector(vector_num); \
									\
		vector = sg_vector.vector;				\
		num_feat = sg_vector.vlen;;				\
		vector_num++;						\
									\
	}								\
	
GET_STRING(get_bool_string, bool)
GET_STRING(get_byte_string, uint8_t)
GET_STRING(get_char_string, char)
GET_STRING(get_int_string, int32_t)
GET_STRING(get_shortreal_string, float32_t)
GET_STRING(get_real_string, float64_t)
GET_STRING(get_short_string, int16_t)
GET_STRING(get_word_string, uint16_t)
GET_STRING(get_int8_string, int8_t)
GET_STRING(get_uint_string, uint32_t)
GET_STRING(get_long_string, int64_t)
GET_STRING(get_ulong_string, uint64_t)
GET_STRING(get_longreal_string, floatmax_t)
#undef GET_STRING

/* Functions to return the vector from the StringFeatures object with label */
#define GET_STRING_AND_LABEL(fname, sg_type)				\
	void CStreamingFileFromStringFeatures::fname			\
	(sg_type*& vector, int32_t& num_feat, float64_t& label)		\
	{								\
		CStringFeatures<sg_type>* feat				\
			=(CStringFeatures<sg_type>*) features;		\
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

GET_STRING_AND_LABEL(get_bool_string_and_label, bool)
GET_STRING_AND_LABEL(get_byte_string_and_label, uint8_t)
GET_STRING_AND_LABEL(get_char_string_and_label, char)
GET_STRING_AND_LABEL(get_int_string_and_label, int32_t)
GET_STRING_AND_LABEL(get_shortreal_string_and_label, float32_t)
GET_STRING_AND_LABEL(get_real_string_and_label, float64_t)
GET_STRING_AND_LABEL(get_short_string_and_label, int16_t)
GET_STRING_AND_LABEL(get_word_string_and_label, uint16_t)
GET_STRING_AND_LABEL(get_int8_string_and_label, int8_t)
GET_STRING_AND_LABEL(get_uint_string_and_label, uint32_t)
GET_STRING_AND_LABEL(get_long_string_and_label, int64_t)
GET_STRING_AND_LABEL(get_ulong_string_and_label, uint64_t)
GET_STRING_AND_LABEL(get_longreal_string_and_label, floatmax_t)
#undef GET_STRING_AND_LABEL
