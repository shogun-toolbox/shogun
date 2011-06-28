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

CStreamingFileFromSimpleFeatures::CStreamingFileFromSimpleFeatures(void)
	: CStreamingFileFromFeatures()
{
	init();
}

CStreamingFileFromSimpleFeatures::CStreamingFileFromSimpleFeatures(CFeatures* feat)
	: CStreamingFileFromFeatures(feat)
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
	
GET_VECTOR(get_bool_vector, bool)
GET_VECTOR(get_byte_vector, uint8_t)
GET_VECTOR(get_char_vector, char)
GET_VECTOR(get_int_vector, int32_t)
GET_VECTOR(get_shortreal_vector, float32_t)
GET_VECTOR(get_real_vector, float64_t)
GET_VECTOR(get_short_vector, int16_t)
GET_VECTOR(get_word_vector, uint16_t)
GET_VECTOR(get_int8_vector, int8_t)
GET_VECTOR(get_uint_vector, uint32_t)
GET_VECTOR(get_long_vector, int64_t)
GET_VECTOR(get_ulong_vector, uint64_t)
GET_VECTOR(get_longreal_vector, floatmax_t)
#undef GET_VECTOR

