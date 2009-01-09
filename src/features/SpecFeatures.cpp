/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/SpecFeatures.h"
#include "lib/io.h"

CSpecFeatures::CSpecFeatures(CStringFeatures<uint8_t>* str) : CDotFeatures()
{
	ASSERT(str);

	strings=str;
	string_length=str->get_max_vector_length();
}

CSpecFeatures::~CSpecFeatures()
{
}

float64_t CSpecFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	return 0;
}

float64_t CSpecFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	return 0;
}

void CSpecFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
}
