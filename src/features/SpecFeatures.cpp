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

CSpecFeatures::CSpecFeatures(CStringFeatures<uint16_t>* str) : CDotFeatures()
{
	ASSERT(str);

	num_strings = str->get_num_vectors();
	spec_size = str->get_num_symbols();

	obtain_kmer_spectrum(str);
}

CSpecFeatures::~CSpecFeatures()
{
	delete_kmer_spectrum();
}

float64_t CSpecFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	ASSERT(vec_idx1 < num_strings);
	ASSERT(vec_idx2 < num_strings);
	int32_t* vec1=k_spectrum[vec_idx1];
	int32_t* vec2=k_spectrum[vec_idx2];

	return CMath::dot(vec1, vec2, spec_size);
}

float64_t CSpecFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == spec_size);
	ASSERT(vec_idx1 < num_strings);
	int32_t* vec1=k_spectrum[vec_idx1];
	float64_t result=0;
	
	for (int32_t i=0; i<spec_size; i++)
		result+=vec1[i]*vec2[i];

	return result;
}

void CSpecFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == spec_size);
	ASSERT(vec_idx1 < num_strings);
	int32_t* vec1=k_spectrum[vec_idx1];

	if (abs_val)
	{
		for (int32_t i=0; i<spec_size; i++)
			vec2[i]+=alpha*CMath::abs(vec1[i]);
	}
	else
	{
		for (int32_t i=0; i<spec_size; i++)
			vec2[i]+=alpha*vec1[i];
	}
}

void CSpecFeatures::obtain_kmer_spectrum(CStringFeatures<uint16_t>* str)
{
	k_spectrum= new int32_t*[num_strings];

	for (int32_t i=0; i<num_strings; i++)
	{
		k_spectrum[i]=new int32_t[spec_size];
		memset(k_spectrum[i], 0, sizeof(int32_t)*spec_size);

		int32_t len=0;
		uint16_t* fv=str->get_feature_vector(i, len);

		for (int32_t j=0; j<len; j++)
			k_spectrum[i][fv[j]]++;
	}
}

void CSpecFeatures::delete_kmer_spectrum()
{
	for (int32_t i=0; i<num_strings; i++)
		delete[] k_spectrum[i];

	delete[] k_spectrum;
	k_spectrum=NULL;
}
