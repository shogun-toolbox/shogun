/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/WDFeatures.h"
#include "lib/io.h"

CWDFeatures::CWDFeatures(CStringFeatures<uint8_t>* str,
		int32_t order) : CDotFeatures()
{
	ASSERT(str);
	ASSERT(str->have_same_length());
	strings=str;
	string_length=str->get_max_vector_length();

	degree=order;
	set_wd_weights();
}

CWDFeatures::~CWDFeatures()
{
}

float64_t CWDFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	return 0;
}

float64_t CWDFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
/*	ASSERT(features);
	if (!wd_weights)
		set_wd_weights();

	int32_t len=0;
	float64_t sum=0;
	uint8_t* vec=features->get_feature_vector(num, len);
	SG_INFO("len %d, string_length %d\n", len, string_length);
	ASSERT(len==string_length);

	for (int32_t j=0; j<string_length; j++)
	{
		int32_t offs=w_dim_single_char*j;
		int32_t val=0;
		for (int32_t k=0; (j+k<string_length) && (k<degree); k++)
		{
			val=val*alphabet_size + vec[j+k];
			sum+=wd_weights[k] * w[offs+val];
			offs+=w_offsets[k];
		}
	}
	return sum/normalization_const;
	*/
	return 0;
}

void CWDFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	/*
	int32_t lim=CMath::min(degree, string_length-j);
	int32_t len;

	for (int32_t k=0; k<lim; k++)
	{
		uint8_t* vec = f->get_feature_vector(j+k, len);
		float32_t wd = wd_weights[k]/normalization_const;

		for(uint32_t i=0; i < cut_length; i++) 
		{
			val[i]=val[i]*alphabet_size + vec[new_cut[i]];
			vec2[offs+val[i]]+=wd * y[new_cut[i]];
		}
		offs+=w_offsets[k];
	}
	*/
}

void CWDFeatures::set_wd_weights()
{
	ASSERT(degree>0 && degree<=8);
	delete[] wd_weights;
	wd_weights=new float64_t[degree];
	delete[] w_offsets;
	w_offsets=new int32_t[degree];
	w_dim=0;

	for (int32_t i=0; i<degree; i++)
	{
		w_offsets[i]=CMath::pow(alphabet_size, i+1);
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));
		w_dim+=w_offsets[i];
	}
}


void CWDFeatures::set_normalization_const()
{
	normalization_const=0;
	for (int32_t i=0; i<degree; i++)
		normalization_const+=(string_length-i)*wd_weights[i]*wd_weights[i];

	normalization_const=CMath::sqrt(normalization_const);
	SG_DEBUG("normalization_const:%f\n", normalization_const);
}
