/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "features/SNPFeatures.h"
#include "lib/io.h"

using namespace shogun;

CSNPFeatures::CSNPFeatures(CStringFeatures<uint8_t>* str,
		int32_t order, int32_t from_order) : CDotFeatures()
{
	ASSERT(str);
	ASSERT(str->have_same_length());
	SG_REF(str);

	strings=str;
	string_length=str->get_max_vector_length();
	num_strings=str->get_num_vectors();
	CAlphabet* alpha=str->get_alphabet();
	ASSERT(alpha->get_alphabet()==DIGIT2);
	SG_UNREF(alpha);

	set_normalization_const();

}

CSNPFeatures::CSNPFeatures(const CSNPFeatures& orig)
	: CDotFeatures(orig), strings(orig.strings),
	normalization_const(orig.normalization_const)
{
	SG_REF(strings);
	string_length=strings->get_max_vector_length();
	num_strings=strings->get_num_vectors();
	CAlphabet* alpha=strings->get_alphabet();
	SG_UNREF(alpha);
}

CSNPFeatures::~CSNPFeatures()
{
	SG_UNREF(strings);
}

float64_t CSNPFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CSNPFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim);

	float64_t sum=0;
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void CSNPFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim);

	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void CSNPFeatures::obtain_base_strings()
{
	string_length=0;

	for (int32_t i=0; i<num_strings; i++)
	{
		int32_t len;
		bool free_vec;
		uint8_t* vec = ((CStringFeatures<uint8_t>*) strings)->get_feature_vector(i, len, free_vec);

		if (string_length==0)
		{
			string_length=len;
			size_t tlen=(len+1)*sizeof(uint8_t);
			m_str_min=(uint8_t*) malloc(tlen);
			m_str_maj=(uint8_t*) malloc(tlen);
			memset(m_str_min, 0, tlen);
			memset(m_str_maj, 0, tlen);
		}
		else
		{
			ASSERT(string_length==len);
		}

		for (int32_t j=0; j<len; j++)
		{
			// skip sequencing errors
			if (vec[j]=='0')
				continue;

			if (m_str_min[j]==0)
				m_str_min[j]=vec[j];
            else if (m_str_maj[j]==0 && vec[j]!=m_str_min[j])
				m_str_maj[j]=vec[j];
		}

		((CStringFeatures<uint8_t>*) strings)->free_feature_vector(vec, i, free_vec);
	}

	for (int32_t j=0; j<string_length; j++)
	{
        // if only one symbol occurs use 0
		if (m_str_min[j]==0)
            m_str_min[j]='0';
		if (m_str_maj[j]==0)
            m_str_maj[j]='0';

		if (m_str_min[j]>m_str_maj[j])
			CMath::swap(m_str_min[j], m_str_maj[j]);
	}
}

void CSNPFeatures::set_normalization_const(float64_t n)
{
	if (n==0)
	{
		normalization_const=string_length;
		normalization_const=CMath::sqrt(normalization_const);
	}
	else
		normalization_const=n;

	SG_DEBUG("normalization_const:%f\n", normalization_const);
}

void* CSNPFeatures::get_feature_iterator(int32_t vector_index)
{
	return NULL;
}

bool CSNPFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	return false;
}

void CSNPFeatures::free_feature_iterator(void* iterator)
{
}

CFeatures* CSNPFeatures::duplicate() const
{
	return new CSNPFeatures(*this);
}
