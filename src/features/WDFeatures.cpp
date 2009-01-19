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
		int32_t order, int32_t from_order) : CDotFeatures()
{
	ASSERT(str);
	ASSERT(str->have_same_length());
	SG_REF(strings);

	strings=str;
	string_length=str->get_max_vector_length();
	num_strings=str->get_num_vectors();
	CAlphabet* alpha=str->get_alphabet();
	alphabet_size=alpha->get_num_symbols();
	SG_UNREF(alpha);

	degree=order;
	from_degree=from_order;
	set_wd_weights();
	set_normalization_const();

}

CWDFeatures::CWDFeatures(const CWDFeatures& orig)
	: CDotFeatures(orig), strings(orig.strings),
	degree(orig.degree), from_degree(orig.from_degree)
{
	SG_REF(strings);
	string_length=strings->get_max_vector_length();
	num_strings=strings->get_num_vectors();
	CAlphabet* alpha=strings->get_alphabet();
	alphabet_size=alpha->get_num_symbols();
	SG_UNREF(alpha);

	set_wd_weights();
	set_normalization_const();
}

CWDFeatures::~CWDFeatures()
{
	SG_UNREF(strings);
}

float64_t CWDFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	int32_t len1, len2;
	uint8_t* vec1=strings->get_feature_vector(vec_idx1, len1);
	uint8_t* vec2=strings->get_feature_vector(vec_idx2, len2);

	ASSERT(len1==len2);

	float64_t sum=0.0;

	for (int32_t i=0; i<len1; i++)
	{
		for (int32_t j=0; (i+j<len1) && (j<degree); j++)
		{
			if (vec1[i+j]!=vec2[i+j])
				break ;
			sum += wd_weights[j];
		}
	}
	return sum;
}

float64_t CWDFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	float64_t sum=0;
	int32_t lim=CMath::min(degree, string_length);
	int32_t len;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len);
	int32_t* val=new int32_t[len];
	CMath::fill_vector(val, len, 0);

	for (int32_t k=0; k<lim; k++)
	{
		float64_t wd = wd_weights[k]/normalization_const;
		int32_t asize=alphabet_size;
		int32_t offs=0;

		for (int32_t i=0; i < len; i++) 
		{
			val[i]=val[i]*alphabet_size*len + alphabet_size*i + vec[i];
			sum+=vec2[offs+val[i]]*wd;
		}
		offs+=asize*len;
		asize*=alphabet_size;
	}
	delete[] val;
	return sum/normalization_const;
}

void CWDFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	int32_t lim=CMath::min(degree, string_length);
	int32_t len;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len);
	int32_t* val=new int32_t[len];
	CMath::fill_vector(val, len, 0);

	for (int32_t k=0; k<lim; k++)
	{
		float64_t wd = alpha*wd_weights[k]/normalization_const;
		int32_t asize=alphabet_size;
		int32_t offs=0;

		for (int32_t i=0; i < len; i++) 
		{
			val[i]=val[i]*alphabet_size*len + alphabet_size*i + vec[i];
			vec2[offs+val[i]]+=wd;
		}
		offs+=asize*len;
		asize*=alphabet_size;
	}
	delete[] val;
}

void CWDFeatures::set_wd_weights()
{
	ASSERT(degree>0 && degree<=8);
	wd_weights=new float64_t[degree];
	w_dim=0;

	for (int32_t i=0; i<degree; i++)
	{
		w_dim+=CMath::pow(alphabet_size, i+1)*string_length;
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));
	}
	SG_DEBUG("created WDFeatures with d=%d (%d), dim=%d num=%d, len=%d\n", degree, from_degree, w_dim, num_strings, string_length);
}


void CWDFeatures::set_normalization_const()
{
	normalization_const=0;
	for (int32_t i=0; i<degree; i++)
		normalization_const+=(string_length-i)*wd_weights[i]*wd_weights[i];

	normalization_const=CMath::sqrt(normalization_const);
	SG_DEBUG("normalization_const:%f\n", normalization_const);
}

CFeatures* CWDFeatures::duplicate() const
{
	return new CWDFeatures(*this);
}
