/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "features/HashedWDFeaturesTransposed.h"
#include "lib/io.h"

using namespace shogun;

CHashedWDFeaturesTransposed::CHashedWDFeaturesTransposed(CStringFeatures<uint8_t>* str,
		int32_t start_order, int32_t order, int32_t from_order,
		int32_t hash_bits) : CDotFeatures()
{
	ASSERT(start_order>=0);
	ASSERT(start_order<order);
	ASSERT(order<=from_order);
	ASSERT(hash_bits>0);
	ASSERT(str);
	ASSERT(str->have_same_length());
	SG_REF(str);

	strings=str;
	int32_t transposed_num_feat=0;
	int32_t transposed_num_vec=0;
	transposed_strings=str->get_transposed(transposed_num_feat, transposed_num_vec);

	string_length=str->get_max_vector_length();
	num_strings=str->get_num_vectors();
	ASSERT(transposed_num_feat==num_strings);
	ASSERT(transposed_num_vec==string_length);

	CAlphabet* alpha=str->get_alphabet();
	alphabet_size=alpha->get_num_symbols();
	SG_UNREF(alpha);

	degree=order;
	start_degree=start_order;
	from_degree=from_order;
	m_hash_bits=hash_bits;
	set_wd_weights();
	set_normalization_const();
}

CHashedWDFeaturesTransposed::CHashedWDFeaturesTransposed(const CHashedWDFeaturesTransposed& orig)
	: CDotFeatures(orig), strings(orig.strings), transposed_strings(transposed_strings),
	degree(orig.degree), start_degree(orig.start_degree), 
	from_degree(orig.from_degree), m_hash_bits(orig.m_hash_bits),
	normalization_const(orig.normalization_const)
{
	SG_REF(strings);
	string_length=strings->get_max_vector_length();
	num_strings=strings->get_num_vectors();
	CAlphabet* alpha=strings->get_alphabet();
	alphabet_size=alpha->get_num_symbols();
	SG_UNREF(alpha);

	set_wd_weights();
}

CHashedWDFeaturesTransposed::~CHashedWDFeaturesTransposed()
{
	for (int32_t i=0; i<string_length; i++)
		delete[] transposed_strings[i].string;
	delete[] transposed_strings;

	SG_UNREF(strings);
	delete[] wd_weights;
}

float64_t CHashedWDFeaturesTransposed::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	int32_t len1, len2;
	bool free_vec1, free_vec2;

	uint8_t* vec1=strings->get_feature_vector(vec_idx1, len1, free_vec1);
	uint8_t* vec2=strings->get_feature_vector(vec_idx2, len2, free_vec2);

	ASSERT(len1==len2);

	float64_t sum=0.0;

	for (int32_t i=0; i<len1; i++)
	{
		for (int32_t j=0; (i+j<len1) && (j<degree); j++)
		{
			if (vec1[i+j]!=vec2[i+j])
				break;
			if (j>=start_degree)
				sum += wd_weights[j]*wd_weights[j];
		}
	}
	strings->free_feature_vector(vec1, vec_idx1, free_vec1);
	strings->free_feature_vector(vec2, vec_idx2, free_vec2);
	return sum;
}

float64_t CHashedWDFeaturesTransposed::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim);

	float64_t sum=0;
	int32_t lim=CMath::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	uint32_t* val=new uint32_t[len];

	uint32_t offs=0;

	if (start_degree>0)
	{
		// compute hash for strings of length start_degree-1
		for (int32_t i=0; i+start_degree < len; i++) 
			val[i]=CHash::MurmurHash2(&vec[i], start_degree, 0xDEADBEAF);
	}
	else
		CMath::fill_vector(val, len, 0xDEADBEAF);

	for (int32_t k=start_degree; k<lim; k++)
	{
		float64_t wd = wd_weights[k];

		uint32_t o=offs;
		for (int32_t i=0; i+k < len; i++) 
		{
			const uint32_t h=CHash::IncrementalMurmurHash2(vec[i+k], val[i]);
			val[i]=h;
#ifdef DEBUG_HASHEDWD
			SG_PRINT("vec[i]=%d, k=%d, offs=%d o=%d h=%d \n", vec[i], k,offs, o, h);
#endif
			sum+=vec2[o+(h & mask)]*wd;
			o+=partial_w_dim;
		}
		offs+=partial_w_dim*len;
	}
	delete[] val;
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void CHashedWDFeaturesTransposed::dense_dot_range(float64_t* output, int32_t start, int32_t stop, float64_t* alphas, float64_t* vec, int32_t vec_len, float64_t bias)
{
	if (vec_len != w_dim)
		SG_ERROR("Dimensions don't match, vec_len=%d, w_dim=%d\n", vec_len, w_dim);

	int32_t num_rows=degree-1;
	uint32_t* index=new uint32_t[num_strings*degree];
	CMath::fill_vector(&index[start], stop-start, 0xDEADBEAF);
	CMath::fill_vector(&output[start], stop-start, 0.0);

	uint32_t offs=0;
	for (int32_t i=0; i<string_length; i++)
	{
		uint32_t o=offs;

		for (int32_t k=0; k<degree && i+k<string_length; k++)
		{
			float64_t wd = wd_weights[k];
			uint8_t* dim=transposed_strings[i+k].string;
			uint32_t* cur_row=&index[num_strings*((i+k) % degree)];
			uint32_t* next_row=&index[num_strings*((i+k+1) % degree)];

			for (int32_t j=start; j<stop; j++)
			{
				const uint32_t h=CHash::IncrementalMurmurHash2(dim[j], cur_row[j]);
				next_row[j]=h;
				output[j]+=vec[o + (h & mask)]*wd;
			}
			o+=partial_w_dim;
		}
		offs+=partial_w_dim*degree;
		SG_PROGRESS(i, 0, string_length);
	}

	if (alphas)
	{
		for (int32_t j=start; j<stop; j++)
			output[j]=output[j]*alphas[j]/normalization_const+bias;
	}
	else
	{
		for (int32_t j=start; j<stop; j++)
			output[j]=output[j]/normalization_const+bias;
	}

	delete[] index;
}

void CHashedWDFeaturesTransposed::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim);

	int32_t lim=CMath::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	uint32_t* val=new uint32_t[len];

	uint32_t offs=0;

	if (start_degree>0)
	{
		// compute hash for strings of length start_degree-1
		for (int32_t i=0; i+start_degree < len; i++) 
			val[i]=CHash::MurmurHash2(&vec[i], start_degree, 0xDEADBEAF);
	}
	else
		CMath::fill_vector(val, len, 0xDEADBEAF);

	for (int32_t k=start_degree; k<lim; k++)
	{
		float64_t wd = alpha*wd_weights[k]/normalization_const;

		if (abs_val)
			wd=CMath::abs(wd);

		uint32_t o=offs;
		for (int32_t i=0; i+k < len; i++) 
		{
			const uint32_t h=CHash::IncrementalMurmurHash2(vec[i+k], val[i]);
			val[i]=h;

#ifdef DEBUG_HASHEDWD
			SG_PRINT("offs=%d o=%d h=%d \n", offs, o, h);
			SG_PRINT("vec[i]=%d, k=%d, offs=%d o=%d h=%d \n", vec[i], k,offs, o, h);
#endif
			vec2[o+(h & mask)]+=wd;
			o+=partial_w_dim;
		}
		offs+=partial_w_dim*len;
	}

	delete[] val;
	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void CHashedWDFeaturesTransposed::set_wd_weights()
{
	ASSERT(degree>0);

	mask=(uint32_t) (((uint64_t) 1)<<m_hash_bits)-1;
	partial_w_dim=1<<m_hash_bits;
	w_dim=partial_w_dim*string_length*(degree-start_degree);

	wd_weights=new float64_t[degree];

	for (int32_t i=0; i<degree; i++)
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));

	SG_DEBUG("created HashedWDFeaturesTransposed with d=%d (%d), alphabetsize=%d, "
			"dim=%d partial_dim=%d num=%d, len=%d\n", 
			degree, from_degree, alphabet_size, 
			w_dim, partial_w_dim, num_strings, string_length);
}


void CHashedWDFeaturesTransposed::set_normalization_const(float64_t n)
{
	if (n==0)
	{
		normalization_const=0;
		for (int32_t i=0; i<degree; i++)
			normalization_const+=(string_length-i)*wd_weights[i]*wd_weights[i];

		normalization_const=CMath::sqrt(normalization_const);
	}
	else
		normalization_const=n;

	SG_DEBUG("normalization_const:%f\n", normalization_const);
}

CFeatures* CHashedWDFeaturesTransposed::duplicate() const
{
	return new CHashedWDFeaturesTransposed(*this);
}
