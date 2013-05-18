/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/ImplicitWeightedSpecFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CImplicitWeightedSpecFeatures::CImplicitWeightedSpecFeatures()
	:CDotFeatures()
{
	SG_UNSTABLE("CImplicitWeightedSpecFeatures::"
				"CImplicitWeightedSpecFeatures()", "\n");

	strings = NULL;
	normalization_factors = NULL;
	num_strings = 0;
	alphabet_size = 0;

	degree = 0;
	spec_size = 0;
	spec_weights = 0;
}

CImplicitWeightedSpecFeatures::CImplicitWeightedSpecFeatures(CStringFeatures<uint16_t>* str, bool normalize) : CDotFeatures()
{
	ASSERT(str)
	strings=str;
	SG_REF(strings)
	normalization_factors=NULL;
	spec_weights=NULL;
	num_strings = str->get_num_vectors();
	alphabet_size = str->get_original_num_symbols();
	degree=str->get_order();
	set_wd_weights();

	SG_DEBUG("WEIGHTED SPEC alphasz=%d, size=%d, num_str=%d\n", alphabet_size,
			spec_size, num_strings);

	if (normalize)
		compute_normalization_const();
}

void CImplicitWeightedSpecFeatures::compute_normalization_const()
{
	float64_t* factors=SG_MALLOC(float64_t, num_strings);

	for (int32_t i=0; i<num_strings; i++)
		factors[i]=1.0/CMath::sqrt(dot(i, this, i));

	normalization_factors=factors;
	//CMath::display_vector(normalization_factors, num_strings, "n");
}

bool CImplicitWeightedSpecFeatures::set_wd_weights()
{
	SG_FREE(spec_weights);
	spec_weights=SG_MALLOC(float64_t, degree);

	int32_t i;
	float64_t sum=0;
	spec_size=0;

	for (i=0; i<degree; i++)
	{
		spec_size+=CMath::pow(alphabet_size, i+1);
		spec_weights[i]=degree-i;
		sum+=spec_weights[i];
	}
	for (i=0; i<degree; i++)
		spec_weights[i]=CMath::sqrt(spec_weights[i]/sum);

	return spec_weights!=NULL;
}

bool CImplicitWeightedSpecFeatures::set_weights(float64_t* w, int32_t d)
{
	ASSERT(d==degree)

	SG_FREE(spec_weights);
	spec_weights=SG_MALLOC(float64_t, degree);
	for (int32_t i=0; i<degree; i++)
		spec_weights[i]=CMath::sqrt(w[i]);
	return true;
}

CImplicitWeightedSpecFeatures::CImplicitWeightedSpecFeatures(const CImplicitWeightedSpecFeatures& orig) : CDotFeatures(orig),
	num_strings(orig.num_strings),
	alphabet_size(orig.alphabet_size), spec_size(orig.spec_size)
{
	SG_NOTIMPLEMENTED
	SG_REF(strings);
}

CImplicitWeightedSpecFeatures::~CImplicitWeightedSpecFeatures()
{
	SG_UNREF(strings);
	SG_FREE(spec_weights);
	SG_FREE(normalization_factors);
}

float64_t CImplicitWeightedSpecFeatures::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CImplicitWeightedSpecFeatures* sf = (CImplicitWeightedSpecFeatures*) df;

	ASSERT(vec_idx1 < num_strings)
	ASSERT(vec_idx2 < sf->get_num_vectors())

	int32_t len1=-1;
	int32_t len2=-1;
	bool free_vec1;
	bool free_vec2;
	uint16_t* vec1=strings->get_feature_vector(vec_idx1, len1, free_vec1);
	uint16_t* vec2=sf->strings->get_feature_vector(vec_idx2, len2, free_vec2);

	float64_t result=0;
	uint8_t mask=0;

	for (int32_t d=0; d<degree; d++)
	{
		mask = mask | (1 << (degree-d-1));
		uint16_t masked=strings->get_masked_symbols(0xffff, mask);

		int32_t left_idx=0;
		int32_t right_idx=0;
		float64_t weight=spec_weights[d]*spec_weights[d];

		while (left_idx < len1 && right_idx < len2)
		{
			uint16_t lsym=vec1[left_idx] & masked;
			uint16_t rsym=vec2[right_idx] & masked;

			if (lsym == rsym)
			{
				int32_t old_left_idx=left_idx;
				int32_t old_right_idx=right_idx;

				while (left_idx<len1 && (vec1[left_idx] & masked) ==lsym)
					left_idx++;

				while (right_idx<len2 && (vec2[right_idx] & masked) ==lsym)
					right_idx++;

				result+=weight*(left_idx-old_left_idx)*(right_idx-old_right_idx);
			}
			else if (lsym<rsym)
				left_idx++;
			else
				right_idx++;
		}
	}

	strings->free_feature_vector(vec1, vec_idx1, free_vec1);
	sf->strings->free_feature_vector(vec2, vec_idx2, free_vec2);

	if (normalization_factors)
		return result*normalization_factors[vec_idx1]*normalization_factors[vec_idx2];
	else
		return result;
}

float64_t CImplicitWeightedSpecFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == spec_size)
	ASSERT(vec_idx1 < num_strings)

	float64_t result=0;
	int32_t len1=-1;
	bool free_vec1;
	uint16_t* vec1=strings->get_feature_vector(vec_idx1, len1, free_vec1);

	if (vec1 && len1>0)
	{
		for (int32_t j=0; j<len1; j++)
		{
			uint8_t mask=0;
			int32_t offs=0;
			uint16_t v=*vec1++;

			for (int32_t d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				int32_t idx=strings->get_masked_symbols(v, mask);
				idx=strings->shift_symbol(idx, degree-d-1);
				result += vec2[offs + idx]*spec_weights[d];
				offs+=strings->shift_offset(1,d+1);
			}
		}

		strings->free_feature_vector(vec1, vec_idx1, free_vec1);

		if (normalization_factors)
			result*=normalization_factors[vec_idx1];
	}
	else
		SG_ERROR("huh?\n")

	return result;
}

void CImplicitWeightedSpecFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	int32_t len1=-1;
	bool free_vec1;
	uint16_t* vec=strings->get_feature_vector(vec_idx1, len1, free_vec1);

	if (normalization_factors)
		alpha*=normalization_factors[vec_idx1];

	if (vec && len1>0)
	{
		for (int32_t j=0; j<len1; j++)
		{
			uint8_t mask=0;
			int32_t offs=0;
			for (int32_t d=0; d<degree; d++)
			{
				mask = mask | (1 << (degree-d-1));
				int32_t idx=strings->get_masked_symbols(vec[j], mask);
				idx=strings->shift_symbol(idx, degree-d-1);
				if (abs_val)
					vec2[offs + idx] += CMath::abs(alpha*spec_weights[d]);
				else
					vec2[offs + idx] += alpha*spec_weights[d];
				offs+=strings->shift_offset(1,d+1);
			}
		}
	}

	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

CFeatures* CImplicitWeightedSpecFeatures::duplicate() const
{
	return new CImplicitWeightedSpecFeatures(*this);
}

int32_t CImplicitWeightedSpecFeatures::get_dim_feature_space() const
{
	return spec_size;
}

void* CImplicitWeightedSpecFeatures::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=num_strings)
	{
		SG_ERROR("Index out of bounds (number of strings %d, you "
				"requested %d)\n", num_strings, vector_index);
	}

	wspec_feature_iterator* it=SG_MALLOC(wspec_feature_iterator, 1);
	it->vec= strings->get_feature_vector(vector_index, it->vlen, it->vfree);
	it->vidx=vector_index;

	it->offs=0;
	it->d=0;
	it->j=0;
	it->mask=0;
	it->alpha=normalization_factors[vector_index];

	return it;
}

bool CImplicitWeightedSpecFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	wspec_feature_iterator* it=(wspec_feature_iterator*) iterator;

	if (it->d>=degree)
	{
		if (it->j < it->vlen-1)
		{
			it->j++;
			it->d=0;
			it->mask=0;
			it->offs=0;
		}
		else
			return false;
	}

	int32_t d=it->d;

	it->mask = it->mask | (1 << (degree-d-1));
	int32_t idx=strings->get_masked_symbols(it->vec[it->j], it->mask);
	idx=strings->shift_symbol(idx, degree-d-1);
	value=it->alpha*spec_weights[d];
	index=it->offs + idx;
	it->offs+=strings->shift_offset(1,d+1);

	it->d=d+1;
	return true;
}

void CImplicitWeightedSpecFeatures::free_feature_iterator(void* iterator)
{
	ASSERT(iterator)
	wspec_feature_iterator* it=(wspec_feature_iterator*) iterator;
	strings->free_feature_vector(it->vec, it->vidx, it->vfree);
	SG_FREE(it);
}


int32_t CImplicitWeightedSpecFeatures::get_nnz_features_for_vector(int32_t num)
{
	int32_t vlen=-1;
	bool free_vec;
	uint16_t* vec1=strings->get_feature_vector(num, vlen, free_vec);
	strings->free_feature_vector(vec1, num, free_vec);
	int32_t nnz=0;
	for (int32_t i=1; i<=degree; i++)
		nnz+=CMath::min(CMath::pow(alphabet_size,i), vlen);
	return nnz;
}

EFeatureType CImplicitWeightedSpecFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass CImplicitWeightedSpecFeatures::get_feature_class() const
{
	return C_WEIGHTEDSPEC;
}

int32_t CImplicitWeightedSpecFeatures::get_num_vectors() const
{
	return num_strings;
}
