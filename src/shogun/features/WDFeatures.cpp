/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/WDFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CWDFeatures::CWDFeatures() :CDotFeatures()
{
	SG_UNSTABLE("CWDFeatures::CWDFeatures() :CDotFeatures()",
				"\n");

	strings = NULL;

	degree = 0;
	from_degree = 0;
	string_length = 0;
	num_strings = 0;
	alphabet_size = 0;
	w_dim = 0;
	wd_weights = NULL;
	normalization_const = 0.0;
}

CWDFeatures::CWDFeatures(CStringFeatures<uint8_t>* str,
		int32_t order, int32_t from_order) : CDotFeatures()
{
	ASSERT(str)
	ASSERT(str->have_same_length())
	SG_REF(str);

	strings=str;
	string_length=str->get_max_vector_length();
	num_strings=str->get_num_vectors();
	CAlphabet* alpha=str->get_alphabet();
	alphabet_size=alpha->get_num_symbols();
	SG_UNREF(alpha);

	degree=order;
	from_degree=from_order;
	wd_weights=NULL;
	set_wd_weights();
	set_normalization_const();

}

CWDFeatures::~CWDFeatures()
{
	SG_UNREF(strings);
	SG_FREE(wd_weights);
}

float64_t CWDFeatures::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CWDFeatures* wdf = (CWDFeatures*) df;

	int32_t len1, len2;
	bool free_vec1, free_vec2;

	uint8_t* vec1=strings->get_feature_vector(vec_idx1, len1, free_vec1);
	uint8_t* vec2=wdf->strings->get_feature_vector(vec_idx2, len2, free_vec2);

	ASSERT(len1==len2)

	float64_t sum=0.0;

	for (int32_t i=0; i<len1; i++)
	{
		for (int32_t j=0; (i+j<len1) && (j<degree); j++)
		{
			if (vec1[i+j]!=vec2[i+j])
				break ;
			sum += wd_weights[j]*wd_weights[j];
		}
	}
	strings->free_feature_vector(vec1, vec_idx1, free_vec1);
	wdf->strings->free_feature_vector(vec2, vec_idx2, free_vec2);
	return sum/CMath::sq(normalization_const);
}

float64_t CWDFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim)

	float64_t sum=0;
	int32_t lim=CMath::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t* val=SG_MALLOC(int32_t, len);
	SGVector<int32_t>::fill_vector(val, len, 0);

	int32_t asize=alphabet_size;
	int32_t asizem1=1;
	int32_t offs=0;

	for (int32_t k=0; k<lim; k++)
	{
		float64_t wd = wd_weights[k];

		int32_t o=offs;
		for (int32_t i=0; i+k < len; i++)
		{
			val[i]+=asizem1*vec[i+k];
			sum+=vec2[val[i]+o]*wd;
			o+=asize;
		}
		offs+=asize*len;
		asize*=alphabet_size;
		asizem1*=alphabet_size;
	}
	SG_FREE(val);
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void CWDFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim)

	int32_t lim=CMath::min(degree, string_length);
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t* val=SG_MALLOC(int32_t, len);
	SGVector<int32_t>::fill_vector(val, len, 0);

	int32_t asize=alphabet_size;
	int32_t asizem1=1;
	int32_t offs=0;

	for (int32_t k=0; k<lim; k++)
	{
		float64_t wd = alpha*wd_weights[k]/normalization_const;

		if (abs_val)
			wd=CMath::abs(wd);

		int32_t o=offs;
		for (int32_t i=0; i+k < len; i++)
		{
			val[i]+=asizem1*vec[i+k];
			vec2[val[i]+o]+=wd;
			o+=asize;
		}
		offs+=asize*len;
		asize*=alphabet_size;
		asizem1*=alphabet_size;
	}
	SG_FREE(val);

	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void CWDFeatures::set_wd_weights()
{
	ASSERT(degree>0 && degree<=8)
	SG_FREE(wd_weights);
	wd_weights=SG_MALLOC(float64_t, degree);
	w_dim=0;

	for (int32_t i=0; i<degree; i++)
	{
		w_dim+=CMath::pow(alphabet_size, i+1)*string_length;
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));
	}
	SG_DEBUG("created WDFeatures with d=%d (%d), alphabetsize=%d, dim=%d num=%d, len=%d\n", degree, from_degree, alphabet_size, w_dim, num_strings, string_length)
}


void CWDFeatures::set_normalization_const(float64_t n)
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

	SG_DEBUG("normalization_const:%f\n", normalization_const)
}

void* CWDFeatures::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=num_strings)
	{
		SG_ERROR("Index out of bounds (number of strings %d, you "
				"requested %d)\n", num_strings, vector_index);
	}

	wd_feature_iterator* it=SG_MALLOC(wd_feature_iterator, 1);

	it->lim=CMath::min(degree, string_length);
	it->vec= strings->get_feature_vector(vector_index, it->vlen, it->vfree);
	it->vidx=vector_index;

	it->vec = strings->get_feature_vector(vector_index, it->vlen, it->vfree);
	it->val=SG_MALLOC(int32_t, it->vlen);
	SGVector<int32_t>::fill_vector(it->val, it->vlen, 0);

	it->asize=alphabet_size;
	it->asizem1=1;
	it->offs=0;
	it->k=0;
	it->i=0;
	it->o=0;

	return it;
}

bool CWDFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	wd_feature_iterator* it=(wd_feature_iterator*) iterator;

	if (it->i + it->k >= it->vlen)
	{
		if (it->k < it->lim-1)
		{
			it->offs+=it->asize*it->vlen;
			it->asize*=alphabet_size;
			it->asizem1*=alphabet_size;
			it->k++;
			it->i=0;
			it->o=it->offs;
		}
		else
			return false;
	}

	int32_t i=it->i;
	int32_t k=it->k;
#ifdef DEBUG_WDFEATURES
	SG_PRINT("i=%d k=%d offs=%d o=%d asize=%d asizem1=%d\n", i, k, it->offs, it->o, it->asize, it->asizem1)
#endif

	it->val[i]+=it->asizem1*it->vec[i+k];
	value=wd_weights[k]/normalization_const;
	index=it->val[i]+it->o;
#ifdef DEBUG_WDFEATURES
	SG_PRINT("index=%d val=%f w_size=%d lim=%d vlen=%d\n", index, value, w_dim, it->lim, it->vlen)
#endif

	it->o+=it->asize;
	it->i=i+1;

	return true;
}

void CWDFeatures::free_feature_iterator(void* iterator)
{
	ASSERT(iterator)
	wd_feature_iterator* it=(wd_feature_iterator*) iterator;
	strings->free_feature_vector(it->vec, it->vidx, it->vfree);
	SG_FREE(it->val);
	SG_FREE(it);
}

CFeatures* CWDFeatures::duplicate() const
{
	SG_NOTIMPLEMENTED
	// return new CWDFeatures(*this);
	return NULL;
}

int32_t CWDFeatures::get_dim_feature_space() const
{
	return w_dim;
}

int32_t CWDFeatures::get_nnz_features_for_vector(int32_t num)
{
	int32_t vlen=-1;
	bool free_vec;
	uint8_t* vec=strings->get_feature_vector(num, vlen, free_vec);
	strings->free_feature_vector(vec, num, free_vec);
	return degree*vlen;
}

EFeatureType CWDFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass CWDFeatures::get_feature_class() const
{
	return C_WD;
}

int32_t CWDFeatures::get_num_vectors() const
{
	return num_strings;
}

float64_t CWDFeatures::get_normalization_const()
{
	return normalization_const;
}

void CWDFeatures::set_wd_weights(SGVector<float64_t> weights)
{
	ASSERT(weights.vlen==degree)

	for (int32_t i=0; i<degree; i++)
		wd_weights[i]=weights.vector[i];
}

