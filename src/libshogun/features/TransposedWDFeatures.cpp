/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "features/TransposedWDFeatures.h"
#include "lib/io.h"

using namespace shogun;

CTransposedWDFeatures::CTransposedWDFeatures(CStringFeatures<uint8_t>* str,
		int32_t order, int32_t from_order) : CDotFeatures()
{
	ASSERT(str);
	ASSERT(str->have_same_length());
	SG_REF(str);

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

CTransposedWDFeatures::CTransposedWDFeatures(const CTransposedWDFeatures& orig)
	: CDotFeatures(orig), strings(orig.strings),
	degree(orig.degree), from_degree(orig.from_degree),
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

CTransposedWDFeatures::~CTransposedWDFeatures()
{
	SG_UNREF(strings);
	delete[] wd_weights;
}

float64_t CTransposedWDFeatures::dot(int32_t vec_idx1, int32_t vec_idx2)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

float64_t CTransposedWDFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	SG_NOTIMPLEMENTED;
	return 0;
}

void CTransposedWDFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	SG_NOTIMPLEMENTED;
}

void CTransposedWDFeatures::set_wd_weights()
{
	ASSERT(degree>0 && degree<=8);
	wd_weights=new float64_t[degree];
	w_dim=0;
	partial_w_dim=0;

	for (int32_t i=0; i<degree; i++)
	{
		partial_w_dim+=CMath::pow(alphabet_size, i+1);
		wd_weights[i]=sqrt(2.0*(from_degree-i)/(from_degree*(from_degree+1)));
	}
	w_dim=partial_w_dim*string_length;
	SG_DEBUG("created WDFeatures with d=%d (%d), alphabetsize=%d, dim=%d num=%d, len=%d\n", degree, from_degree, alphabet_size, w_dim, num_strings, string_length);
}


void CTransposedWDFeatures::set_normalization_const(float64_t n)
{
	if (n==0)
	{
		normalization_const=0;
		for (int32_t i=0; i<degree; i++)
			normalization_const+=(num_strings-i)*wd_weights[i]*wd_weights[i];

		normalization_const=CMath::sqrt(normalization_const);
	}
	else
		normalization_const=n;

	SG_DEBUG("normalization_const:%f\n", normalization_const);
}

void* CTransposedWDFeatures::get_feature_iterator(int32_t vector_index)
{
	if (vector_index>=num_strings)
	{
		SG_ERROR("Index out of bounds (number of strings %d, you "
				"requested %d)\n", num_strings, vector_index);
	}

	wd_feature_iterator* it=new wd_feature_iterator[1];

	int32_t lim=CMath::min(degree, num_strings);
	it->vec=new uint8_t*[lim];
	it->vidx=new int32_t[lim];
	it->vlen=new int32_t[lim];
	it->vfree=new bool[lim];
	it->lim=lim;

	for (int32_t i=0; i<lim; i++)
	{
		it->vec[i]=NULL;
		it->vidx[i]=NULL;
	}

	int32_t j=0;
	for (int32_t i=vector_index; i<vector_index+lim; i++)
	{
		it->vec[j]= strings->get_feature_vector(i, it->vlen[j], it->vfree[j]);
		it->vidx[j]=vector_index;
		j++;
	}

	it->val=new int32_t[it->vlen[0]];
	CMath::fill_vector(it->val, it->vlen[0], 0);

	it->asize=alphabet_size;
	it->asizem1=1;
	it->offs=0;
	it->k=0;
	it->i=0;
	it->o=0;

	return it;
}

bool CTransposedWDFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	wd_feature_iterator* it=(wd_feature_iterator*) iterator;

	if (it->i > it->vlen[0]-1)
		return false;

	int32_t i=it->i;
	int32_t k=it->k;
	SG_PRINT("i=%d k=%d offs=%d o=%d asize=%d asizem1=%d\n", i, k, it->offs, it->o, it->asize, it->asizem1);

	it->val[i]+=it->asizem1*it->vec[k][i];

	value=wd_weights[k]/normalization_const;
	index=it->val[i]+it->o;
	SG_PRINT("index=%d val=%f partial_w_dim=%d w_size=%d lim=%d vlen=%d\n", index, value, partial_w_dim, w_dim, it->lim, it->vlen[0]);

	it->o+=it->asize;
	it->asize*=alphabet_size;
	it->asizem1*=alphabet_size;
	it->k++;

	if (it->k >= it->lim-1)
	{
		it->i++;
		it->k=0;
		it->offs+=partial_w_dim;
		it->o=it->offs;
		it->asize=alphabet_size;
		it->asizem1=1;
	}

	return true;
}

void CTransposedWDFeatures::free_feature_iterator(void* iterator)
{
	ASSERT(iterator);
	wd_feature_iterator* it=(wd_feature_iterator*) iterator;
	for (int32_t i=0; i<it->lim; i++)
	{
		if (it->vec[i])
			strings->free_feature_vector(it->vec[i], it->vidx[i], it->vfree[i]);
	}
	delete[] it->vec;
	delete[] it->vidx;
	delete[] it->vlen;
	delete[] it->vfree;
	delete[] it->val;
	delete[] it;
}

CFeatures* CTransposedWDFeatures::duplicate() const
{
	return new CTransposedWDFeatures(*this);
}
