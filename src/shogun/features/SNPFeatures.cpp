/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/SNPFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Alphabet.h>
#include <shogun/lib/memory.h>

using namespace shogun;

CSNPFeatures::CSNPFeatures()
{
	SG_UNSTABLE("CSNPFeatures::CSNPFeatures()", "\n")

	strings = NULL;

	string_length = 0;
	num_strings = 0;
	w_dim = 0;

	normalization_const = 0.0;

	m_str_min = NULL;
	m_str_maj = NULL;
}

CSNPFeatures::CSNPFeatures(CStringFeatures<uint8_t>* str) : CDotFeatures(),
	m_str_min(NULL), m_str_maj(NULL)
{
	ASSERT(str)
	ASSERT(str->have_same_length())
	SG_REF(str);

	strings=str;
	string_length=str->get_max_vector_length();
	ASSERT((string_length & 1) == 0) // length divisible by 2
	w_dim=3*string_length/2;
	num_strings=str->get_num_vectors();
	CAlphabet* alpha=str->get_alphabet();
	ASSERT(alpha->get_alphabet()==SNP)
	SG_UNREF(alpha);

	obtain_base_strings();
	set_normalization_const();

}

CSNPFeatures::CSNPFeatures(const CSNPFeatures& orig)
	: CDotFeatures(orig), strings(orig.strings),
	normalization_const(orig.normalization_const),
	m_str_min(NULL), m_str_maj(NULL)
{
	SG_REF(strings);

	if (strings)
	{
		string_length=strings->get_max_vector_length();
		ASSERT((string_length & 1) == 0) // length divisible by 2
			w_dim=3*string_length;
		num_strings=strings->get_num_vectors();
	}
	else
	{
		string_length = 0;
		w_dim = 0;
		num_strings = 0;
	}

	obtain_base_strings();
}

CSNPFeatures::~CSNPFeatures()
{
	SG_UNREF(strings);
}

int32_t CSNPFeatures::get_dim_feature_space() const
{
	return w_dim;
}

int32_t CSNPFeatures::get_nnz_features_for_vector(int32_t num)
{
	return w_dim/3;
}

EFeatureType CSNPFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass CSNPFeatures::get_feature_class() const
{
	return C_WD;
}

int32_t CSNPFeatures::get_num_vectors() const
{
	return num_strings;
}

float64_t CSNPFeatures::get_normalization_const()
{
	return normalization_const;
}

void CSNPFeatures::set_minor_base_string(const char* str)
{
	m_str_min=(uint8_t*) get_strdup(str);
}

void CSNPFeatures::set_major_base_string(const char* str)
{
	m_str_maj=(uint8_t*) get_strdup(str);
}

char* CSNPFeatures::get_minor_base_string()
{
	return (char*) m_str_min;
}

char* CSNPFeatures::get_major_base_string()
{
	return (char*) m_str_maj;
}

float64_t CSNPFeatures::dot(int32_t idx_a, CDotFeatures* df, int32_t idx_b)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CSNPFeatures* sf=(CSNPFeatures*) df;

	int32_t alen, blen;
	bool free_avec, free_bvec;

	uint8_t* avec = strings->get_feature_vector(idx_a, alen, free_avec);
	uint8_t* bvec = sf->strings->get_feature_vector(idx_b, blen, free_bvec);

	ASSERT(alen==blen)
	if (alen!=string_length)
		SG_ERROR("alen (%d) !=string_length (%d)\n", alen, string_length)
	ASSERT(m_str_min)
	ASSERT(m_str_maj)

	float64_t total=0;

	for (int32_t i = 0; i<alen-1; i+=2)
	{
		int32_t sumaa=0;
		int32_t sumbb=0;
		int32_t sumab=0;

		uint8_t a1=avec[i];
		uint8_t a2=avec[i+1];
		uint8_t b1=bvec[i];
		uint8_t b2=bvec[i+1];

		if ((a1!=a2 || a1=='0' || a1=='0') && (b1!=b2 || b1=='0' || b2=='0'))
			sumab++;
		else if (a1==a2 && b1==b2)
		{
			if (a1!=b1)
				continue;

			if (a1==m_str_min[i])
				sumaa++;
			else if (a1==m_str_maj[i])
				sumbb++;
			else
			{
				SG_ERROR("The impossible happened i=%d a1=%c "
						"a2=%c b1=%c b2=%c min=%c maj=%c\n", i, a1,a2, b1,b2, m_str_min[i], m_str_maj[i]);
			}

		}
		total+=sumaa+sumbb+sumab;
	}

	strings->free_feature_vector(avec, idx_a, free_avec);
	sf->strings->free_feature_vector(bvec, idx_b, free_bvec);
	return total;
}

float64_t CSNPFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim)

	float64_t sum=0;
	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t offs=0;

	for (int32_t i=0; i<len; i+=2)
	{
		int32_t dim=0;

		char a1=vec[i];
		char a2=vec[i+1];

		if (a1==a2 && a1!='0' && a2!='0')
		{
			if (a1==m_str_min[i])
				dim=1;
			else if (a1==m_str_maj[i])
				dim=2;
			else
			{
				SG_ERROR("The impossible happened i=%d a1=%c a2=%c min=%c maj=%c\n",
						i, a1,a2, m_str_min[i], m_str_maj[i]);
			}
		}

		sum+=vec2[offs+dim];
		offs+=3;
	}
	strings->free_feature_vector(vec, vec_idx1, free_vec1);

	return sum/normalization_const;
}

void CSNPFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	if (vec2_len != w_dim)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, w_dim=%d\n", vec2_len, w_dim)

	int32_t len;
	bool free_vec1;
	uint8_t* vec = strings->get_feature_vector(vec_idx1, len, free_vec1);
	int32_t offs=0;

	if (abs_val)
		alpha=CMath::abs(alpha);

	for (int32_t i=0; i<len; i+=2)
	{
		int32_t dim=0;

		char a1=vec[i];
		char a2=vec[i+1];

		if (a1==a2 && a1!='0' && a2!='0')
		{
			if (a1==m_str_min[i])
				dim=1;
			else if (a1==m_str_maj[i])
				dim=2;
			else
			{
				SG_ERROR("The impossible happened i=%d a1=%c a2=%c min=%c maj=%c\n",
						i, a1,a2, m_str_min[i], m_str_maj[i]);
			}
		}

		vec2[offs+dim]+=alpha;
		offs+=3;
	}
	strings->free_feature_vector(vec, vec_idx1, free_vec1);
}

void CSNPFeatures::find_minor_major_strings(uint8_t* minor, uint8_t* major)
{
	for (int32_t i=0; i<num_strings; i++)
	{
		int32_t len;
		bool free_vec;
		uint8_t* vec = ((CStringFeatures<uint8_t>*) strings)->get_feature_vector(i, len, free_vec);
		ASSERT(string_length==len)

		for (int32_t j=0; j<len; j++)
		{
			// skip sequencing errors
			if (vec[j]=='0')
				continue;

			if (minor[j]==0)
				minor[j]=vec[j];
            else if (major[j]==0 && vec[j]!=minor[j])
				major[j]=vec[j];
		}

		((CStringFeatures<uint8_t>*) strings)->free_feature_vector(vec, i, free_vec);
	}
}

void CSNPFeatures::obtain_base_strings(CSNPFeatures* snp)
{
	SG_FREE(m_str_min);
	SG_FREE(m_str_maj);
	size_t tlen=(string_length+1)*sizeof(uint8_t);

	m_str_min=SG_CALLOC(uint8_t, tlen);
	m_str_maj=SG_CALLOC(uint8_t, tlen);

	find_minor_major_strings(m_str_min, m_str_maj);

	if (snp)
		snp->find_minor_major_strings(m_str_min, m_str_maj);

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

	SG_DEBUG("normalization_const:%f\n", normalization_const)
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

SGMatrix<float64_t> CSNPFeatures::get_histogram(bool normalize)
{
	int32_t nsym=3;
	float64_t* h= SG_CALLOC(float64_t, size_t(nsym)*string_length/2);

	float64_t* h_normalizer=SG_MALLOC(float64_t, string_length/2);
	memset(h_normalizer, 0, string_length/2*sizeof(float64_t));
	int32_t num_str=get_num_vectors();
	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len;
		bool free_vec;
		uint8_t* vec = strings->get_feature_vector(i, len, free_vec);

		for (int32_t j=0; j<len; j+=2)
		{
			int32_t dim=0;

			char a1=vec[j];
			char a2=vec[j+1];

			if (a1==a2 && a1!='0' && a2!='0')
			{
				if (a1==m_str_min[j])
					dim=1;
				else if (a1==m_str_maj[j])
					dim=2;
				else
				{
					SG_ERROR("The impossible happened j=%d a1=%c a2=%c min=%c maj=%c\n",
							j, a1,a2, m_str_min[j], m_str_maj[j]);
				}
			}

			h[int64_t(j/2)*nsym+dim]++;
			h_normalizer[j/2]++;
		}

		strings->free_feature_vector(vec, i, free_vec);
	}

	if (normalize)
	{
		for (int32_t i=0; i<string_length/2; i++)
		{
			for (int32_t j=0; j<nsym; j++)
			{
				if (h_normalizer && h_normalizer[i])
					h[int64_t(i)*nsym+j]/=h_normalizer[i];
			}
		}
	}
	SG_FREE(h_normalizer);

	return SGMatrix<float64_t>(h, nsym, string_length/2);
}

SGMatrix<float64_t> CSNPFeatures::get_2x3_table(CSNPFeatures* pos, CSNPFeatures* neg)
{

	ASSERT(pos->strings->get_max_vector_length() == neg->strings->get_max_vector_length())
	int32_t len=pos->strings->get_max_vector_length();

	float64_t* table=SG_MALLOC(float64_t, 3*2*len/2);

	SGMatrix<float64_t> p_hist=pos->get_histogram(false);
	SGMatrix<float64_t> n_hist=neg->get_histogram(false);


	for (int32_t i=0; i<3*len/2; i++)
	{
		table[2*i]=p_hist.matrix[i];
		table[2*i+1]=n_hist.matrix[i];
	}
	return SGMatrix<float64_t>(table, 2,3*len/2);
}
