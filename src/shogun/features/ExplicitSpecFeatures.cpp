/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/features/ExplicitSpecFeatures.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

CExplicitSpecFeatures::CExplicitSpecFeatures() :CDotFeatures()
{
	SG_UNSTABLE("CExplicitSpecFeatures::CExplicitSpecFeatures()",
				"\n");

	use_normalization = false;
	num_strings = 0;
	alphabet_size = 0;

	spec_size = 0;
	k_spectrum = NULL;
}


CExplicitSpecFeatures::CExplicitSpecFeatures(CStringFeatures<uint16_t>* str, bool normalize) : CDotFeatures()
{
	ASSERT(str)

	use_normalization=normalize;
	num_strings = str->get_num_vectors();
	spec_size = str->get_num_symbols();

	obtain_kmer_spectrum(str);

	SG_DEBUG("SPEC size=%d, num_str=%d\n", spec_size, num_strings)
}

CExplicitSpecFeatures::CExplicitSpecFeatures(const CExplicitSpecFeatures& orig) : CDotFeatures(orig),
	num_strings(orig.num_strings), alphabet_size(orig.alphabet_size), spec_size(orig.spec_size)
{
	k_spectrum= SG_MALLOC(float64_t*, num_strings);
	for (int32_t i=0; i<num_strings; i++)
		k_spectrum[i]=SGVector<float64_t>::clone_vector(k_spectrum[i], spec_size);
}

CExplicitSpecFeatures::~CExplicitSpecFeatures()
{
	delete_kmer_spectrum();
}

int32_t CExplicitSpecFeatures::get_dim_feature_space() const
{
	return spec_size;
}

float64_t CExplicitSpecFeatures::dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())
	CExplicitSpecFeatures* sf = (CExplicitSpecFeatures*) df;

	ASSERT(vec_idx1 < num_strings)
	ASSERT(vec_idx2 < sf->num_strings)
	float64_t* vec1=k_spectrum[vec_idx1];
	float64_t* vec2=sf->k_spectrum[vec_idx2];

	return SGVector<float64_t>::dot(vec1, vec2, spec_size);
}

float64_t CExplicitSpecFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	ASSERT(vec2_len == spec_size)
	ASSERT(vec_idx1 < num_strings)
	float64_t* vec1=k_spectrum[vec_idx1];
	float64_t result=0;

	for (int32_t i=0; i<spec_size; i++)
		result+=vec1[i]*vec2[i];

	return result;
}

void CExplicitSpecFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(vec2_len == spec_size)
	ASSERT(vec_idx1 < num_strings)
	float64_t* vec1=k_spectrum[vec_idx1];

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

void CExplicitSpecFeatures::obtain_kmer_spectrum(CStringFeatures<uint16_t>* str)
{
	k_spectrum= SG_MALLOC(float64_t*, num_strings);

	for (int32_t i=0; i<num_strings; i++)
	{
		k_spectrum[i]=SG_MALLOC(float64_t, spec_size);
		memset(k_spectrum[i], 0, sizeof(float64_t)*spec_size);

		int32_t len=0;
		bool free_fv;
		uint16_t* fv=str->get_feature_vector(i, len, free_fv);

		for (int32_t j=0; j<len; j++)
			k_spectrum[i][fv[j]]++;

		str->free_feature_vector(fv, i, free_fv);

		if (use_normalization)
		{
			float64_t n=0;
			for (int32_t j=0; j<spec_size; j++)
				n+=CMath::sq(k_spectrum[i][j]);

			n=CMath::sqrt(n);

			for (int32_t j=0; j<spec_size; j++)
				k_spectrum[i][j]/=n;
		}
	}
}

void CExplicitSpecFeatures::delete_kmer_spectrum()
{
	for (int32_t i=0; i<num_strings; i++)
		SG_FREE(k_spectrum[i]);

	SG_FREE(k_spectrum);
	k_spectrum=NULL;
}

CFeatures* CExplicitSpecFeatures::duplicate() const
{
	return new CExplicitSpecFeatures(*this);
}



void* CExplicitSpecFeatures::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CExplicitSpecFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CExplicitSpecFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED
}

int32_t CExplicitSpecFeatures::get_nnz_features_for_vector(int32_t num)
{
	SG_NOTIMPLEMENTED
	return 0;
}

EFeatureType CExplicitSpecFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass CExplicitSpecFeatures::get_feature_class() const
{
	return C_SPEC;
}

int32_t CExplicitSpecFeatures::get_num_vectors() const
{
	return num_strings;
}
