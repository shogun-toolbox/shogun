/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/features/RandomKitchenSinksDotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <typeinfo>

namespace shogun
{

class CRKSFunctions;

CRandomKitchenSinksDotFeatures::CRandomKitchenSinksDotFeatures()
	: CDotFeatures()
{
	init(NULL, 0);
}

CRandomKitchenSinksDotFeatures::CRandomKitchenSinksDotFeatures(
	CDotFeatures* dataset, int32_t K)
{
	init(dataset, K);
}

CRandomKitchenSinksDotFeatures::CRandomKitchenSinksDotFeatures(
	CDotFeatures* dataset, int32_t K, SGMatrix<float64_t> coeff)
{
	init(dataset, K);
	random_coeff = coeff;
}

SGMatrix<float64_t> CRandomKitchenSinksDotFeatures::generate_random_coefficients()
{
	SGVector<float64_t> vec = generate_random_parameter_vector();
	SGMatrix<float64_t> random_params(vec.vlen, num_samples);
	for (index_t dim=0; dim<random_params.num_rows; dim++)
		random_params(dim, 0) = vec[dim];

	for (index_t sample=1; sample<num_samples; sample++)
	{
		vec = generate_random_parameter_vector();
		for (index_t dim=0; dim<random_params.num_rows; dim++)
			random_params(dim, sample) = vec[dim];
	}
	return random_params;
}

CRandomKitchenSinksDotFeatures::CRandomKitchenSinksDotFeatures(CFile* loader)
{
	SG_NOTIMPLEMENTED
}

CRandomKitchenSinksDotFeatures::CRandomKitchenSinksDotFeatures(
	const CRandomKitchenSinksDotFeatures& orig)
{
	init(orig.feats, orig.num_samples);
	random_coeff = orig.random_coeff;
}

CRandomKitchenSinksDotFeatures::~CRandomKitchenSinksDotFeatures()
{
	SG_UNREF(feats);
}

void CRandomKitchenSinksDotFeatures::init(CDotFeatures* dataset,
	int32_t K)
{
	feats = dataset;
	SG_REF(feats);

	num_samples = K;

	SG_ADD((CSGObject** ) &feats, "feats", "Features to work on",
			MS_NOT_AVAILABLE);
	m_parameters->add(&random_coeff, "random_coeff", "Random function parameters");
}

int32_t CRandomKitchenSinksDotFeatures::get_dim_feature_space() const
{
	return num_samples;
}

float64_t CRandomKitchenSinksDotFeatures::dot(int32_t vec_idx1, CDotFeatures* df,
	int32_t vec_idx2)
{
	ASSERT(typeid(*this) == typeid(*df));
	CRandomKitchenSinksDotFeatures* other = (CRandomKitchenSinksDotFeatures* ) df;
	ASSERT(get_dim_feature_space()==other->get_dim_feature_space());

	float64_t dot_product = 0;
	for (index_t i=0; i<num_samples; i++)
	{
		float64_t tmp_dot_1 = dot(vec_idx1, i);
		float64_t tmp_dot_2 = other->dot(vec_idx2, i);

		tmp_dot_1 = post_dot(tmp_dot_1, i);
		tmp_dot_2 = other->post_dot(tmp_dot_2, i);
		dot_product += tmp_dot_1 * tmp_dot_2;
	}
	return dot_product;
}

float64_t CRandomKitchenSinksDotFeatures::dense_dot(
	int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	SG_DEBUG("entering dense_dot()\n");
	ASSERT(vec2_len == get_dim_feature_space());

	float64_t dot_product = 0;
	for (index_t i=0; i<num_samples; i++)
	{
		float64_t tmp_dot = dot(vec_idx1, i);
		tmp_dot = post_dot(tmp_dot, i);
		dot_product += tmp_dot * vec2[i];
	}
	SG_DEBUG("Leaving dense_dot()\n");
	return dot_product;
}

void CRandomKitchenSinksDotFeatures::add_to_dense_vec(float64_t alpha,
	int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	SG_DEBUG("Entering add_to_dense()\n");
	ASSERT(vec2_len == get_dim_feature_space());

	for (index_t i=0; i<num_samples; i++)
	{
		float64_t tmp_dot = dot(vec_idx1, i);
		tmp_dot = post_dot(tmp_dot, i);
		if (abs_val)
			vec2[i] += CMath::abs(alpha * tmp_dot);
		else
			vec2[i] += alpha * tmp_dot;
	}
	SG_DEBUG("Leaving add_to_dense()\n");
}

int32_t CRandomKitchenSinksDotFeatures::get_nnz_features_for_vector(int32_t num)
{
	return num_samples;
}

void* CRandomKitchenSinksDotFeatures::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool CRandomKitchenSinksDotFeatures::get_next_feature(int32_t& index,
	float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED
	return false;
}

void CRandomKitchenSinksDotFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED
}

EFeatureType CRandomKitchenSinksDotFeatures::get_feature_type() const
{
	return F_DREAL;
}

EFeatureClass CRandomKitchenSinksDotFeatures::get_feature_class() const
{
	return C_DENSE;
}

int32_t CRandomKitchenSinksDotFeatures::get_num_vectors() const
{
	return feats->get_num_vectors();
}

const char* CRandomKitchenSinksDotFeatures::get_name() const
{
	return "RandomKitchenSinksDotFeatures";
}

CFeatures* CRandomKitchenSinksDotFeatures::duplicate() const
{
	SG_NOTIMPLEMENTED
	return NULL;
}

SGMatrix<float64_t> CRandomKitchenSinksDotFeatures::get_random_coefficients()
{
	return random_coeff;
}

float64_t CRandomKitchenSinksDotFeatures::dot(index_t vec_idx, index_t par_idx)
{
	return feats->dense_dot(vec_idx, random_coeff.get_column_vector(par_idx),
					feats->get_dim_feature_space());
}

float64_t CRandomKitchenSinksDotFeatures::post_dot(float64_t dot_result, index_t par_idx)
{
	return dot_result;
}

}
