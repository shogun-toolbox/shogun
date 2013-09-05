/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 * Copyright (C) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/base/Parameter.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/RandomFourierDotFeatures.h>

namespace shogun {

CRandomFourierDotFeatures::CRandomFourierDotFeatures() : CDotFeatures()
{
	init(NULL, 0, SGMatrix<float64_t>(), SGVector<float64_t>());
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(CDotFeatures* feats,
	int32_t num_samples, KernelName name, SGVector<float64_t> params) : CDotFeatures()
{
	SGMatrix<float64_t> ww = CRandomFourierDotFeatures::generate_random_w(num_samples,
			feats->get_dim_feature_space(),	name, params);
	SGVector<float64_t> bb = CRandomFourierDotFeatures::generate_random_b(num_samples);
	init(feats, num_samples, ww, bb);
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(CDotFeatures* feats,
	int32_t num_samples, SGMatrix<float64_t> ww, SGVector<float64_t> bb) : CDotFeatures()
{
	init(feats, num_samples, ww, bb);
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(CFile* loader)
{
	SG_NOTIMPLEMENTED;
}

void CRandomFourierDotFeatures::init(CDotFeatures* feats, int32_t num_samples,
	SGMatrix<float64_t> ww, SGVector<float64_t> bb)
{
	dot_feats = feats;
	SG_REF(dot_feats);

	w = ww;
	b = bb;
	D = num_samples;

	SG_ADD((CSGObject** ) &dot_feats, "dot_feats", "Dot features to work on",
		MS_NOT_AVAILABLE);
	SG_ADD(&D, "D", "Dimension of random features", MS_NOT_AVAILABLE);
	m_parameters->add(&w, "w", "Multiplicative coefficients");
	m_parameters->add(&b, "b", "Additive coefficients");
}

SGVector<float64_t> CRandomFourierDotFeatures::generate_random_b(int32_t num_samples)
{
	SGVector<float64_t> vec(num_samples);
	for (index_t i=0; i<num_samples; i++)
		vec[i] = CMath::random(0.0, 2 * CMath::PI);
	return vec;
}

SGVector<float64_t> CRandomFourierDotFeatures::get_b() const
{
	return b;
}

SGMatrix<float64_t> CRandomFourierDotFeatures::get_w() const
{
	return w;
}

SGMatrix<float64_t> CRandomFourierDotFeatures::generate_random_w(int32_t num_samples, int32_t dim,
	KernelName kernel, SGVector<float64_t> params)
{
	SGMatrix<float64_t> mat(dim, num_samples);
	switch (kernel)
	{
		case GAUSSIAN:
			for (index_t i=0; i<dim; i++)
			{
				for (index_t j=0; j<num_samples; j++)
					mat(i,j) = CMath::sqrt((float64_t) 1/params[0]) * CMath::sqrt(2.0) * CMath::normal_random(0.0, 1);
			}
			break;
		default:
			SG_SERROR("Unknown kernel\n")
	}
	return mat;
}

CRandomFourierDotFeatures::CRandomFourierDotFeatures(const CRandomFourierDotFeatures& orig)
: CDotFeatures(orig)
{
	init(orig.dot_feats, orig.D, orig.w, orig.b);
}

CRandomFourierDotFeatures::~CRandomFourierDotFeatures()
{
	SG_UNREF(dot_feats);
}

CFeatures* CRandomFourierDotFeatures::duplicate() const
{
	return new CRandomFourierDotFeatures(*this);
}

int32_t CRandomFourierDotFeatures::get_dim_feature_space() const
{
	return D;
}

float64_t CRandomFourierDotFeatures::dot(int32_t vec_idx1, CDotFeatures* df,
	int32_t vec_idx2)
{
	if (strcmp(get_name(), df->get_name())!=0)
		SG_ERROR("Specified parameter in dot() is not of type \"%s\"\n", get_name())
	
	CRandomFourierDotFeatures* other = (CRandomFourierDotFeatures* ) df;	

	float64_t dot_product = 0;
	float64_t norm_val = (float64_t) 2 / D;
	for (index_t i=0; i<D; i++)
	{
		float64_t tmp_dot_1 = dot_feats->dense_dot(vec_idx1, w.get_column_vector(i), w.num_rows);
		float64_t tmp_dot_2 = other->dot_feats->dense_dot(vec_idx2, w.get_column_vector(i), w.num_rows);

		tmp_dot_1 = CMath::cos(tmp_dot_1 + b[i]);
		tmp_dot_2 = CMath::cos(tmp_dot_2 + b[i]);
		dot_product += norm_val * tmp_dot_1 * tmp_dot_2;
	}
	return dot_product;
}

float64_t CRandomFourierDotFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2,
	int32_t vec2_len)
{
	ASSERT(D == vec2_len)

	float64_t dot_product = 0;
	float64_t norm_val = CMath::sqrt((float64_t) 2 / D);
	for (index_t i=0; i<D; i++)
	{
		float64_t tmp_dot_1 = dot_feats->dense_dot(vec_idx1, w.get_column_vector(i), w.num_rows);
		tmp_dot_1 = CMath::cos(tmp_dot_1 + b[i]);
		dot_product += norm_val * tmp_dot_1 * vec2[i];
	}
	return dot_product;
}

void CRandomFourierDotFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
	float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	ASSERT(D == vec2_len)

	float64_t norm_val = CMath::sqrt((float64_t) 2 / D);
	for (index_t i=0; i<D; i++)
	{
		float64_t tmp_dot_1 = dot_feats->dense_dot(vec_idx1, w.get_column_vector(i), w.num_rows);
		tmp_dot_1 = CMath::cos(tmp_dot_1 + b[i]);

		if (abs_val)
			vec2[i] += CMath::abs(alpha * norm_val * tmp_dot_1);
		else
			vec2[i] += alpha * norm_val * tmp_dot_1;
	}
}

int32_t CRandomFourierDotFeatures::get_nnz_features_for_vector(int32_t num)
{
	return D;
}

void* CRandomFourierDotFeatures::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}
bool CRandomFourierDotFeatures::get_next_feature(int32_t& index, float64_t& value,
	void* iterator)
{
	SG_NOTIMPLEMENTED;
	return false;
}

void CRandomFourierDotFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED;
}

const char* CRandomFourierDotFeatures::get_name() const
{
	return "RandomFourierDotFeatures";
}

EFeatureType CRandomFourierDotFeatures::get_feature_type() const
{
	return F_DREAL;
}

EFeatureClass CRandomFourierDotFeatures::get_feature_class() const
{
	return C_DENSE;
}

int32_t CRandomFourierDotFeatures::get_num_vectors() const
{
	return dot_feats->get_num_vectors();
}

}
