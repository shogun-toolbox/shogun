/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Viktor Gal
 * Copyright (C) 2012 Viktor Gal
 */

#include <shogun/features/LatentFeatures.h>

using namespace shogun;

CLatentFeatures::CLatentFeatures ()
{
	
}
	
CLatentFeatures::~CLatentFeatures ()
{
	
}

CFeatures* CLatentFeatures::duplicate () const
{
	return new CLatentFeatures (*this);
}

EFeatureType CLatentFeatures::get_feature_type ()
{
	
	return F_ANY;
}

EFeatureClass CLatentFeatures::get_feature_class ()
{
	
	return C_LATENT;
}
			
			
int32_t CLatentFeatures::get_num_vectors () const
{
	
	return 0;
}

int32_t CLatentFeatures::get_size ()
{
	
	return 0;
}

int32_t CLatentFeatures::get_dim_feature_space () const
{
	return 0;
}
			
float64_t CLatentFeatures::dot (int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
{
	
	return 0.0;
}
			
float64_t CLatentFeatures::dense_dot (int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
{
	return 0.0;
}

void CLatentFeatures::add_to_dense_vec (float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val)
{
	
}

int32_t CLatentFeatures::get_nnz_features_for_vector (int32_t num)
{
	return 0;
}

bool CLatentFeatures::get_next_feature (int32_t& index, float64_t& value, void* iterator)
{
	return false;
}

void CLatentFeatures::free_feature_iterator (void* iterator)
{
	
}

void* CLatentFeatures::get_feature_iterator (int32_t vector_index)
{
	return NULL;
}
