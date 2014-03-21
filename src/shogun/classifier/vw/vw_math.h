/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VW_MATH_H__
#define _VW_MATH_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/classifier/vw/vw_common.h>

namespace shogun
{

/**
 * Get the truncated weight value
 *
 * @param w weight
 * @param gravity threshold for the weight
 *
 * @return truncated weight
 */
inline float32_t real_weight(float32_t w,float32_t gravity)
{
	float32_t wprime = 0.;
	if (gravity < fabsf(w))
		wprime = CMath::sign(w)*(fabsf(w) - gravity);
	return wprime;
}

/**
 * Dot product of feature vector with the weight vector
 * with an offset added to the feature indices.
 *
 * @param weights weight vector
 * @param mask mask
 * @param begin first feature of the vector
 * @param end last feature of the vector
 * @param offset index offset
 *
 * @return dot product
 */
float32_t sd_offset_add(float32_t* weights, vw_size_t mask, VwFeature* begin,
			VwFeature* end, vw_size_t offset);

/**
 * Dot product of feature vector with the weight vector
 * with an offset added to the feature indices.
 *
 * Weights are taken as the truncated weights.
 *
 * @param weights weights
 * @param mask mask
 * @param begin first feature of the vector
 * @param end last feature of the vector
 * @param offset index offset
 * @param gravity weight threshold value
 *
 * @return dot product
 */
float32_t sd_offset_truncadd(float32_t* weights, vw_size_t mask, VwFeature* begin,
			     VwFeature* end, vw_size_t offset, float32_t gravity);

/**
 * Get the prediction contribution from one feature.
 *
 * @param weights weights
 * @param f feature
 * @param cross_features paired features
 * @param mask mask
 *
 * @return prediction from one feature
 */
float32_t one_pf_quad_predict(float32_t* weights, VwFeature& f,
			      v_array<VwFeature> &cross_features, vw_size_t mask);

/**
 * Get the prediction contribution from one feature.
 *
 * Weights are taken as truncated weights.
 *
 * @param weights weights
 * @param f feature
 * @param cross_features paired features
 * @param mask mask
 * @param gravity weight threshold value
 *
 * @return prediction from one feature
 */
float32_t one_pf_quad_predict_trunc(float32_t* weights, VwFeature& f,
				    v_array<VwFeature> &cross_features,
				    vw_size_t mask, float32_t gravity);
}
#endif // _VW_MATH_H__
