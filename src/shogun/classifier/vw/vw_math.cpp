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

#include <classifier/vw/vw_math.h>

namespace shogun
{

float32_t sd_offset_add(float32_t* weights, vw_size_t mask, VwFeature* begin, VwFeature* end, vw_size_t offset)
{
	float32_t ret = 0.;
	for (VwFeature* f = begin; f!= end; f++)
		ret += weights[(f->weight_index + offset) & mask] * f->x;
	return ret;
}

float32_t sd_offset_truncadd(float32_t* weights, vw_size_t mask, VwFeature* begin, VwFeature* end, vw_size_t offset, float32_t gravity)
{
	float32_t ret = 0.;
	for (VwFeature* f = begin; f!= end; f++)
	{
		float32_t w = weights[(f->weight_index+offset) & mask];
		float32_t wprime = real_weight(w,gravity);
		ret += wprime*f->x;
	}
	return ret;
}

float32_t one_pf_quad_predict(float32_t* weights, VwFeature& f, v_array<VwFeature> &cross_features, vw_size_t mask)
{
	vw_size_t halfhash = quadratic_constant * f.weight_index;

	return f.x *
		sd_offset_add(weights, mask, cross_features.begin, cross_features.end, halfhash);
}

float32_t one_pf_quad_predict_trunc(float32_t* weights, VwFeature& f, v_array<VwFeature> &cross_features, vw_size_t mask, float32_t gravity)
{
	vw_size_t halfhash = quadratic_constant * f.weight_index;

	return f.x *
		sd_offset_truncadd(weights, mask, cross_features.begin, cross_features.end, halfhash, gravity);
}

}
