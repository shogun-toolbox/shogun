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
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#include <classifier/vw/learners/VwNonAdaptiveLearner.h>

using namespace shogun;

CVwNonAdaptiveLearner::CVwNonAdaptiveLearner()
	: CVwLearner()
{
}

CVwNonAdaptiveLearner::CVwNonAdaptiveLearner(CVwRegressor* regressor, CVwEnvironment* vw_env)
	: CVwLearner(regressor, vw_env)
{
}

CVwNonAdaptiveLearner::~CVwNonAdaptiveLearner()
{
}

void CVwNonAdaptiveLearner::train(VwExample* &ex, float32_t update)
{
	if (fabs(update) == 0.)
		return;
	vw_size_t thread_mask = env->thread_mask;

	vw_size_t thread_num = 0;
	float32_t* weights = reg->weight_vectors[thread_num];

	for (vw_size_t* i = ex->indices.begin; i != ex->indices.end; i++)
	{
		for (VwFeature* f = ex->atomics[*i].begin; f != ex->atomics[*i].end; f++)
			weights[f->weight_index & thread_mask] += update * f->x;
	}

	for (int32_t k = 0; k < env->pairs.get_num_elements(); k++)
	{
		char* i = env->pairs.get_element(k);

		v_array<VwFeature> temp = ex->atomics[(int32_t)(i[0])];
		temp.begin = ex->atomics[(int32_t)(i[0])].begin;
		temp.end = ex->atomics[(int32_t)(i[0])].end;
		for (; temp.begin != temp.end; temp.begin++)
			quad_update(weights, *temp.begin, ex->atomics[(int32_t)(i[1])], thread_mask, update);
	}
}

void CVwNonAdaptiveLearner::quad_update(float32_t* weights, VwFeature& page_feature, v_array<VwFeature> &offer_features, vw_size_t mask, float32_t update)
{
	vw_size_t halfhash = quadratic_constant * page_feature.weight_index;
	update *= page_feature.x;
	for (VwFeature* elem = offer_features.begin; elem != offer_features.end; elem++)
		weights[(halfhash + elem->weight_index) & mask] += update * elem->x;
}
