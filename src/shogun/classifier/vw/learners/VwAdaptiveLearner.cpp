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

#include <classifier/vw/learners/VwAdaptiveLearner.h>

using namespace shogun;

CVwAdaptiveLearner::CVwAdaptiveLearner()
	: CVwLearner()
{
}

CVwAdaptiveLearner::CVwAdaptiveLearner(CVwRegressor* regressor, CVwEnvironment* vw_env)
	: CVwLearner(regressor, vw_env)
{
}

CVwAdaptiveLearner::~CVwAdaptiveLearner()
{
}

void CVwAdaptiveLearner::train(VwExample* &ex, float32_t update)
{
	if (fabs(update) == 0.)
		return;

	vw_size_t thread_num = 0;

	vw_size_t thread_mask = env->thread_mask;
	float32_t* weights = reg->weight_vectors[thread_num];

	float32_t g = reg->loss->get_square_grad(ex->final_prediction, ex->ld->label) * ex->ld->weight;
	vw_size_t ctr = 0;
	for (vw_size_t* i = ex->indices.begin; i != ex->indices.end; i++)
	{
		for (VwFeature *f = ex->atomics[*i].begin; f != ex->atomics[*i].end; f++)
		{
			float32_t* w = &weights[f->weight_index & thread_mask];
			w[1] += g * f->x * f->x;
			float32_t t = f->x * CMath::invsqrt(w[1]);
			w[0] += update * t;
		}
	}

	for (int32_t k = 0; k < env->pairs.get_num_elements(); k++)
	{
		char* i = env->pairs.get_element(k);

		v_array<VwFeature> temp = ex->atomics[(int32_t)(i[0])];
		temp.begin = ex->atomics[(int32_t)(i[0])].begin;
		temp.end = ex->atomics[(int32_t)(i[0])].end;
		for (; temp.begin != temp.end; temp.begin++)
			quad_update(weights, *temp.begin, ex->atomics[(int32_t)(i[1])], thread_mask, update, g, ex, ctr);
	}
}

void CVwAdaptiveLearner::quad_update(float32_t* weights, VwFeature& page_feature,
				     v_array<VwFeature> &offer_features, vw_size_t mask,
				     float32_t update, float32_t g, VwExample* ex, vw_size_t& ctr)
{
	vw_size_t halfhash = quadratic_constant * page_feature.weight_index;
	update *= page_feature.x;
	float32_t update2 = g * page_feature.x * page_feature.x;

	for (VwFeature* elem = offer_features.begin; elem != offer_features.end; elem++)
	{
		float32_t* w = &weights[(halfhash + elem->weight_index) & mask];
		w[1] += update2 * elem->x * elem->x;
		float32_t t = elem->x * CMath::invsqrt(w[1]);
		w[0] += update * t;
	}
}
